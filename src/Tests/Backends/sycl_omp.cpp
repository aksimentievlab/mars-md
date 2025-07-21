#include <vector>
#include <numeric>
#include <iostream>

#include <cassert>
#include <cmath>

#ifdef USE_SYCL

#include "Backend/SYCL/SYCLManager.h"


using namespace ARBD::SYCL;


// Helper function for floating point comparison
bool within_tolerance(float a, float b, float tolerance = 1e-6f) {
    return std::fabs(a - b) <= tolerance;
}

void test_sycl_cpu_initialization() {
    std::cout << "[TEST] SYCL CPU Initialization... ";
    try {
        // Set CPU preference before initialization for AdaptiveCpp
        SYCLManager::prefer_device_type(sycl::info::device_type::cpu);
        
        // Initialize the SYCL manager
        SYCLManager::init();
        
        // Verify we have at least one device
        if (SYCLManager::all_device_size() == 0) {
            std::cout << "SKIP: No SYCL devices found\n";
            return;
        }
        
        // Verify AdaptiveCpp CPU device functionality
        try {
            auto platforms = sycl::platform::get_platforms();
            bool cpu_device_found = false;
            
            for (const auto& platform : platforms) {
                auto devices = platform.get_devices();
                for (const auto& device : devices) {
                    if (device.is_cpu()) {
                        // Test basic queue creation with in-order property (AdaptiveCpp recommendation)
                        sycl::queue test_queue{device, sycl::property::queue::in_order{}};
                        
                        // Simple functionality test
                        int test_value = 42;
                        int* device_ptr = sycl::malloc_device<int>(1, test_queue);
                        if (device_ptr) {
                            test_queue.memcpy(device_ptr, &test_value, sizeof(int)).wait();
                            int result = 0;
                            test_queue.memcpy(&result, device_ptr, sizeof(int)).wait();
                            sycl::free(device_ptr, test_queue);
                            
                            if (result == test_value) {
                                cpu_device_found = true;
                                break;
                            }
                        }
                    }
                }
                if (cpu_device_found) break;
            }
            
            if (!cpu_device_found) {
                std::cout << "SKIP: No functional CPU device available\n";
                return;
            }
        } catch (const sycl::exception& e) {
            std::cout << "SKIP: CPU device verification failed - " << e.what() << "\n";
            return;
        }
        
        // Display device information
        const auto& devices = SYCLManager::all_devices();
        bool found_cpu = false;
        for (const auto& device : devices) {
            if (device.is_cpu()) {
                found_cpu = true;
                std::cout << "\n  CPU Device " << device.id() << ": " << device.name()
                          << " (" << device.vendor() << ")\n";
                std::cout << "  - Compute units: " << device.max_compute_units() << "\n";
                std::cout << "  - Global memory: " << device.global_mem_size() / (1024*1024) << " MB\n";
                break; // Only show first CPU device to reduce output
            }
        }
        
        if (!found_cpu) {
            std::cout << "SKIP: No CPU device found in SYCLManager\n";
            return;
        }
        
        // Load device info and verify initialization
        SYCLManager::load_info();
        assert(SYCLManager::devices().size() > 0);
        std::cout << "PASS\n";
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_basic_memory_operations() {
    std::cout << "[TEST] SYCL Basic Memory Operations... ";
    try {
        auto& queue = SYCLManager::get_current_queue();
        
        // Test memory allocation and copy operations using USM
        constexpr size_t SIZE = 1000;
        std::vector<float> host_data(SIZE);
        std::iota(host_data.begin(), host_data.end(), 0.0f);
        
        // Use ARBD DeviceMemory wrapper (already optimized for USM)
        DeviceMemory<float> device_mem(queue.get(), SIZE);
        device_mem.copyFromHost(host_data);
        
        std::vector<float> result(SIZE, -1.0f); // Initialize with different value
        device_mem.copyToHost(result);
        
        // Verify the copy operation
        bool copy_successful = true;
        for (size_t i = 0; i < SIZE; ++i) {
            if (!within_tolerance(result[i], host_data[i])) {
                copy_successful = false;
                break;
            }
        }
        
        assert(copy_successful && "Memory copy verification failed");
        std::cout << "PASS\n";
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_parallel_for() {
    std::cout << "[TEST] SYCL Parallel For... ";
    try {
        auto& queue = SYCLManager::get_current_queue();
        
        // Vector addition test using USM and in-order queue
        constexpr size_t SIZE = 1000;
        std::vector<float> a(SIZE, 1.0f);
        std::vector<float> b(SIZE, 2.0f);
        std::vector<float> c(SIZE, 0.0f);
        
        // Use ARBD DeviceMemory for automatic USM management
        DeviceMemory<float> d_a(queue.get(), SIZE);
        DeviceMemory<float> d_b(queue.get(), SIZE);
        DeviceMemory<float> d_c(queue.get(), SIZE);
        
        d_a.copyFromHost(a);
        d_b.copyFromHost(b);
        
        // Get raw pointers for kernel
        float* ptr_a = d_a.get();
        float* ptr_b = d_b.get();
        float* ptr_c = d_c.get();
        
        // Submit parallel kernel using USM pointers
        auto event = queue.submit([=](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                ptr_c[i] = ptr_a[i] + ptr_b[i];
            });
        });
        
        event.wait(); // Wait for kernel completion
        d_c.copyToHost(c);
        
        // Verify results
        bool computation_correct = true;
        for (size_t i = 0; i < SIZE; ++i) {
            if (!within_tolerance(c[i], 3.0f)) {
                computation_correct = false;
                break;
            }
        }
        
        assert(computation_correct && "Parallel computation verification failed");
        std::cout << "PASS\n";
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_reduction() {
    std::cout << "[TEST] SYCL Reduction... ";
    try {
        auto& queue = SYCLManager::get_current_queue();
        
        // Parallel reduction test
        constexpr size_t SIZE = 1000;
        std::vector<int> data(SIZE);
        std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ..., SIZE
        
        DeviceMemory<int> d_data(queue.get(), SIZE);
        DeviceMemory<int> d_result(queue.get(), 1);
        
        d_data.copyFromHost(data);
        
        int* ptr_data = d_data.get();
        int* ptr_result = d_result.get();
        
        // Initialize result to 0
        int zero = 0;
        queue.get().memcpy(ptr_result, &zero, sizeof(int)).wait();
        
        // Parallel sum reduction using SYCL2020 reduction
        auto event = queue.submit([=](sycl::handler& h) {
            auto sum_reduction = sycl::reduction(ptr_result, sycl::plus<int>());
            h.parallel_for(sycl::range<1>(SIZE), sum_reduction,
                          [=](sycl::id<1> idx, auto& sum) {
                              sum += ptr_data[idx[0]];
                          });
        });
        
        event.wait();
        
        std::vector<int> result(1);
        d_result.copyToHost(result);
        
        // Verify result: sum of 1 to SIZE = SIZE * (SIZE + 1) / 2
        int expected = SIZE * (SIZE + 1) / 2;
        assert(result[0] == expected && "Reduction computation verification failed");
        
        std::cout << "PASS\n";
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

 

int main() {
    try {
        // Run tests in sequence with proper error handling
        test_sycl_cpu_initialization();
        test_sycl_basic_memory_operations();
        test_sycl_parallel_for();
        test_sycl_reduction();
        
        std::cout << "\nAll SYCL OpenMP tests passed!\n";
        
        // Explicit cleanup to prevent mutex issues
        try {
            SYCLManager::finalize();
        } catch (...) {
            // Ignore cleanup errors
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with error: " << e.what() << "\n";
        
        // Attempt cleanup even on failure
        try {
            SYCLManager::finalize();
        } catch (...) {
            // Ignore cleanup errors
        }
        
        return 1;
    }
}

#else // PROJECT_USES_SYCL

int main() {
    std::cout << "SYCL support not enabled, skipping OpenMP SYCL tests\n";
    return 0;
}

#endif // PROJECT_USES_SYCL 