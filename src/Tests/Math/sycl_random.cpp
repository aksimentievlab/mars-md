#include "../catch_boiler.h"
#ifdef USE_SYCL

#include "Math/Bitmask.h"
#include "Backend/SYCL/SYCLManager.h"
#include "Backend/Resource.h"
#include "Random/Random.h"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>

using namespace ARBD;

TEST_CASE("Random number reproducibility", "[sycl][random][reproducibility]"){
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    SYCL::SYCLManager::sync();
    LOGINFO("SYCL Random Number Reproducibility Test");
    auto resource = Resource(ResourceType::SYCL, 0);
    
    const size_t buffer_size = 1000;
    const unsigned long test_seed = 12345;
    
    SECTION("Same seed produces identical results") {
        // First run
        RandomSYCL<128> random_gen1(resource);
        random_gen1.init(test_seed);
        
        DeviceBuffer<float> buffer1(buffer_size, resource);
        auto event1 = random_gen1.generate_uniform(buffer1, 0.0f, 1.0f);
        std::vector<float> data1(buffer_size);
        event1.wait();
        buffer1.copy_to_host(data1.data());
        
        // Second run with same seed
        RandomSYCL<128> random_gen2(resource);
        random_gen2.init(test_seed);  // Same seed
        
        DeviceBuffer<float> buffer2(buffer_size, resource);
        auto event2 = random_gen2.generate_uniform(buffer2, 0.0f, 1.0f);
        std::vector<float> data2(buffer_size);
        event2.wait();
        buffer2.copy_to_host(data2.data());
        
        // Compare results
        std::cout << "\n=== Reproducibility Test: Same Seed ===" << std::endl;
        std::cout << "Seed: " << test_seed << std::endl;
        std::cout << "Comparing first 10 values:" << std::endl;
        std::cout << "Index |    Run 1     |    Run 2     | Match" << std::endl;
        std::cout << "------|--------------|--------------|------" << std::endl;
        
        bool all_match = true;
        for (size_t i = 0; i < 10; ++i) {
            bool match = (data1[i] == data2[i]);
            all_match = all_match && match;
            std::cout << std::setw(5) << i << " | " 
                      << std::setprecision(8) << std::fixed << std::setw(12) << data1[i] << " | "
                      << std::setprecision(8) << std::fixed << std::setw(12) << data2[i] << " | "
                      << (match ? "✓" : "✗") << std::endl;
        }
        
        // Check all values
        for (size_t i = 0; i < buffer_size; ++i) {
            REQUIRE(data1[i] == data2[i]);
        }
        
        std::cout << "\nResult: " << (all_match ? "✓ REPRODUCIBLE" : "✗ NOT REPRODUCIBLE") << std::endl;
        REQUIRE(all_match);
        LOGINFO("Same seed reproducibility test passed");
    }
    
    SECTION("Different seeds produce different results") {
        // First run with seed A
        RandomSYCL<128> random_gen1(resource);
        random_gen1.init(test_seed);
        
        DeviceBuffer<float> buffer1(buffer_size, resource);
        auto event1 = random_gen1.generate_uniform(buffer1, 0.0f, 1.0f);
        std::vector<float> data1(buffer_size);
        event1.wait();
        buffer1.copy_to_host(data1.data());
        
        // Second run with different seed
        RandomSYCL<128> random_gen2(resource);
        random_gen2.init(test_seed + 1);  // Different seed
        
        DeviceBuffer<float> buffer2(buffer_size, resource);
        auto event2 = random_gen2.generate_uniform(buffer2, 0.0f, 1.0f);
        std::vector<float> data2(buffer_size);
        event2.wait();
        buffer2.copy_to_host(data2.data());
        
        // Compare results - should be different
        std::cout << "\n=== Reproducibility Test: Different Seeds ===" << std::endl;
        std::cout << "Seed 1: " << test_seed << ", Seed 2: " << (test_seed + 1) << std::endl;
        std::cout << "Comparing first 10 values:" << std::endl;
        std::cout << "Index |   Seed " << test_seed << "   |   Seed " << (test_seed + 1) << "   | Different" << std::endl;
        std::cout << "------|--------------|--------------|----------" << std::endl;
        
        int different_count = 0;
        for (size_t i = 0; i < 10; ++i) {
            bool different = (data1[i] != data2[i]);
            if (different) different_count++;
            std::cout << std::setw(5) << i << " | " 
                      << std::setprecision(8) << std::fixed << std::setw(12) << data1[i] << " | "
                      << std::setprecision(8) << std::fixed << std::setw(12) << data2[i] << " | "
                      << (different ? "✓" : "✗") << std::endl;
        }
        
        // Count total differences
        int total_different = 0;
        for (size_t i = 0; i < buffer_size; ++i) {
            if (data1[i] != data2[i]) total_different++;
        }
        
        std::cout << "\nResult: " << total_different << "/" << buffer_size 
                  << " values different (" << (100.0f * total_different / buffer_size) << "%)" << std::endl;
        
        // Should be mostly different (expect >95% different)
        REQUIRE(total_different > (buffer_size * 0.95));
        LOGINFO("Different seed test passed");
    }
    
    SECTION("Thread-specific reproducibility") {
        // Test that specific thread IDs produce consistent results
        const size_t test_size = 100;
        
        RandomSYCL<128> random_gen1(resource);
        random_gen1.init(test_seed);
        
        DeviceBuffer<float> buffer1(test_size, resource);
        auto event1 = random_gen1.generate_uniform(buffer1, 0.0f, 1.0f);
        std::vector<float> data1(test_size);
        event1.wait();
        buffer1.copy_to_host(data1.data());
        
        // Second run - should get exact same sequence
        RandomSYCL<128> random_gen2(resource);
        random_gen2.init(test_seed);
        
        DeviceBuffer<float> buffer2(test_size, resource);
        auto event2 = random_gen2.generate_uniform(buffer2, 0.0f, 1.0f);
        std::vector<float> data2(test_size);
        event2.wait();
        buffer2.copy_to_host(data2.data());
        
        std::cout << "\n=== Thread-Specific Reproducibility ===" << std::endl;
        std::cout << "Testing that each thread gets same sequence with same seed..." << std::endl;
        
        // Since Philox is counter-based, thread 0 should always get the same value,
        // thread 1 should always get the same value, etc.
        bool perfectly_reproducible = true;
        for (size_t i = 0; i < test_size; ++i) {
            if (data1[i] != data2[i]) {
                perfectly_reproducible = false;
                std::cout << "Mismatch at index " << i << ": " 
                          << data1[i] << " vs " << data2[i] << std::endl;
            }
        }
        
        std::cout << "Result: " << (perfectly_reproducible ? "✓ PERFECTLY REPRODUCIBLE" : "✗ NOT REPRODUCIBLE") << std::endl;
        REQUIRE(perfectly_reproducible);
        LOGINFO("Thread-specific reproducibility test passed");
    }
    
    SECTION("Gaussian reproducibility") {
        // Test that Gaussian generation is also reproducible
        RandomSYCL<128> random_gen1(resource);
        random_gen1.init(test_seed);
        
        DeviceBuffer<float> buffer1(buffer_size, resource);
        auto event1 = random_gen1.generate_gaussian(buffer1, 0.0f, 1.0f);
        std::vector<float> data1(buffer_size);
        event1.wait();
        buffer1.copy_to_host(data1.data());
        
        // Second run with same seed
        RandomSYCL<128> random_gen2(resource);
        random_gen2.init(test_seed);
        
        DeviceBuffer<float> buffer2(buffer_size, resource);
        auto event2 = random_gen2.generate_gaussian(buffer2, 0.0f, 1.0f);
        std::vector<float> data2(buffer_size);
        event2.wait();
        buffer2.copy_to_host(data2.data());
        
        std::cout << "\n=== Gaussian Reproducibility Test ===" << std::endl;
        std::cout << "Comparing first 10 Gaussian values:" << std::endl;
        
        bool all_match = true;
        for (size_t i = 0; i < 10; ++i) {
            bool match = (data1[i] == data2[i]);
            all_match = all_match && match;
            if (i < 10) {  // Only print first 10
                std::cout << std::setw(3) << i << ": " 
                          << std::setprecision(6) << std::fixed << std::setw(10) << data1[i] 
                          << " vs " << std::setw(10) << data2[i] 
                          << " " << (match ? "✓" : "✗") << std::endl;
            }
        }
        
        // Check all values
        for (size_t i = 0; i < buffer_size; ++i) {
            REQUIRE(data1[i] == data2[i]);
        }
        
        std::cout << "Gaussian Result: " << (all_match ? "✓ REPRODUCIBLE" : "✗ NOT REPRODUCIBLE") << std::endl;
        REQUIRE(all_match);
        LOGINFO("Gaussian reproducibility test passed");
    }
}

#endif // USE_SYCL