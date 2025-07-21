#include "../catch_boiler.h"
#ifdef USE_SYCL

#include "Backend/SYCL/SYCLManager.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include <vector>
#include <numeric>
#include <string>

using namespace ARBD;

// ============================================================================
// Test Fixture for SYCL Backend
// ============================================================================
// This fixture handles the initialization and finalization of the SYCLManager
// for each test case, ensuring a clean state.
struct SYCLTestFixture {
    Resource sycl_resource;

    SYCLTestFixture() {
        try {
            SYCL::SYCLManager::init();
            SYCL::SYCLManager::load_info();
            
            // Skip tests if no SYCL devices are found.
            if (SYCL::SYCLManager::all_devices().empty()) {
                WARN("No SYCL devices found. Skipping SYCL backend tests.");
                return;
            }
            
            // Use the first available SYCL device for all tests.
            sycl_resource = Resource(ResourceType::SYCL, 0);
            SYCL::SYCLManager::use(0);

        } catch (const std::exception& e) {
            FAIL("Failed to initialize SYCLManager in test fixture: " << e.what());
        }
    }

    ~SYCLTestFixture() {
        try {
            SYCL::SYCLManager::finalize();
        } catch (const std::exception& e) {
            // It's not ideal to throw from a destructor. Log the error instead.
            std::cerr << "Error during SYCLManager finalization in test fixture: " << e.what() << std::endl;
        }
    }
};

// A simple class for proxy testing
class TestMath {
public:
    int add(int a, int b) {
        return a + b;
    }
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(SYCLTestFixture, "SYCL Resource Creation and Properties", "[sycl][resource]") {
    if (SYCL::SYCLManager::all_devices().empty()) return;

    SECTION("Resource type is correct") {
        CHECK(sycl_resource.type == ResourceType::SYCL);
        CHECK(sycl_resource.id == 0);
    }

    SECTION("Resource type string") {
        CHECK(std::string(sycl_resource.getTypeString()) == "SYCL");
    }

    SECTION("Resource is device") {
        CHECK(sycl_resource.is_device());
        CHECK_FALSE(sycl_resource.is_host());
    }
}

TEST_CASE_METHOD(SYCLTestFixture, "SYCL DeviceBuffer Basic Operations", "[sycl][buffer]") {
    if (SYCL::SYCLManager::all_devices().empty()) return;

    SECTION("Buffer allocation") {
        const size_t count = 1000;
        DeviceBuffer<float> buffer(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.get_resource() == sycl_resource);
        CHECK(buffer.data() != nullptr);
    }
    
    SECTION("Buffer move semantics") {
        const size_t count = 500;
        DeviceBuffer<double> buffer1(count, sycl_resource);
        auto* original_ptr = buffer1.data();
        
        DeviceBuffer<double> buffer2 = std::move(buffer1);
        
        CHECK(buffer2.size() == count);
        CHECK(buffer2.data() == original_ptr);
        CHECK(buffer1.size() == 0);
        CHECK(buffer1.data() == nullptr);
    }
}

TEST_CASE_METHOD(SYCLTestFixture, "SYCL DeviceBuffer Data Transfer", "[sycl][buffer][transfer]") {
    if (SYCL::SYCLManager::all_devices().empty()) return;
    const size_t count = 100;
    
    SECTION("Host to device and device to host copy") {
        DeviceBuffer<int> buffer(count, sycl_resource);
        std::vector<int> host_input(count);
        std::iota(host_input.begin(), host_input.end(), 10); // Fill with 10, 11, 12, ...

        // Copy to device
        REQUIRE_NOTHROW(buffer.copy_from_host(host_input.data(), count));

        // Copy back to host
        std::vector<int> host_output(count, 0);
        REQUIRE_NOTHROW(buffer.copy_to_host(host_output.data(), count));
        
        // Verify data integrity
        CHECK(host_output == host_input);
    }
}

TEST_CASE_METHOD(SYCLTestFixture, "SYCL UnifiedBuffer Operations", "[sycl][unified_buffer]") {
    if (SYCL::SYCLManager::all_devices().empty()) return;
    const size_t count = 200;

    SECTION("Unified buffer creation and data copy") {
        UnifiedBuffer<int> buffer(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        CHECK(buffer.primary_location() == sycl_resource);

        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 42);
        
        REQUIRE_NOTHROW(buffer.copy_from_host(host_data.data(), sycl_resource, count));
        
        std::vector<int> readback(count, 0);
        REQUIRE_NOTHROW(buffer.copy_to_host(readback.data(), sycl_resource, count));
        
        CHECK(readback == host_data);
    }
}


TEST_CASE_METHOD(SYCLTestFixture, "SYCL MemoryOps", "[sycl][memory]") {
    if (SYCL::SYCLManager::all_devices().empty()) return;
    const size_t count = 128;
    const size_t bytes = count * sizeof(float);

    SECTION("Allocate, copy, and deallocate") {
        // Allocate
        void* device_ptr = MemoryOps::allocate(bytes, sycl_resource);
        REQUIRE(device_ptr != nullptr);

        // Copy from host
        std::vector<float> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1.5f);
        REQUIRE_NOTHROW(MemoryOps::copy_from_host(device_ptr, host_data.data(), bytes, sycl_resource));

        // Copy to host
        std::vector<float> host_result(count, 0.0f);
        REQUIRE_NOTHROW(MemoryOps::copy_to_host(host_result.data(), device_ptr, bytes, sycl_resource));

        // Verify
        CHECK(host_result == host_data);

        // Deallocate
        REQUIRE_NOTHROW(MemoryOps::deallocate(device_ptr, sycl_resource));
    }
}

TEST_CASE_METHOD(SYCLTestFixture, "SYCL GenericAllocator", "[sycl][allocator]") {
    if (SYCL::SYCLManager::all_devices().empty()) return;
    const size_t count = 64;

    SECTION("allocate_zeroed") {
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, sycl_resource);
        
        std::vector<int> result(count, -1); // Fill with a non-zero value
        buffer.copy_to_host(result.data(), count);
        
        std::vector<int> expected(count, 0);
        CHECK(result == expected);
    }

    SECTION("allocate_sequence") {
        auto buffer = GenericAllocator<int>::allocate_sequence(count, sycl_resource, 10, 2);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        std::vector<int> expected(count);
        for(size_t i = 0; i < count; ++i) {
            expected[i] = 10 + static_cast<int>(i) * 2;
        }
        CHECK(result == expected);
    }
}

#else

TEST_CASE("SYCL Support Not Enabled", "[sycl]") {
    SUCCEED("SYCL support not enabled, tests are skipped.");
}

#endif // USE_SYCL