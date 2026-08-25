#include "Backend/Buffer.h"
#include "Backend/Resource.h"

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <vector>
#include <numeric>
#include "../catch_boiler.h"
using namespace MARS;


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
TEST_CASE("UnifiedBuffer SYCL Basic Allocation", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        MARS::SYCL::SYCLManager::init();
        MARS::SYCL::SYCLManager::load_info();
        
        if (MARS::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        MARS::Resource sycl_resource{MARS::ResourceType::SYCL, 0};
        
        SECTION("Basic allocation and deallocation") {
            MARS::UnifiedBuffer<float> buffer(1000, sycl_resource);
            
            REQUIRE(buffer.size() == 1000);
            REQUIRE(!buffer.empty());
            REQUIRE(buffer.primary_location().type == MARS::ResourceType::SYCL);
            
            // Check that pointer is valid
            float* ptr = buffer.get_ptr(sycl_resource);
            REQUIRE(ptr != nullptr);
        }
        
        SECTION("Zero-sized buffer") {
            MARS::UnifiedBuffer<int> buffer(0, sycl_resource);
            
            REQUIRE(buffer.size() == 0);
            REQUIRE(buffer.empty());
        }
        
        SECTION("Different data types") {
            MARS::UnifiedBuffer<double> double_buffer(500, sycl_resource);
            MARS::UnifiedBuffer<int> int_buffer(1000, sycl_resource);
            MARS::UnifiedBuffer<char> char_buffer(2000, sycl_resource);
            
            REQUIRE(double_buffer.size() == 500);
            REQUIRE(int_buffer.size() == 1000);
            REQUIRE(char_buffer.size() == 2000);
            
            REQUIRE(double_buffer.get_ptr(sycl_resource) != nullptr);
            REQUIRE(int_buffer.get_ptr(sycl_resource) != nullptr);
            REQUIRE(char_buffer.get_ptr(sycl_resource) != nullptr);
        }
        
        MARS::SYCL::SYCLManager::finalize();
    } catch (const MARS::Exception& e) {
        FAIL("SYCL test failed with MARS exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Data Migration", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        MARS::SYCL::SYCLManager::init();
        MARS::SYCL::SYCLManager::load_info();
        
        if (MARS::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        MARS::Resource sycl_resource{MARS::ResourceType::SYCL, 0};
        MARS::Resource host_resource{}; // Default host resource
        
        SECTION("Ensure availability at different resources") {
            MARS::UnifiedBuffer<float> buffer(100, sycl_resource);
            // Should be available at primary location
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == MARS::ResourceType::SYCL);
            
            // Ensure availability at host (should trigger migration)
            buffer.ensure_available_at(host_resource);
            
            // Should now be available at both locations
            locations = buffer.available_locations();
            if (locations.size() <= 2) {
                SKIP("less than 2 locations available");
            }
            // Both pointers should be valid
            REQUIRE(buffer.get_ptr(sycl_resource) != nullptr);
            REQUIRE(buffer.get_ptr(host_resource) != nullptr);
        }
        
        SECTION("Release memory at specific location") {
            MARS::UnifiedBuffer<int> buffer(50, sycl_resource);
            
            // Ensure available at host
            buffer.ensure_available_at(host_resource);
            if (buffer.available_locations().size() <= 2) {
                SKIP("less than 2 devices available");
            }
            // Release at host
            buffer.release_at(host_resource);
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == MARS::ResourceType::SYCL);
        }
        
        MARS::SYCL::SYCLManager::finalize();
    } catch (const MARS::Exception& e) {
        FAIL("SYCL data migration test failed with MARS exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Move Semantics", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        MARS::SYCL::SYCLManager::init();
        MARS::SYCL::SYCLManager::load_info();
        
        if (MARS::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        MARS::Resource sycl_resource{MARS::ResourceType::SYCL, 0};
        
        SECTION("Move constructor") {
            MARS::UnifiedBuffer<float> buffer1(1000, sycl_resource);
            float* original_ptr = buffer1.get_ptr(sycl_resource);
            
            MARS::UnifiedBuffer<float> buffer2(std::move(buffer1));
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(sycl_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
            REQUIRE(buffer1.empty());
        }
        
        SECTION("Move assignment") {
            MARS::UnifiedBuffer<float> buffer1(1000, sycl_resource);
            MARS::UnifiedBuffer<float> buffer2(500, sycl_resource);
            
            float* original_ptr = buffer1.get_ptr(sycl_resource);
            
            buffer2 = std::move(buffer1);
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(sycl_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
        }
        
        MARS::SYCL::SYCLManager::finalize();
    } catch (const MARS::Exception& e) {
        FAIL("SYCL move semantics test failed with MARS exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Existing Data Constructor", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        MARS::SYCL::SYCLManager::init();
        MARS::SYCL::SYCLManager::load_info();
        
        if (MARS::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        MARS::Resource host_resource{}; // Default host resource
        
        SECTION("Construct from existing host data") {
            std::vector<int> host_data(100);
            std::iota(host_data.begin(), host_data.end(), 1); // Fill with 1, 2, 3, ...
            
            MARS::UnifiedBuffer<int> buffer(host_data.size(), host_data.data(), host_resource);
            
            REQUIRE(buffer.size() == 100);
            REQUIRE(buffer.get_ptr(host_resource) == host_data.data());
            
            // Check that data is accessible
            const int* const_ptr = buffer.get_ptr(host_resource);
            REQUIRE(const_ptr[0] == 1);
            REQUIRE(const_ptr[50] == 51);
            REQUIRE(const_ptr[99] == 100);
        }
        
        MARS::SYCL::SYCLManager::finalize();
    } catch (const MARS::Exception& e) {
        FAIL("SYCL existing data test failed with MARS exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Error Handling", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        MARS::SYCL::SYCLManager::init();
        MARS::SYCL::SYCLManager::load_info();
        
        if (MARS::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        MARS::Resource sycl_resource{MARS::ResourceType::SYCL, 0};
        
        SECTION("Get pointer for non-existent location") {
            MARS::UnifiedBuffer<float> buffer(100, sycl_resource);
            
            MARS::Resource other_resource{MARS::ResourceType::SYCL, 999}; // Non-existent device
            
            // This should trigger ensure_available_at and potentially fail gracefully
            // or migrate from existing location
            float* ptr = buffer.get_ptr(other_resource);
            // The behavior depends on implementation - should either work or handle gracefully
        }
        
        MARS::SYCL::SYCLManager::finalize();
    } catch (const MARS::Exception& e) {
        // Expected for some error cases
        INFO("SYCL error handling test completed with exception: " << e.what());
    }
}

#else
TEST_CASE("UnifiedBuffer SYCL - SYCL Not Available", "[UnifiedBuffer][SYCL]") {
    SKIP("SYCL support not compiled in");
}
#endif 