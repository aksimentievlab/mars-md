#ifdef USE_METAL
#include "catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Backend/METAL/METALManager.h"
#include <vector>
#include <numeric>

TEST_CASE("UnifiedBuffer Metal Basic Allocation", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        
        SECTION("Basic allocation and deallocation") {
            ARBD::UnifiedBuffer<float> buffer(1000, metal_resource);
            
            REQUIRE(buffer.size() == 1000);
            REQUIRE(!buffer.empty());
            REQUIRE(buffer.primary_location().type == ARBD::ResourceType::METAL);
            
            // Check that pointer is valid
            float* ptr = buffer.get_ptr(metal_resource);
            REQUIRE(ptr != nullptr);
        }
        
        SECTION("Zero-sized buffer") {
            ARBD::UnifiedBuffer<int> buffer(0, metal_resource);
            
            REQUIRE(buffer.size() == 0);
            REQUIRE(buffer.empty());
        }
        
        SECTION("Different data types") {
            ARBD::UnifiedBuffer<double> double_buffer(500, metal_resource);
            ARBD::UnifiedBuffer<int> int_buffer(1000, metal_resource);
            ARBD::UnifiedBuffer<char> char_buffer(2000, metal_resource);
            
            REQUIRE(double_buffer.size() == 500);
            REQUIRE(int_buffer.size() == 1000);
            REQUIRE(char_buffer.size() == 2000);
            
            REQUIRE(double_buffer.get_ptr(metal_resource) != nullptr);
            REQUIRE(int_buffer.get_ptr(metal_resource) != nullptr);
            REQUIRE(char_buffer.get_ptr(metal_resource) != nullptr);
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Data Migration", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        ARBD::Resource host_resource{}; // Default host resource
        
        SECTION("Ensure availability at different resources") {
            ARBD::UnifiedBuffer<float> buffer(100, metal_resource);
            
            // Should be available at primary location
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == ARBD::ResourceType::METAL);
            
            // Metal buffers should only have 1 location
            REQUIRE(buffer.get_ptr(metal_resource) != nullptr);
        }
        
        SECTION("Metal buffer location management") {
            ARBD::UnifiedBuffer<int> buffer(50, metal_resource);
            
            // Metal buffer should only have 1 location
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == ARBD::ResourceType::METAL);
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal data migration test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Unified Memory", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        auto& device = ARBD::METAL::METALManager::get_current_device();
        
        SECTION("Test unified memory access") {
            ARBD::UnifiedBuffer<float> buffer(1000, metal_resource);
            float* ptr = buffer.get_ptr(metal_resource);
            REQUIRE(ptr != nullptr);
            
            // On Metal with unified memory, we should be able to write from host
            if (device.has_unified_memory()) {
                // Fill with test data
                for (size_t i = 0; i < 1000; ++i) {
                    ptr[i] = static_cast<float>(i * 0.5f);
                }
                
                // Verify data
                REQUIRE(ptr[0] == 0.0f);
                REQUIRE(ptr[500] == 250.0f);
                REQUIRE(ptr[999] == 499.5f);
            }
        }
        
        SECTION("Device properties verification") {
            INFO("Device name: " << device.name());
            INFO("Has unified memory: " << device.has_unified_memory());
            INFO("Is low power: " << device.is_low_power());
            INFO("Max threads per group: " << device.max_threads_per_group());
            
            // Basic sanity checks
            REQUIRE(!device.name().empty());
            REQUIRE(device.max_threads_per_group() > 0);
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal unified memory test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Move Semantics", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        
        SECTION("Move constructor") {
            ARBD::UnifiedBuffer<float> buffer1(1000, metal_resource);
            float* original_ptr = buffer1.get_ptr(metal_resource);
            
            ARBD::UnifiedBuffer<float> buffer2(std::move(buffer1));
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(metal_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
            REQUIRE(buffer1.empty());
        }
        
        SECTION("Move assignment") {
            ARBD::UnifiedBuffer<float> buffer1(1000, metal_resource);
            ARBD::UnifiedBuffer<float> buffer2(500, metal_resource);
            
            float* original_ptr = buffer1.get_ptr(metal_resource);
            
            buffer2 = std::move(buffer1);
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(metal_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal move semantics test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Raw Allocation Functions", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        SECTION("Test raw allocation/deallocation functions") {
            // Test the raw allocation functions we added
            size_t test_size = 1024; // 1KB
            void* ptr = ARBD::METAL::METALManager::allocate_raw(test_size);
            
            REQUIRE(ptr != nullptr);
            
            // Try to write to the memory (should work with unified memory)
            auto& device = ARBD::METAL::METALManager::get_current_device();
            if (device.has_unified_memory()) {
                char* char_ptr = static_cast<char*>(ptr);
                char_ptr[0] = 'A';
                char_ptr[test_size - 1] = 'Z';
                
                REQUIRE(char_ptr[0] == 'A');
                REQUIRE(char_ptr[test_size - 1] == 'Z');
            }
            
            // Clean up
            ARBD::METAL::METALManager::deallocate_raw(ptr);
        }
        
        SECTION("Test multiple allocations") {
            std::vector<void*> ptrs;
            
            // Allocate multiple buffers
            for (int i = 0; i < 10; ++i) {
                void* ptr = ARBD::METAL::METALManager::allocate_raw(100 * (i + 1));
                REQUIRE(ptr != nullptr);
                ptrs.push_back(ptr);
            }
            
            // All pointers should be different
            for (size_t i = 0; i < ptrs.size(); ++i) {
                for (size_t j = i + 1; j < ptrs.size(); ++j) {
                    REQUIRE(ptrs[i] != ptrs[j]);
                }
            }
            
            // Clean up all allocations
            for (void* ptr : ptrs) {
                ARBD::METAL::METALManager::deallocate_raw(ptr);
            }
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal raw allocation test failed with ARBD exception: " << e.what());
    }
}

#endif // USE_METAL 