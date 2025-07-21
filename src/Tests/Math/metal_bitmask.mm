#include "../catch_boiler.h"
#ifdef USE_METAL

#include "Math/Bitmask.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"
#include <vector>
#include <memory>

using namespace ARBD;

// Test fixture for Metal Bitmask tests
struct MetalBitmaskTestFixture {
    Resource metal_resource;

    MetalBitmaskTestFixture() {
        try {
            ARBD::METAL::METALManager::init();
            ARBD::METAL::METALManager::load_info();
            
            if (ARBD::METAL::METALManager::devices().empty()) {
                WARN("No Metal devices found. Skipping Metal Bitmask tests.");
                return;
            }
            
            metal_resource = Resource(ResourceType::METAL, 0);
            ARBD::METAL::METALManager::use(0);

        } catch (const std::exception& e) {
            FAIL("Failed to initialize METALManager in Bitmask test fixture: " << e.what());
        }
    }

    ~MetalBitmaskTestFixture() {
        try {
            ARBD::METAL::METALManager::finalize();
        } catch (const std::exception& e) {
            std::cerr << "Error during METALManager finalization in Bitmask test fixture: " << e.what() << std::endl;
        }
    }
};

TEST_CASE_METHOD(MetalBitmaskTestFixture, "Metal Bitmask Basic Operations", "[metal][bitmask]") {
    if (ARBD::METAL::METALManager::devices().empty()) {
        SKIP("No Metal devices available");
    }
    
    SECTION("Basic bitmask creation and operations") {
        Bitmask host_bitmask(64);
        
        // Set some bits on host
        host_bitmask.set_mask(0, true);
        host_bitmask.set_mask(15, true);
        host_bitmask.set_mask(31, true);
        host_bitmask.set_mask(63, true);
        
        // Verify bits are set
        REQUIRE(host_bitmask.get_mask(0) == true);
        REQUIRE(host_bitmask.get_mask(15) == true);
        REQUIRE(host_bitmask.get_mask(31) == true);
        REQUIRE(host_bitmask.get_mask(63) == true);
        
        // Verify unset bits
        REQUIRE(host_bitmask.get_mask(1) == false);
        REQUIRE(host_bitmask.get_mask(16) == false);
        REQUIRE(host_bitmask.get_mask(32) == false);
        
        // Test string representation
        std::string bit_string = host_bitmask.to_string();
        REQUIRE(bit_string.length() == 64);
        REQUIRE(bit_string[0] == '1');
        REQUIRE(bit_string[15] == '1');
        REQUIRE(bit_string[31] == '1');
        REQUIRE(bit_string[63] == '1');
    }
    
    SECTION("Bitmask equality operations") {
        Bitmask b1(32);
        Bitmask b2(32);
        
        // Initially equal
        REQUIRE(b1 == b2);
        
        // Set same bits
        b1.set_mask(5, true);
        b2.set_mask(5, true);
        REQUIRE(b1 == b2);
        
        // Set different bits
        b1.set_mask(10, true);
        REQUIRE_FALSE(b1 == b2);
        
        // Make equal again
        b2.set_mask(10, true);
        REQUIRE(b1 == b2);
    }
}

TEST_CASE_METHOD(MetalBitmaskTestFixture, "Metal Bitmask Backend Operations", "[metal][bitmask][backend]") {
    if (ARBD::METAL::METALManager::devices().empty()) {
        SKIP("No Metal devices available");
    }
    
    SECTION("Send bitmask to Metal backend") {
        Bitmask host_bitmask(128);
        
        // Set a pattern on host
        for (size_t i = 0; i < 128; i += 8) {
            host_bitmask.set_mask(i, true);
        }
        
        // Send to Metal device
        Bitmask* device_bitmask = host_bitmask.send_to_backend(metal_resource);
        REQUIRE(device_bitmask != nullptr);
        
        // Receive back from device
        Bitmask received_bitmask = Bitmask::receive_from_backend(device_bitmask, metal_resource);
        
        // Verify the pattern is preserved
        REQUIRE(received_bitmask.get_len() == 128);
        for (size_t i = 0; i < 128; i += 8) {
            REQUIRE(received_bitmask.get_mask(i) == true);
        }
        for (size_t i = 1; i < 128; i += 8) {
            REQUIRE(received_bitmask.get_mask(i) == false);
        }
        
        // Verify equality
        REQUIRE(host_bitmask == received_bitmask);
        
        // Clean up device memory
        Bitmask::remove_from_backend(device_bitmask, metal_resource);
    }
    
    SECTION("Multiple bitmask operations") {
        std::vector<std::unique_ptr<Bitmask>> host_bitmasks;
        std::vector<Bitmask*> device_bitmasks;
        
        // Create multiple bitmasks with different patterns
        for (int i = 0; i < 5; ++i) {
            auto bitmask = std::make_unique<Bitmask>(64);
            
            // Set different patterns for each bitmask
            for (size_t j = i; j < 64; j += 5) {
                bitmask->set_mask(j, true);
            }
            
            // Send to device
            device_bitmasks.push_back(bitmask->send_to_backend(metal_resource));
            host_bitmasks.push_back(std::move(bitmask));
        }
        
        // Verify all transfers
        for (int i = 0; i < 5; ++i) {
            Bitmask received = Bitmask::receive_from_backend(device_bitmasks[i], metal_resource);
            REQUIRE(*host_bitmasks[i] == received);
        }
        
        // Clean up
        for (auto* device_bitmask : device_bitmasks) {
            Bitmask::remove_from_backend(device_bitmask, metal_resource);
        }
    }
}

TEST_CASE_METHOD(MetalBitmaskTestFixture, "Metal SparseBitmask Operations", "[metal][bitmask][sparse]") {
    if (ARBD::METAL::METALManager::devices().empty()) {
        SKIP("No Metal devices available");
    }
    
    SECTION("SparseBitmask basic operations") {
        SparseBitmask<64> sparse_bitmask(10000);
        
        // Set sparse bits
        sparse_bitmask.set_mask(0, true);
        sparse_bitmask.set_mask(1000, true);
        sparse_bitmask.set_mask(5000, true);
        sparse_bitmask.set_mask(9999, true);
        
        // Verify bits are set
        REQUIRE(sparse_bitmask.get_mask(0) == true);
        REQUIRE(sparse_bitmask.get_mask(1000) == true);
        REQUIRE(sparse_bitmask.get_mask(5000) == true);
        REQUIRE(sparse_bitmask.get_mask(9999) == true);
        
        // Verify unset bits
        REQUIRE(sparse_bitmask.get_mask(1) == false);
        REQUIRE(sparse_bitmask.get_mask(500) == false);
        REQUIRE(sparse_bitmask.get_mask(2000) == false);
        
        // Check that we're using sparse storage efficiently
        REQUIRE(sparse_bitmask.get_allocated_chunks() <= 4);
        
        INFO("SparseBitmask allocated " << sparse_bitmask.get_allocated_chunks() << " chunks for 10000 bits");
    }
    
    SECTION("SparseBitmask chunk management") {
        SparseBitmask<32> sparse_bitmask(1000);
        
        // Set bits in different chunks
        sparse_bitmask.set_mask(0, true);    // Chunk 0
        sparse_bitmask.set_mask(100, true);  // Chunk 3
        sparse_bitmask.set_mask(500, true);  // Chunk 15
        
        REQUIRE(sparse_bitmask.get_allocated_chunks() == 3);
        
        // Clear a bit (chunk should remain allocated)
        sparse_bitmask.set_mask(100, false);
        REQUIRE(sparse_bitmask.get_mask(100) == false);
        REQUIRE(sparse_bitmask.get_allocated_chunks() == 3); // Still allocated
        
        // Other bits should remain set
        REQUIRE(sparse_bitmask.get_mask(0) == true);
        REQUIRE(sparse_bitmask.get_mask(500) == true);
    }
}

TEST_CASE_METHOD(MetalBitmaskTestFixture, "Metal Bitmask Stress Tests", "[metal][bitmask][stress]") {
    if (ARBD::METAL::METALManager::devices().empty()) {
        SKIP("No Metal devices available");
    }
    
    SECTION("Large bitmask operations") {
        const size_t large_size = 10000;
        Bitmask large_bitmask(large_size);
        
        // Set every 100th bit
        for (size_t i = 0; i < large_size; i += 100) {
            large_bitmask.set_mask(i, true);
        }
        
        // Verify pattern
        for (size_t i = 0; i < large_size; ++i) {
            bool expected = (i % 100 == 0);
            REQUIRE(large_bitmask.get_mask(i) == expected);
        }
        
        // Test backend transfer
        Bitmask* device_bitmask = large_bitmask.send_to_backend(metal_resource);
        Bitmask received_bitmask = Bitmask::receive_from_backend(device_bitmask, metal_resource);
        
        REQUIRE(large_bitmask == received_bitmask);
        
        Bitmask::remove_from_backend(device_bitmask, metal_resource);
    }
}

TEST_CASE_METHOD(MetalBitmaskTestFixture, "Metal Bitmask Performance Tests", "[metal][bitmask][performance]") {
    if (ARBD::METAL::METALManager::devices().empty()) {
        SKIP("No Metal devices available");
    }
    
    SECTION("Batch operations") {
        const size_t batch_size = 100;
        const size_t bitmask_size = 1024;
        
        std::vector<std::unique_ptr<Bitmask>> host_bitmasks;
        std::vector<Bitmask*> device_bitmasks;
        
        // Create batch of bitmasks
        for (size_t i = 0; i < batch_size; ++i) {
            auto bitmask = std::make_unique<Bitmask>(bitmask_size);
            
            // Set a unique pattern for each bitmask
            for (size_t j = i; j < bitmask_size; j += batch_size) {
                bitmask->set_mask(j, true);
            }
            
            host_bitmasks.push_back(std::move(bitmask));
        }
        
        // Send all to device
        for (size_t i = 0; i < batch_size; ++i) {
            device_bitmasks.push_back(host_bitmasks[i]->send_to_backend(metal_resource));
        }
        
        // Receive all back and verify
        for (size_t i = 0; i < batch_size; ++i) {
            Bitmask received = Bitmask::receive_from_backend(device_bitmasks[i], metal_resource);
            REQUIRE(*host_bitmasks[i] == received);
        }
        
        // Clean up
        for (auto* device_bitmask : device_bitmasks) {
            Bitmask::remove_from_backend(device_bitmask, metal_resource);
        }
        
        INFO("Successfully processed " << batch_size << " bitmasks of size " << bitmask_size);
    }
}

// Additional Metal-specific tests
TEST_CASE_METHOD(MetalBitmaskTestFixture, "Metal Bitmask Memory Management", "[metal][bitmask][memory]") {
    if (ARBD::METAL::METALManager::devices().empty()) {
        SKIP("No Metal devices available");
    }
    
    SECTION("Memory allocation and deallocation") {
        std::vector<Bitmask*> device_bitmasks;
        
        // Allocate multiple bitmasks
        for (int i = 0; i < 10; ++i) {
            Bitmask host_bitmask(256);
            host_bitmask.set_mask(i * 10, true);
            device_bitmasks.push_back(host_bitmask.send_to_backend(metal_resource));
        }
        
        // Verify they're all valid
        for (size_t i = 0; i < device_bitmasks.size(); ++i) {
            REQUIRE(device_bitmasks[i] != nullptr);
            
            Bitmask received = Bitmask::receive_from_backend(device_bitmasks[i], metal_resource);
            REQUIRE(received.get_len() == 256);
            REQUIRE(received.get_mask(i * 10) == true);
        }
        
        // Clean up in reverse order
        for (auto it = device_bitmasks.rbegin(); it != device_bitmasks.rend(); ++it) {
            Bitmask::remove_from_backend(*it, metal_resource);
        }
    }
}

#endif // USE_METAL 