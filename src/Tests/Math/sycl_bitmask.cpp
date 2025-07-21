#include "../catch_boiler.h"
#ifdef USE_SYCL

#include "Math/Bitmask.h"
#include "Backend/SYCL/SYCLManager.h"
#include "Backend/Resource.h"
#include <vector>
#include <memory>

using namespace ARBD;

// Test fixture for SYCL Bitmask tests
struct SYCLBitmaskTestFixture {
    Resource sycl_resource;

    SYCLBitmaskTestFixture() {
        try {
            SYCL::SYCLManager::init();
            SYCL::SYCLManager::load_info();
            
            if (SYCL::SYCLManager::all_devices().empty()) {
                WARN("No SYCL devices found. Skipping SYCL Bitmask tests.");
                return;
            }
            
            sycl_resource = Resource(ResourceType::SYCL, 0);
            SYCL::SYCLManager::use(0);

        } catch (const std::exception& e) {
            FAIL("Failed to initialize SYCLManager in Bitmask test fixture: " << e.what());
        }
    }

    ~SYCLBitmaskTestFixture() {
        try {
            SYCL::SYCLManager::finalize();
        } catch (const std::exception& e) {
            std::cerr << "Error during SYCLManager finalization in Bitmask test fixture: " << e.what() << std::endl;
        }
    }
};

TEST_CASE_METHOD(SYCLBitmaskTestFixture, "SYCL Bitmask Basic Operations", "[sycl][bitmask]") {
    if (SYCL::SYCLManager::all_devices().empty()) {
        SKIP("No SYCL devices available");
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

TEST_CASE_METHOD(SYCLBitmaskTestFixture, "SYCL Bitmask Backend Operations", "[sycl][bitmask][backend]") {
    if (SYCL::SYCLManager::all_devices().empty()) {
        SKIP("No SYCL devices available");
    }
    
    SECTION("Send bitmask to SYCL backend") {
        Bitmask host_bitmask(128);
        
        // Set a pattern on host
        for (size_t i = 0; i < 128; i += 8) {
            host_bitmask.set_mask(i, true);
        }
        
        // Send to SYCL device
        Bitmask* device_bitmask = host_bitmask.send_to_backend(sycl_resource);
        REQUIRE(device_bitmask != nullptr);
        
        // Receive back from device
        Bitmask received_bitmask = Bitmask::receive_from_backend(device_bitmask, sycl_resource);
        
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
        Bitmask::remove_from_backend(device_bitmask, sycl_resource);
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
            device_bitmasks.push_back(bitmask->send_to_backend(sycl_resource));
            host_bitmasks.push_back(std::move(bitmask));
        }
        
        // Verify all transfers
        for (int i = 0; i < 5; ++i) {
            Bitmask received = Bitmask::receive_from_backend(device_bitmasks[i], sycl_resource);
            REQUIRE(*host_bitmasks[i] == received);
        }
        
        // Clean up
        for (auto* device_bitmask : device_bitmasks) {
            Bitmask::remove_from_backend(device_bitmask, sycl_resource);
        }
    }
}

TEST_CASE_METHOD(SYCLBitmaskTestFixture, "SYCL SparseBitmask Operations", "[sycl][bitmask][sparse]") {
    if (SYCL::SYCLManager::all_devices().empty()) {
        SKIP("No SYCL devices available");
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

TEST_CASE_METHOD(SYCLBitmaskTestFixture, "SYCL Bitmask Stress Tests", "[sycl][bitmask][stress]") {
    if (SYCL::SYCLManager::all_devices().empty()) {
        SKIP("No SYCL devices available");
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
        Bitmask* device_bitmask = large_bitmask.send_to_backend(sycl_resource);
        Bitmask received_bitmask = Bitmask::receive_from_backend(device_bitmask, sycl_resource);
        
        REQUIRE(large_bitmask == received_bitmask);
        
        Bitmask::remove_from_backend(device_bitmask, sycl_resource);
    }
}

TEST_CASE("Bitmask Host-Only Tests", "[bitmask][host]") {
    SECTION("Run built-in test functions") {
        REQUIRE(BitmaskTests::run_all_tests());
    }
    
    SECTION("Edge cases") {
        // Zero-length bitmask
        Bitmask empty_bitmask(0);
        REQUIRE(empty_bitmask.get_len() == 0);
        
        // Single bit
        Bitmask single_bit(1);
        REQUIRE(single_bit.get_len() == 1);
        single_bit.set_mask(0, true);
        REQUIRE(single_bit.get_mask(0) == true);
        
        // Power of 2 sizes
        for (size_t size : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
            Bitmask power_of_2(size);
            REQUIRE(power_of_2.get_len() == size);
            
            // Test first and last bits
            power_of_2.set_mask(0, true);
            if (size > 1) {
                power_of_2.set_mask(size - 1, true);
            }
            
            REQUIRE(power_of_2.get_mask(0) == true);
            if (size > 1) {
                REQUIRE(power_of_2.get_mask(size - 1) == true);
            }
        }
    }
}

#endif // USE_SYCL 