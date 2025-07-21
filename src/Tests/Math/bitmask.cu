#include "../catch_boiler.h"
#include "Math/Bitmask.h"
#include "Math/Types.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#include "../../extern/Catch2/extras/catch_amalgamated.hpp"

namespace Catch {
    template <>
    struct StringMaker<ARBD::Bitmask> {
        static std::string convert( ARBD::Bitmask const& value ) {
            return value.to_string();
        }
    };
}

namespace Tests::Bitmask {

    void run_basic_tests() {
        using T = ARBD::Bitmask;
        INFO( "Testing Bitmask functionally" );
        {
            int i = 10;
            T b = T(i);	// bitmask of length i
            for (int j: {0,3}) {  // Remove invalid indices
                if (j < i) b.set_mask(j,1);
            }

            REQUIRE( b.to_string() == "1001000000" );
            
            // Test equality
            T b2(i);
            b2.set_mask(0, 1);
            b2.set_mask(3, 1);
            REQUIRE( b == b2 );
        }
    }

    void run_backend_tests() {
        using T = ARBD::Bitmask;
        
        // Test different sizes
        for (size_t size : {8, 16, 32, 64, 128, 256}) {
            T host_bitmask(size);
            
            // Set a pattern
            for (size_t i = 0; i < size; i += 4) {
                host_bitmask.set_mask(i, true);
            }
            
#ifdef USE_CUDA
            // Test CUDA backend operations
            ARBD::Resource cuda_resource(ARBD::ResourceType::CUDA, 0);
            T* device_bitmask = host_bitmask.send_to_backend(cuda_resource);
            REQUIRE(device_bitmask != nullptr);
            
            T received_bitmask = T::receive_from_backend(device_bitmask, cuda_resource);
            REQUIRE(host_bitmask == received_bitmask);
            
            T::remove_from_backend(device_bitmask, cuda_resource);
#endif
        }
    }

    void run_sparse_tests() {
        using SparseBitmask = ARBD::SparseBitmask<64>;
        
        // Test large sparse bitmask
        SparseBitmask sparse(100000);
        
        // Set sparse bits
        std::vector<size_t> test_indices = {0, 1000, 10000, 50000, 99999};
        for (size_t idx : test_indices) {
            sparse.set_mask(idx, true);
        }
        
        // Verify bits are set
        for (size_t idx : test_indices) {
            REQUIRE(sparse.get_mask(idx) == true);
        }
        
        // Verify unset bits are false
        for (size_t idx : {1, 500, 5000, 25000, 75000}) {
            REQUIRE(sparse.get_mask(idx) == false);
        }
        
        // Check efficiency
        INFO("SparseBitmask allocated " << sparse.get_allocated_chunks() << " chunks for 100000 bits");
        REQUIRE(sparse.get_allocated_chunks() <= test_indices.size());
    }

    void run_stress_tests() {
        using T = ARBD::Bitmask;
        
        // Test large bitmask
        const size_t large_size = 10000;
        T large_bitmask(large_size);
        
        // Set every 100th bit
        for (size_t i = 0; i < large_size; i += 100) {
            large_bitmask.set_mask(i, true);
        }
        
        // Verify pattern
        for (size_t i = 0; i < large_size; ++i) {
            bool expected = (i % 100 == 0);
            REQUIRE(large_bitmask.get_mask(i) == expected);
        }
        
        // Test string representation
        std::string bit_string = large_bitmask.to_string();
        REQUIRE(bit_string.length() == large_size);
        
        // Verify pattern in string
        for (size_t i = 0; i < large_size; ++i) {
            char expected = (i % 100 == 0) ? '1' : '0';
            REQUIRE(bit_string[i] == expected);
        }
    }

    void run_edge_case_tests() {
        using T = ARBD::Bitmask;
        
        // Zero-length bitmask
        T empty_bitmask(0);
        REQUIRE(empty_bitmask.get_len() == 0);
        
        // Single bit
        T single_bit(1);
        REQUIRE(single_bit.get_len() == 1);
        single_bit.set_mask(0, true);
        REQUIRE(single_bit.get_mask(0) == true);
        REQUIRE(single_bit.to_string() == "1");
        
        // Power of 2 sizes
        for (size_t size : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
            T power_of_2(size);
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
            
            // Test string representation
            std::string str = power_of_2.to_string();
            REQUIRE(str.length() == size);
            REQUIRE(str[0] == '1');
            if (size > 1) {
                REQUIRE(str[size - 1] == '1');
            }
        }
    }

    TEST_CASE( "Testing Bitmask for CPU consistency", "[Tests::Bitmask]" ) {
        run_basic_tests();
    }

    TEST_CASE( "Testing Bitmask backend operations", "[Tests::Bitmask][backend]" ) {
        run_backend_tests();
    }

    TEST_CASE( "Testing SparseBitmask functionality", "[Tests::Bitmask][sparse]" ) {
        run_sparse_tests();
    }

    TEST_CASE( "Testing Bitmask stress tests", "[Tests::Bitmask][stress]" ) {
        run_stress_tests();
    }

    TEST_CASE( "Testing Bitmask edge cases", "[Tests::Bitmask][edge]" ) {
        run_edge_case_tests();
    }

    TEST_CASE( "Testing built-in Bitmask test functions", "[Tests::Bitmask][builtin]" ) {
        REQUIRE(ARBD::BitmaskTests::run_all_tests());
    }

    TEST_CASE( "Testing Bitmask atomic operations", "[Tests::Bitmask][atomic]" ) {
        using T = ARBD::Bitmask;
        
        T bitmask(32);
        
        // Test atomic set operations
        bitmask.set_mask(0, true);
        bitmask.set_mask(15, true);
        bitmask.set_mask(31, true);
        
        REQUIRE(bitmask.get_mask(0) == true);
        REQUIRE(bitmask.get_mask(15) == true);
        REQUIRE(bitmask.get_mask(31) == true);
        
        // Test atomic clear operations
        bitmask.set_mask(15, false);
        REQUIRE(bitmask.get_mask(15) == false);
        
        // Other bits should remain unchanged
        REQUIRE(bitmask.get_mask(0) == true);
        REQUIRE(bitmask.get_mask(31) == true);
    }

}
