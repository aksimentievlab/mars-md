// Initialize Metal

#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"

TEST_CASE("Metal Vector Addition", "[metal][kernels][simple_test]") {
	// Metal manager is already initialized by the test framework
	ARBD::METAL::Manager::load_info(); 

	// Check if Metal library is loaded
	auto* library = ARBD::METAL::Manager::get_library();
	if (!library) {
		FAIL("Metal library not loaded - cannot run kernel test");
	}
	
	INFO("Metal library loaded successfully");

	try {
		// Use Metal resource instead of CPU
		ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
		const size_t n = 1000;  // Test with just 1 element first // Very small test size for debugging

		auto buffer_a = ARBD::DeviceBuffer<float>(n);
		auto buffer_b = ARBD::DeviceBuffer<float>(n);
		auto buffer_result = ARBD::DeviceBuffer<float>(n);

		// Initialize data
		std::vector<float> host_a(n, 1.0f);
		std::vector<float> host_b(n, 2.0f);
		std::vector<float> host_result_init(n, 0.0f); // Initialize result buffer
		
		buffer_a.copy_from_host(host_a.data(), n);
		buffer_b.copy_from_host(host_b.data(), n);
		buffer_result.copy_from_host(host_result_init.data(), n);

		// Verify data was copied correctly
		std::vector<float> verify_a(n), verify_b(n);
		buffer_a.copy_to_host(verify_a.data(), n);
		buffer_b.copy_to_host(verify_b.data(), n);
		
		INFO("Buffer A data: " << verify_a[0] << ", " << verify_a[1]);
		INFO("Buffer B data: " << verify_b[0] << ", " << verify_b[1]);
		
		REQUIRE(verify_a[0] == 1.0f);
		REQUIRE(verify_b[0] == 2.0f);

		ARBD::KernelConfig config;
		config.async = false;  // Force synchronous execution for testing

		INFO("Launching Metal kernel...");
		INFO("Thread count: " << n);
		INFO("Buffer A size: " << buffer_a.size());
		INFO("Buffer B size: " << buffer_b.size());
		INFO("Buffer result size: " << buffer_result.size());
		
		// Check if buffers are properly allocated
		INFO("Buffer A device pointer: " << buffer_a.data());
		INFO("Buffer B device pointer: " << buffer_b.data());
		INFO("Buffer result device pointer: " << buffer_result.data());
		
		// Check if Metal buffers can be retrieved
		auto* metal_buffer_a = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_a.data());
		auto* metal_buffer_b = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_b.data());
		auto* metal_buffer_result = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_result.data());
		
		INFO("Metal buffer A: " << (void*)metal_buffer_a);
		INFO("Metal buffer B: " << (void*)metal_buffer_b);
		INFO("Metal buffer result: " << (void*)metal_buffer_result);
		
		// Verify that Metal buffer contents pointers match device pointers
		if (metal_buffer_result) {
			void* metal_contents = metal_buffer_result->contents();
			INFO("Metal buffer result contents pointer: " << metal_contents);
			if (metal_contents != buffer_result.data()) {
				INFO("ERROR: Metal buffer contents pointer != device buffer pointer");
				INFO("Expected: " << buffer_result.data() << ", Got: " << metal_contents);
			}
		}
		
		// Double-check buffer pointer before kernel launch
		INFO("Buffer result pointer just before kernel launch: " << buffer_result.data());
		
		ARBD::Event event = ARBD::launch_metal_kernel(
			metal_res,
			n,
			std::make_tuple(buffer_a, buffer_b),  // No input buffers for debug kernel
			std::forward_as_tuple(buffer_result),  // Use forward_as_tuple to avoid copy
			config,
			"scalar_add",
			nullptr
		);
		
		// Check buffer pointer after kernel launch
		INFO("Buffer result pointer just after kernel launch: " << buffer_result.data());	

		INFO("Waiting for kernel completion...");
		event.wait();
		INFO("Kernel execution completed");

		// Try reading the Metal buffer contents directly after kernel execution  
		if (metal_buffer_result) {
			float* metal_contents = (float*)metal_buffer_result->contents();
			INFO("Metal buffer contents direct access after execution: " << metal_contents[0]);
		}
		
		// Also try reading buffer data directly (unified memory)
		INFO("Buffer data direct access: " << *((float*)buffer_result.data()));
		
		// Double-check that we still have the same Metal buffer mapping
		auto* metal_buffer_result_after = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_result.data());
		INFO("Metal buffer result after kernel: " << (void*)metal_buffer_result_after);
		if (metal_buffer_result != metal_buffer_result_after) {
			INFO("WARNING: Metal buffer mapping changed during kernel execution!");
		}
		
		// Get results and verify
		std::vector<float> result(n);
		buffer_result.copy_to_host(result.data(), n);

		INFO("Result data after copy_to_host: " << result[0] << ", " << result[1]);

		for (size_t i = 0; i < n; ++i) {
			INFO("Checking result[" << i << "] = " << result[i] << " (expected 42.0)");
			REQUIRE(result[i] == 3.0f); // Debug constant
		}

		REQUIRE(true); // Test passed

	} catch (const std::exception& e) {
		FAIL("Metal test failed with exception: " << e.what());
	}

	ARBD::METAL::Manager::finalize();
}