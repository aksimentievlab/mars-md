#include "../catch_boiler.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Resource.h"
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
using Catch::Approx;
using Catch::Matchers::Contains;

TEST_CASE("CUDAManager Basic Initialization", "[CUDAManager][Backend]") {
	SECTION("Initialize and discover devices") {
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::init());

		// Check that we found some devices
		const auto& all_devices = ARBD::CUDA::CUDAManager::all_devices();
		REQUIRE(!all_devices.empty());
		LOGINFO("Found {} CUDA devices", all_devices.size());

		// Check that at least one device is usable
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::load_info());
		const auto& devices = ARBD::CUDA::CUDAManager::devices();
		REQUIRE(!devices.empty());
	}

	SECTION("Device properties validation") {
		ARBD::CUDA::CUDAManager::init();
		ARBD::CUDA::CUDAManager::load_info();

		const auto& devices = ARBD::CUDA::CUDAManager::devices();
		for (size_t i = 0; i < devices.size(); ++i) {
			const auto& device = devices[i];
			const auto& props = device.properties();

			INFO("Testing device " << i << ": " << props.name);

			// Basic property checks
			REQUIRE(props.totalGlobalMem > 0);
			REQUIRE(props.multiProcessorCount > 0);
			REQUIRE(props.maxThreadsPerBlock > 0);
			REQUIRE(props.major >= 1); // Minimum compute capability

			// Check convenience accessors
			REQUIRE(device.total_memory() == props.totalGlobalMem);
			REQUIRE(device.compute_capability_major() == props.major);
			REQUIRE(device.compute_capability_minor() == props.minor);
			REQUIRE(device.multiprocessor_count() == props.multiProcessorCount);
			REQUIRE(device.max_threads_per_block() == props.maxThreadsPerBlock);

			LOGINFO("Device {}: {} | SM {}.{} | {:.1f}GB | {} MPs | {} max threads/block",
					i,
					props.name,
					props.major,
					props.minor,
					static_cast<float>(props.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f),
					props.multiProcessorCount,
					props.maxThreadsPerBlock);
		}
	}
}

TEST_CASE("CUDAManager Device Selection and Usage", "[CUDAManager][Backend]") {
	ARBD::CUDA::CUDAManager::init();
	ARBD::CUDA::CUDAManager::load_info();

	const auto& devices = ARBD::CUDA::CUDAManager::devices();
	REQUIRE(!devices.empty());

	SECTION("Device selection") {
		// Test using device 0
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::use(0));
		int current_id = ARBD::CUDA::CUDAManager::current();
		REQUIRE((current_id == devices[0].id()));

		// Test current device access
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::get_current_device());
		const auto& current_device = ARBD::CUDA::CUDAManager::get_current_device();
		REQUIRE((current_device.id() == devices[0].id()));

		// Test cycling through devices if multiple available
		if (devices.size() > 1) {
			REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::use(1));
			current_id = ARBD::CUDA::CUDAManager::current();
			REQUIRE((current_id == devices[1].id()));

			// Test wraparound
			int wrapped_id = static_cast<int>(devices.size());
			REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::use(wrapped_id));
			current_id = ARBD::CUDA::CUDAManager::current();
			REQUIRE((current_id == devices[0].id())); // Should wrap to 0
		}
	}

	SECTION("Device selection by IDs") {
		// Create a vector with device IDs
		std::vector<unsigned int> device_ids;
		for (const auto& device : devices) {
			device_ids.push_back(device.id());
		}

		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::select_devices(device_ids));

		// Verify devices are still accessible
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::load_info());
		const auto& selected_devices = ARBD::CUDA::CUDAManager::devices();
		REQUIRE(selected_devices.size() == device_ids.size());
	}
}

TEST_CASE("CUDAManager Stream Management", "[CUDAManager][Backend]") {
	ARBD::CUDA::CUDAManager::init();
	ARBD::CUDA::CUDAManager::load_info();
	ARBD::CUDA::CUDAManager::use(0);

	auto& device = ARBD::CUDA::CUDAManager::get_current_device();

	SECTION("Stream access") {
		// Test getting specific streams
		for (size_t i = 0; i < ARBD::CUDA::CUDAManager::NUM_STREAMS; ++i) {
			cudaStream_t stream = device.get_stream(i);
			REQUIRE(stream != nullptr);
		}

		// Test next stream cycling
		std::vector<cudaStream_t> seen_streams;
		for (size_t i = 0; i < ARBD::CUDA::CUDAManager::NUM_STREAMS * 2; ++i) {
			cudaStream_t stream = device.get_next_stream();
			REQUIRE(stream != nullptr);
			// Only add unique streams
			if (std::find(seen_streams.begin(), seen_streams.end(), stream) == seen_streams.end()) {
				seen_streams.push_back(stream);
			}
		}
		// Should have seen all unique streams
		REQUIRE(seen_streams.size() == ARBD::CUDA::CUDAManager::NUM_STREAMS);
	}

	SECTION("Stream synchronization") {
		// This should not throw
		REQUIRE_NOTHROW(device.synchronize_all_streams());
	}
}

TEST_CASE("CUDAManager Synchronization", "[CUDAManager][Backend]") {
	ARBD::CUDA::CUDAManager::init();
	ARBD::CUDA::CUDAManager::load_info();

	SECTION("Single device sync") {
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::sync(0));
	}

	SECTION("All devices sync") {
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::sync());
	}

	SECTION("Invalid device sync") {
		const auto& devices = ARBD::CUDA::CUDAManager::devices();
		int invalid_id = static_cast<int>(devices.size());

		// sync() with invalid device ID should throw (unlike use() which wraps around)
		REQUIRE_THROWS_AS(ARBD::CUDA::CUDAManager::sync(invalid_id), ARBD::Exception);
	}

	SECTION("use() with invalid device ID wraps around") {
		const auto& devices = ARBD::CUDA::CUDAManager::devices();
		int invalid_id = static_cast<int>(devices.size());

		// use() doesn't throw - it wraps around using modulo
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::use(invalid_id));
		int current_id = ARBD::CUDA::CUDAManager::current();
		REQUIRE((current_id == devices[0].id())); // Should wrap to 0
	}
}

TEST_CASE("CUDAManager Safe Device Handling", "[CUDAManager][Backend]") {
	ARBD::CUDA::CUDAManager::init();

	const auto& all_devices = ARBD::CUDA::CUDAManager::all_devices();
	const auto& safe_devices = ARBD::CUDA::CUDAManager::safe_devices();

	SECTION("Safe device filtering") {
		// Safe devices should be subset of all devices
		REQUIRE(safe_devices.size() <= all_devices.size());

		// All safe devices should have timeout disabled
		for (const auto& device : safe_devices) {
			REQUIRE_FALSE(device.may_timeout());
		}

		LOGINFO("Found {} safe devices out of {} total", safe_devices.size(), all_devices.size());
	}

	SECTION("Prefer safe devices") {
		// Test setting preference
		ARBD::CUDA::CUDAManager::prefer_safe_devices(true);
		ARBD::CUDA::CUDAManager::load_info();

		if (!safe_devices.empty()) {
			const auto& devices = ARBD::CUDA::CUDAManager::devices();
			REQUIRE(devices.size() == safe_devices.size());
		}

		// Reset to all devices
		ARBD::CUDA::CUDAManager::prefer_safe_devices(false);
		ARBD::CUDA::CUDAManager::load_info();

		const auto& all_loaded_devices = ARBD::CUDA::CUDAManager::devices();
		REQUIRE(all_loaded_devices.size() == all_devices.size());
	}
}

TEST_CASE("CUDAManager Peer Access", "[CUDAManager][Backend]") {
	ARBD::CUDA::CUDAManager::init();
	ARBD::CUDA::CUDAManager::load_info();

	const auto& devices = ARBD::CUDA::CUDAManager::devices();

	SECTION("Peer access matrix") {
		if (devices.size() > 1) {
			const auto& peer_matrix = ARBD::CUDA::CUDAManager::peer_access_matrix();

			// Matrix should be square
			REQUIRE(peer_matrix.size() == devices.size());
			for (const auto& row : peer_matrix) {
				REQUIRE(row.size() == devices.size());
			}

			// Diagonal should be false (device can't have peer access to itself)
			for (size_t i = 0; i < devices.size(); ++i) {
				REQUIRE_FALSE(peer_matrix[i][i]);
			}

			LOGINFO("Peer access matrix ({0}x{0}):", devices.size());
			for (size_t i = 0; i < devices.size(); ++i) {
				std::string row_str;
				for (size_t j = 0; j < devices.size(); ++j) {
					row_str += peer_matrix[i][j] ? "1 " : "0 ";
				}
				LOGINFO("  [{}] {}", i, row_str.c_str());
			}
		} else {
			LOGINFO("Only one device available, skipping peer access tests");
		}
	}
}

TEST_CASE("CUDAManager Resource Management", "[CUDAManager][Backend]") {
	ARBD::CUDA::CUDAManager::init();
	ARBD::CUDA::CUDAManager::load_info();
	ARBD::CUDA::CUDAManager::use(0);

	SECTION("Memory allocation basic test") {
		constexpr size_t test_size = 1024 * sizeof(float);
		void* ptr = nullptr;

		// Test allocation
		REQUIRE_NOTHROW(cudaMalloc(&ptr, test_size));
		REQUIRE_FALSE(ptr == nullptr);

		// Test deallocation
		REQUIRE_NOTHROW(cudaFree(ptr));
	}

	SECTION("Memory copy test") {
		constexpr size_t test_size = 1024;
		std::vector<float> host_data(test_size, 3.14f);
		std::vector<float> result_data(test_size, 0.0f);

		float* device_ptr = nullptr;
		REQUIRE_NOTHROW(cudaMalloc((void**)&device_ptr, test_size * sizeof(float)));

		// Host to device
		REQUIRE_NOTHROW(cudaMemcpy(device_ptr,
								   host_data.data(),
								   test_size * sizeof(float),
								   cudaMemcpyHostToDevice));

		// Device to host
		REQUIRE_NOTHROW(cudaMemcpy(result_data.data(),
								   device_ptr,
								   test_size * sizeof(float),
								   cudaMemcpyDeviceToHost));

		// Verify data
		for (size_t i = 0; i < test_size; ++i) {
			REQUIRE(result_data[i] == Approx(3.14f));
		}

		REQUIRE_NOTHROW(cudaFree(device_ptr));
	}
}

TEST_CASE("CUDAManager Finalization", "[CUDAManager][Backend]") {
	// Initialize first
	ARBD::CUDA::CUDAManager::init();
	ARBD::CUDA::CUDAManager::load_info();

	SECTION("Clean finalization") {
		REQUIRE_NOTHROW(ARBD::CUDA::CUDAManager::finalize());

		// After finalization, devices should be empty
		const auto& devices = ARBD::CUDA::CUDAManager::devices();
		REQUIRE(devices.empty());
	}
}

#else
TEST_CASE("CUDA Backend Not Available", "[CUDAManager][Backend]") {
	SKIP("CUDA backend not compiled in");
}
#endif // USE_CUDA
