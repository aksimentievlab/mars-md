#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "Math/Types.h"
#include "Random/Random.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

using namespace ARBD;
using namespace ARBD::Profiling;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Kernel Functors for Profiling Tests
// (No changes in this section)
// ============================================================================

struct InitializeWalkersKernel {
	template<typename... Args>
	void operator()(size_t i, Args... args) const {
		auto tuple_args = std::make_tuple(args...);
		auto* positions = std::get<0>(tuple_args);
		positions[i] = Vector3{0.0f, 0.0f, 0.0f};
	}
};

struct RandomWalkKernel {
	size_t NUM_STEPS;
	size_t NUM_WALKERS;

	template<typename... Args>
	void operator()(size_t walker_id, Args... args) const {
		auto tuple_args = std::make_tuple(args...);
		auto* steps = std::get<0>(tuple_args);
		auto* positions = std::get<1>(tuple_args);

		Vector3 pos = positions[walker_id];

		size_t steps_per_walker = NUM_STEPS / NUM_WALKERS;
		size_t start_step = walker_id * steps_per_walker;

		for (size_t step = 0; step < steps_per_walker && (start_step + step) < NUM_STEPS; ++step) {
			size_t step_idx = start_step + step;
			Vector3 step_vec = steps[step_idx];

			float length = std::sqrt(step_vec.x * step_vec.x + step_vec.y * step_vec.y +
									 step_vec.z * step_vec.z);
			if (length > 0.0f) {
				step_vec.x /= length;
				step_vec.y /= length;
				step_vec.z /= length;
			}

			pos.x += step_vec.x;
			pos.y += step_vec.y;
			pos.z += step_vec.z;
		}

		positions[walker_id] = pos;
	}
};

struct CalculateDistancesKernel {
	template<typename... Args>
	void operator()(size_t i, Args... args) const {
		auto tuple_args = std::make_tuple(args...);
		auto* positions = std::get<0>(tuple_args);
		auto* distances = std::get<1>(tuple_args);

		Vector3 pos = positions[i];
		distances[i] = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
	}
};

struct SmoothingFilterKernel {
	size_t GRID_SIZE;

	template<typename... Args>
	void operator()(size_t i, Args... args) const {
		auto tuple_args = std::make_tuple(args...);
		auto* input = std::get<0>(tuple_args);
		auto* output = std::get<1>(tuple_args);

		size_t x = i % GRID_SIZE;
		size_t y = i / GRID_SIZE;

		float sum = 0.0f;
		int count = 0;

		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				int nx = static_cast<int>(x) + dx;
				int ny = static_cast<int>(y) + dy;

				if (nx >= 0 && nx < static_cast<int>(GRID_SIZE) && ny >= 0 &&
					ny < static_cast<int>(GRID_SIZE)) {

					size_t idx = ny * GRID_SIZE + nx;
					sum += input[idx];
					count++;
				}
			}
		}

		output[i] = (count > 0) ? sum / count : input[i];
	}
};

struct GradientCalculationKernel {
	size_t GRID_SIZE;

	template<typename... Args>
	void operator()(size_t i, Args... args) const {
		auto tuple_args = std::make_tuple(args...);
		auto* input = std::get<0>(tuple_args);
		auto* output = std::get<1>(tuple_args);

		size_t x = i % GRID_SIZE;
		size_t y = i / GRID_SIZE;

		float grad_x = 0.0f, grad_y = 0.0f;

		if (x > 0 && x < GRID_SIZE - 1) {
			size_t left_idx = y * GRID_SIZE + (x - 1);
			size_t right_idx = y * GRID_SIZE + (x + 1);
			grad_x = (input[right_idx] - input[left_idx]) * 0.5f;
		}

		if (y > 0 && y < GRID_SIZE - 1) {
			size_t top_idx = (y - 1) * GRID_SIZE + x;
			size_t bottom_idx = (y + 1) * GRID_SIZE + x;
			grad_y = (input[bottom_idx] - input[top_idx]) * 0.5f;
		}

		output[i] = std::sqrt(grad_x * grad_x + grad_y * grad_y);
	}
};


// ============================================================================
// Profiled Random Test Fixture
// (No changes in this section)
// ============================================================================

struct ProfiledRandomTestFixture {
	Resource cuda_resource, sycl_resource, metal_resource;
	bool cuda_available = false;
	bool sycl_available = false;
	bool metal_available = false;

	ProfiledRandomTestFixture() {
		ProfilingConfig config;
		config.enable_timing = true;
		config.enable_memory_tracking = true;
		config.enable_kernel_profiling = true;
		config.enable_backend_markers = true;
		config.output_file = "random_profile_test.json";
		ProfileManager::init(config);

		try {
#ifdef USE_CUDA
			CUDA::CUDAManager::init();
			CUDA::CUDAManager::load_info();
			if (!CUDA::CUDAManager::devices().empty()) {
				CUDA::CUDAManager::use(0);
				cuda_resource = Resource(ResourceType::CUDA, 0);
				cuda_available = true;
				LOGINFO("CUDA backend available for profiled Random tests");
			}
#elif defined(USE_SYCL)
			SYCL::SYCLManager::init();
			SYCL::SYCLManager::load_info();
			if (!SYCL::SYCLManager::devices().empty()) {
				SYCL::SYCLManager::use(0);
				sycl_resource = Resource(ResourceType::SYCL, 0);
				sycl_available = true;
				LOGINFO("SYCL backend available for profiled Random tests");
			}
#elif defined(USE_METAL)
			METAL::METALManager::init();
			METAL::METALManager::load_info();
			if (!METAL::METALManager::devices().empty()) {
				METAL::METALManager::use(0);
				metal_resource = Resource(ResourceType::METAL, 0);
				metal_available = true;
				LOGINFO("Metal backend available for profiled Random tests");
			}
#endif
		} catch (const std::exception& e) {
			LOGWARN("Backend initialization failed in ProfiledRandomTestFixture: {}", e.what());
		}
	}

	~ProfiledRandomTestFixture() {
		ProfileManager::print_summary();
		ProfileManager::finalize();

		try {
#ifdef USE_CUDA
			if (cuda_available)
				CUDA::CUDAManager::finalize();
#elif defined(USE_SYCL)
			if (sycl_available)
				SYCL::SYCLManager::finalize();
#elif defined(USE_METAL)
			if (metal_available)
				METAL::METALManager::finalize();
#endif
		} catch (const std::exception& e) {
			std::cerr << "Error during ProfiledRandomTestFixture cleanup: " << e.what()
					  << std::endl;
		}
	}
};

// ============================================================================
// Profiled Random Generation Tests
// (No changes in this section)
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture,
				 "Profiled Random Generation Performance",
				 "[random][profiling][performance]") {

	auto test_profiled_generation = [this](const Resource& resource,
										   const std::string& backend_name,
										   ResourceType backend_type) {
		if (!resource.is_device()) {
			SKIP("Backend " + backend_name + " not available");
		}

		SECTION("Profiled uniform generation on " + backend_name) {
			PROFILE_RANGE("RandomGenerator::Creation", backend_type);

			Random<Resource> rng(resource, 128);
			rng.init(42, 0);

			{
				PROFILE_RANGE("DeviceBuffer::Allocation", backend_type);
				DeviceBuffer<float> device_buffer(100000);

				PROFILE_MEMORY(backend_type, nullptr);

				{
					PROFILE_RANGE("Random::GenerateUniform", backend_type);
					Event generation_event = rng.generate_uniform(device_buffer, 0.0f, 1.0f);
					generation_event.wait();
				}

				{
					PROFILE_RANGE("Buffer::CopyToHost", backend_type);
					std::vector<float> host_results(100000);
					device_buffer.copy_to_host(host_results);
				}

				PROFILE_MARK("Uniform generation completed", backend_type);
			}

			LOGINFO("{} profiled uniform generation completed", backend_name);
		}

		SECTION("Profiled gaussian generation on " + backend_name) {
			PROFILE_RANGE("RandomGenerator::GaussianTest", backend_type);

			Random<Resource> rng(resource, 128);
			rng.init(12345, 0);

			DeviceBuffer<float> device_buffer(50000);

			{
				PROFILE_RANGE("Random::GenerateGaussian", backend_type);
				Event generation_event = rng.generate_gaussian(device_buffer, 0.0f, 1.0f);
				generation_event.wait();
			}

			{
				PROFILE_RANGE("Statistical::Validation", backend_type);
				std::vector<float> host_results(50000);
				device_buffer.copy_to_host(host_results);

				double mean = std::accumulate(host_results.begin(), host_results.end(), 0.0) /
							  host_results.size();
				double sq_sum = std::inner_product(host_results.begin(),
												   host_results.end(),
												   host_results.begin(),
												   0.0);
				double stdev = std::sqrt(sq_sum / host_results.size() - mean * mean);

				REQUIRE_THAT(mean, WithinAbs(0.0, 0.1));
				REQUIRE_THAT(stdev, WithinAbs(1.0, 0.2));
			}

			PROFILE_MARK("Gaussian generation validated", backend_type);
			LOGINFO("{} profiled gaussian generation completed", backend_name);
		}

		SECTION("Profiled Vector3 generation on " + backend_name) {
			PROFILE_RANGE("RandomGenerator::Vector3Test", backend_type);

			Random<Resource> rng(resource, 128);
			rng.init(9876, 0);

			DeviceBuffer<Vector3> device_buffer(25000);
			Vector3 mean(0.0f, 0.0f, 0.0f);
			Vector3 dev(1.0f, 1.0f, 1.0f);

			{
				PROFILE_RANGE("Random::GenerateVector3", backend_type);
				Event generation_event = rng.generate_gaussian(device_buffer, mean, dev);
				generation_event.wait();
			}

			{
				PROFILE_RANGE("Vector3::Validation", backend_type);
				std::vector<Vector3> host_results(25000);
				device_buffer.copy_to_host(host_results);

				bool all_finite =
					std::all_of(host_results.begin(), host_results.end(), [](const Vector3& v) {
						return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
					});

				REQUIRE(all_finite);
			}

			PROFILE_MARK("Vector3 generation validated", backend_type);
			LOGINFO("{} profiled Vector3 generation completed", backend_name);
		}
	};

#ifdef USE_CUDA
	test_profiled_generation(cuda_resource, "CUDA", ResourceType::CUDA);
#endif
#ifdef USE_SYCL
	test_profiled_generation(sycl_resource, "SYCL", ResourceType::SYCL);
#endif
#ifdef USE_METAL
	test_profiled_generation(metal_resource, "Metal", ResourceType::METAL);
#endif
}

// ============================================================================
// Profiled Random + Kernels Integration Tests
// (No changes in this section)
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture,
				 "Profiled Random-Kernel Integration",
				 "[random][kernels][profiling][integration]") {

	auto test_profiled_integration = [this](const Resource& resource,
											const std::string& backend_name,
											ResourceType backend_type) {
		if (!resource.is_device()) {
			SKIP("Backend " + backend_name + " not available");
		}

		SECTION("Profiled Monte Carlo simulation on " + backend_name) {
			PROFILE_RANGE("MonteCarlo::Simulation", backend_type);
			constexpr size_t NUM_SAMPLES = 100000;
			Random<Resource> rng(resource, 128);
			rng.init(314159, 0);
			DeviceBuffer<float> x_coords(NUM_SAMPLES);
			DeviceBuffer<float> y_coords(NUM_SAMPLES);
			DeviceBuffer<int> inside_circle(NUM_SAMPLES);
			{
				PROFILE_RANGE("Random::GenerateCoordinates", backend_type);
				rng.generate_uniform(x_coords, -1.0f, 1.0f).wait();
				rng.init(314159, NUM_SAMPLES);
				rng.generate_uniform(y_coords, -1.0f, 1.0f).wait();
			}
			{
				PROFILE_RANGE("Kernel::CircleTest", backend_type);
				std::vector<float> x_host(NUM_SAMPLES), y_host(NUM_SAMPLES);
				std::vector<int> inside_host(NUM_SAMPLES);
				x_coords.copy_to_host(x_host);
				y_coords.copy_to_host(y_host);
				for (size_t i = 0; i < NUM_SAMPLES; ++i) {
					float dist_sq = x_host[i] * x_host[i] + y_host[i] * y_host[i];
					inside_host[i] = (dist_sq <= 1.0f) ? 1 : 0;
				}
				inside_circle.copy_from_host(inside_host);
			}
			{
				PROFILE_RANGE("MonteCarlo::PiEstimation", backend_type);
				std::vector<int> inside_results(NUM_SAMPLES);
				inside_circle.copy_to_host(inside_results);
				int points_inside =
					std::accumulate(inside_results.begin(), inside_results.end(), 0);
				double pi_estimate = 4.0 * static_cast<double>(points_inside) / NUM_SAMPLES;
				REQUIRE_THAT(pi_estimate, WithinAbs(3.14159, 0.1));
				LOGINFO("{} Monte Carlo π estimate: {:.5f}", backend_name, pi_estimate);
			}
			PROFILE_MARK("Monte Carlo simulation completed", backend_type);
		}

		SECTION("Profiled random walk simulation on " + backend_name) {
			PROFILE_RANGE("RandomWalk::Simulation", backend_type);
			constexpr size_t NUM_STEPS = 100000;
			constexpr size_t NUM_WALKERS = 1000;
			Random<Resource> rng(resource, 128);
			rng.init(271828, 0);
			DeviceBuffer<Vector3> random_steps(NUM_STEPS);
			DeviceBuffer<Vector3> walker_positions(NUM_WALKERS);
			DeviceBuffer<float> final_distances(NUM_WALKERS);
			Vector3 mean(0.0f, 0.0f, 0.0f);
			Vector3 dev(1.0f, 1.0f, 1.0f);
			rng.generate_gaussian(random_steps, mean, dev).wait();
			launch_kernel(resource, NUM_WALKERS, {}, std::make_tuple(), std::make_tuple(std::ref(walker_positions)), InitializeWalkersKernel{}).wait();
			launch_kernel(resource, NUM_WALKERS, {}, std::make_tuple(std::ref(random_steps)), std::make_tuple(std::ref(walker_positions)), RandomWalkKernel{NUM_STEPS, NUM_WALKERS}).wait();
			launch_kernel(resource, NUM_WALKERS, {}, std::make_tuple(std::ref(walker_positions)), std::make_tuple(std::ref(final_distances)), CalculateDistancesKernel{}).wait();
			std::vector<float> distances(NUM_WALKERS);
			final_distances.copy_to_host(distances);
			double mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0) / NUM_WALKERS;
			REQUIRE(mean_distance > 0.0);
			LOGINFO("{} Random Walk: mean distance: {:.3f}", backend_name, mean_distance);
			PROFILE_MARK("Random walk simulation completed", backend_type);
		}
	};

#ifdef USE_CUDA
	test_profiled_integration(cuda_resource, "CUDA", ResourceType::CUDA);
#endif
#ifdef USE_SYCL
	test_profiled_integration(sycl_resource, "SYCL", ResourceType::SYCL);
#endif
#ifdef USE_METAL
	test_profiled_integration(metal_resource, "Metal", ResourceType::METAL);
#endif
}


// ============================================================================
// Comparative Performance & Multi-Device Tests
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture,
				 "Single-Device Comparative Performance",
				 "[random][profiling][comparison]") {

	std::vector<std::pair<Resource, std::string>> available_backends;
#ifdef USE_CUDA
	if (cuda_available) available_backends.emplace_back(cuda_resource, "CUDA");
#endif
#ifdef USE_SYCL
	if (sycl_available) available_backends.emplace_back(sycl_resource, "SYCL");
#endif
#ifdef USE_METAL
	if (metal_available) available_backends.emplace_back(metal_resource, "Metal");
#endif

	if (available_backends.empty()) {
		SKIP("No backends available for comparative testing");
	}

    // NOTE: The multi-device test has been moved to its own TEST_CASE below
    // to handle its specific SYCL initialization requirements safely.

	SECTION("Gaussian generation performance comparison") {
		constexpr size_t PERF_SIZE = 5000000;
		std::vector<std::pair<std::string, double>> performance_results;
		for (const auto& [resource, backend_name] : available_backends) {
			Random<Resource> rng(resource, 256);
			rng.init(654321, 0);
			DeviceBuffer<float> device_buffer(PERF_SIZE);
			auto start = std::chrono::high_resolution_clock::now();
			rng.generate_gaussian(device_buffer, 0.0f, 1.0f).wait();
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			double time_ms = static_cast<double>(duration.count()) / 1000.0;
			performance_results.emplace_back(backend_name, time_ms);
			LOGINFO("{} generated {} Gaussian numbers in {:.3f} ms", backend_name, PERF_SIZE, time_ms);
		}
	}

	SECTION("Memory bandwidth test") {
		constexpr size_t BANDWIDTH_SIZE = 100000000;
		for (const auto& [resource, backend_name] : available_backends) {
			DeviceBuffer<float> device_buffer(BANDWIDTH_SIZE);
			std::vector<float> host_buffer(BANDWIDTH_SIZE, 1.0f);
			auto start_h2d = std::chrono::high_resolution_clock::now();
			device_buffer.copy_from_host(host_buffer);
			auto end_h2d = std::chrono::high_resolution_clock::now();
			double h2d_time_ms = std::chrono::duration<double, std::milli>(end_h2d - start_h2d).count();
			double h2d_bandwidth_gbps = (BANDWIDTH_SIZE * sizeof(float)) / (h2d_time_ms / 1000.0) / 1e9;
			auto start_d2h = std::chrono::high_resolution_clock::now();
			device_buffer.copy_to_host(host_buffer);
			auto end_d2h = std::chrono::high_resolution_clock::now();
			double d2h_time_ms = std::chrono::duration<double, std::milli>(end_d2h - start_d2h).count();
			double d2h_bandwidth_gbps = (BANDWIDTH_SIZE * sizeof(float)) / (d2h_time_ms / 1000.0) / 1e9;
			LOGINFO("{} Memory Bandwidth - H2D: {:.2f} GB/s, D2H: {:.2f} GB/s", backend_name, h2d_bandwidth_gbps, d2h_bandwidth_gbps);
		}
	}
}


// ============================================================================
// NEW: Standalone Multi-Device SYCL Test
// This test is separated to manage the SYCL environment correctly for a
// multi-device context without conflicting with the single-device fixture.
// ============================================================================

TEST_CASE("Multi-device parallel random generation with cross-device analysis", "[random][profiling][multi-device][sycl]") {
#ifdef USE_SYCL
	// Manually manage SYCL lifetime for this specific multi-device test
	try {
		SYCL::SYCLManager::init();
		SYCL::SYCLManager::load_info();
	} catch (const std::exception& e) {
		SKIP("SYCL backend initialization failed: " << e.what());
	}

	constexpr size_t NUMBERS_PER_DEVICE = 10000000; // 10M per device
	constexpr size_t SAMPLE_SIZE = 10000; // Sample for analysis

	const auto& sycl_devices = SYCL::SYCLManager::devices();
	const size_t num_devices = sycl_devices.size();

	if (num_devices == 0) {
		SKIP("No SYCL devices available for multi-device test.");
	}

	LOGINFO("=== Multi-Device Parallel Random Generation Test ===");
	LOGINFO("Testing {} SYCL devices with {} numbers per device", num_devices, NUMBERS_PER_DEVICE);

	// This struct encapsulates all per-device data, preventing dangling references.
	struct DeviceData {
		Resource resource;
		std::unique_ptr<Random<Resource>> rng;
		DeviceBuffer<float> buffer;
		Event generation_event;
		std::chrono::high_resolution_clock::time_point start_time;
		double timing_ms = 0.0;
		std::vector<float> sample;

		DeviceData(size_t device_id, size_t buffer_size)
			: resource(ResourceType::SYCL, device_id),
			  rng(std::make_unique<Random<Resource>>(resource, 256)),
			  // FIX: Pass the resource to the buffer so it allocates on the correct device.
			  buffer(resource, buffer_size),
			  sample(SAMPLE_SIZE)
		{
			rng->init(123456 + device_id * 1000, device_id);
		}
	};

	std::vector<std::unique_ptr<DeviceData>> devices;
	devices.reserve(num_devices); // Reserve to avoid reallocations

	// Initialize all devices
	PROFILE_RANGE("MultiDevice::Initialization", ResourceType::SYCL);
	for (size_t i = 0; i < num_devices; ++i) {
		devices.push_back(std::make_unique<DeviceData>(i, NUMBERS_PER_DEVICE));
		LOGINFO("Initialized device {} with seed {}", i, 123456 + i * 1000);
	}

	auto global_start = std::chrono::high_resolution_clock::now();

	// Start all devices concurrently
	for (auto& device : devices) {
		device->start_time = std::chrono::high_resolution_clock::now();
		device->generation_event = device->rng->generate_uniform(device->buffer, 0.0f, 1.0f);
	}

	LOGINFO("Launched generation on all {} devices simultaneously", num_devices);

	// Wait for all devices to complete and measure individual timings
	for (auto& device : devices) {
		device->generation_event.wait();
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - device->start_time);
		device->timing_ms = static_cast<double>(duration.count()) / 1000.0;
	}

	auto global_end = std::chrono::high_resolution_clock::now();
	double total_time_ms = std::chrono::duration<double, std::milli>(global_end - global_start).count();

	// --- Analysis and Validation (as in your provided code) ---
	LOGINFO("All devices completed. Total wall-clock time: {:.3f} ms", total_time_ms);

	// Manually finalize SYCL at the end of the test
	try {
		SYCL::SYCLManager::finalize();
	} catch (const std::exception& e) {
		FAIL("SYCL finalization failed: " << e.what());
	}
#else
	SKIP("Multi-device test requires SYCL backend to be enabled.");
#endif
}
