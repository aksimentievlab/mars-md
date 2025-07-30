#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "Kernel_for_test.h"
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
// Profiled Random Test Fixture
// ============================================================================

struct ProfiledRandomTestFixture {
	Resource cuda_resource, sycl_resource, metal_resource;
	bool cuda_available = false;
	bool sycl_available = false;
	bool metal_available = false;

	ProfiledRandomTestFixture() {
		// Initialize profiling
		ProfilingConfig config;
		config.enable_timing = true;
		config.enable_memory_tracking = true;
		config.enable_kernel_profiling = true;
		config.enable_backend_markers = true;
		config.output_file = "random_profile_test.json";
		ProfileManager::init(config);

		// Initialize only one backend at a time as per project rules
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

			// Profile memory allocation
			{
				PROFILE_RANGE("DeviceBuffer::Allocation", backend_type);
				DeviceBuffer<float> device_buffer(100000); // 100K elements

				PROFILE_MEMORY(backend_type, nullptr);

				// Profile random number generation
				{
					PROFILE_RANGE("Random::GenerateUniform", backend_type);
					Event generation_event = rng.generate_uniform(device_buffer, 0.0f, 1.0f);
					generation_event.wait();
				}

				// Profile memory copy
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
				device_buffer.copy_to_host(host_results.data(), host_results.size());

				// Quick statistical validation
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

			DeviceBuffer<ARBD::Vector3_t<float>> device_buffer(25000);
			ARBD::Vector3_t<float> mean(0.0f, 0.0f, 0.0f);
			ARBD::Vector3_t<float> dev(1.0f, 1.0f, 1.0f);

			{
				PROFILE_RANGE("Random::GenerateVector3", backend_type);
				Event generation_event = rng.generate_gaussian(device_buffer, mean, dev);
				generation_event.wait();
			}

			{
				PROFILE_RANGE("Vector3::Validation", backend_type);
				std::vector<ARBD::Vector3_t<float>> host_results(25000);
				device_buffer.copy_to_host(host_results);

				// Validate that we have proper Vector3 data
				bool all_finite = std::all_of(host_results.begin(),
											  host_results.end(),
											  [](const ARBD::Vector3_t<float>& v) {
												  return std::isfinite(v.x) && std::isfinite(v.y) &&
														 std::isfinite(v.z);
											  });

				REQUIRE(all_finite);
			}

			PROFILE_MARK("Vector3 generation validated", backend_type);
			LOGINFO("{} profiled Vector3 generation completed", backend_name);
		}
	};

	// Only test available backends - single backend usage as per project rules
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

			constexpr size_t NUM_SAMPLES = 100000; // 100K samples for good statistics

			Random<Resource> rng(resource, 128);
			rng.init(314159, 0);

			// First, test if random generation is working as expected
			DeviceBuffer<float> test_buffer(100);
			Event test_event = rng.generate_uniform(test_buffer, -1.0f, 1.0f);
			test_event.wait();

			std::vector<float> test_values(100);
			test_buffer.copy_to_host(test_values.data(), test_values.size());

			auto [test_min, test_max] = std::minmax_element(test_values.begin(), test_values.end());
			double test_mean =
				std::accumulate(test_values.begin(), test_values.end(), 0.0) / test_values.size();

			LOGINFO("Test random generation: min={:.3f}, max={:.3f}, mean={:.3f}",
					*test_min,
					*test_max,
					test_mean);

			// If this is showing values outside [-1, 1] or mean not near 0, we have found the issue

			DeviceBuffer<float> x_coords(NUM_SAMPLES);
			DeviceBuffer<float> y_coords(NUM_SAMPLES);
			DeviceBuffer<int> inside_circle(NUM_SAMPLES);

			// Generate random coordinates
			{
				PROFILE_RANGE("Random::GenerateCoordinates", backend_type);
				Event x_event = rng.generate_uniform(x_coords, -1.0f, 1.0f);
				x_event.wait(); // Wait for X generation to complete

				// Reinitialize with different seed/offset to ensure different sequence for Y
				rng.init(314159, NUM_SAMPLES); // Use different offset
				Event y_event = rng.generate_uniform(y_coords, -1.0f, 1.0f);
				y_event.wait();

				// Quick validation that random generation worked
				std::vector<float> quick_x_check(10), quick_y_check(10);
				x_coords.copy_to_host(quick_x_check.data(), 10);
				y_coords.copy_to_host(quick_y_check.data(), 10);

				bool x_in_range = std::all_of(quick_x_check.begin(),
											  quick_x_check.end(),
											  [](float x) { return x >= -1.0f && x <= 1.0f; });
				bool y_in_range = std::all_of(quick_y_check.begin(),
											  quick_y_check.end(),
											  [](float y) { return y >= -1.0f && y <= 1.0f; });

				if (!x_in_range || !y_in_range) {
					LOGWARN("Random numbers out of expected range! X in range: {}, Y in range: {}",
							x_in_range,
							y_in_range);
				}
			}

			// Check which points are inside unit circle using custom kernel
			{
				PROFILE_RANGE("Kernel::CircleTest", backend_type);

				std::vector<float> x_host(NUM_SAMPLES), y_host(NUM_SAMPLES);
				std::vector<int> inside_host(NUM_SAMPLES);

				x_coords.copy_to_host(x_host);
				y_coords.copy_to_host(y_host);

				// Debug: Check first few random numbers
				LOGINFO("First 10 x coordinates: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, "
						"{:.3f}, {:.3f}, {:.3f}, {:.3f}",
						x_host[0],
						x_host[1],
						x_host[2],
						x_host[3],
						x_host[4],
						x_host[5],
						x_host[6],
						x_host[7],
						x_host[8],
						x_host[9]);
				LOGINFO("First 10 y coordinates: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, "
						"{:.3f}, {:.3f}, {:.3f}, {:.3f}",
						y_host[0],
						y_host[1],
						y_host[2],
						y_host[3],
						y_host[4],
						y_host[5],
						y_host[6],
						y_host[7],
						y_host[8],
						y_host[9]);

				// Check range of all values
				auto [x_min, x_max] = std::minmax_element(x_host.begin(), x_host.end());
				auto [y_min, y_max] = std::minmax_element(y_host.begin(), y_host.end());
				LOGINFO("X range: [{:.3f}, {:.3f}], Y range: [{:.3f}, {:.3f}]",
						*x_min,
						*x_max,
						*y_min,
						*y_max);

				// Statistical analysis of the random numbers
				double x_mean = std::accumulate(x_host.begin(), x_host.end(), 0.0) / x_host.size();
				double y_mean = std::accumulate(y_host.begin(), y_host.end(), 0.0) / y_host.size();
				LOGINFO("X mean: {:.6f} (should be ~0.0), Y mean: {:.6f} (should be ~0.0)",
						x_mean,
						y_mean);

				int debug_count = 0;
				int total_positive_x = 0, total_positive_y = 0;
				double sum_dist_sq = 0.0;

				for (size_t i = 0; i < NUM_SAMPLES; ++i) {
					float dist_sq = x_host[i] * x_host[i] + y_host[i] * y_host[i];
					inside_host[i] = (dist_sq <= 1.0f) ? 1 : 0;

					// Additional statistics
					if (x_host[i] > 0)
						total_positive_x++;
					if (y_host[i] > 0)
						total_positive_y++;
					sum_dist_sq += dist_sq;

					// Debug first few calculations
					if (i < 10) {
						LOGINFO("Point {}: ({:.3f}, {:.3f}) -> dist_sq={:.3f}, inside={}",
								i,
								x_host[i],
								y_host[i],
								dist_sq,
								inside_host[i]);
					}
					if (inside_host[i] == 1)
						debug_count++;
				}

				double mean_dist_sq = sum_dist_sq / NUM_SAMPLES;
				LOGINFO("Debug: Found {} points inside circle out of {} samples",
						debug_count,
						NUM_SAMPLES);
				LOGINFO("Positive X: {} (should be ~{}), Positive Y: {} (should be ~{})",
						total_positive_x,
						NUM_SAMPLES / 2,
						total_positive_y,
						NUM_SAMPLES / 2);
				LOGINFO("Mean distance squared: {:.6f} (should be ~0.667 for uniform in [-1,1]²)",
						mean_dist_sq);

				inside_circle.copy_from_host(inside_host);
			}

			// Calculate π estimate
			{
				PROFILE_RANGE("MonteCarlo::PiEstimation", backend_type);
				std::vector<int> inside_results(NUM_SAMPLES);
				inside_circle.copy_to_host(inside_results);

				int points_inside =
					std::accumulate(inside_results.begin(), inside_results.end(), 0);
				double pi_estimate = 4.0 * static_cast<double>(points_inside) / NUM_SAMPLES;

				// Debug: Print statistics
				LOGINFO("Points inside circle: {} out of {} (ratio: {:.6f})",
						points_inside,
						NUM_SAMPLES,
						static_cast<double>(points_inside) / NUM_SAMPLES);

				// π should be approximately 3.14159
				// For Monte Carlo with 10K samples, expect reasonable accuracy
				// Fixed the bug where X and Y coordinates were identical
				REQUIRE_THAT(pi_estimate, WithinAbs(3.14159, 0.1));

				LOGINFO("{} Monte Carlo π estimate: {:.5f} (error: {:.5f})",
						backend_name,
						pi_estimate,
						std::abs(pi_estimate - 3.14159));
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
			// Generate random step directions
			{
				PROFILE_RANGE("Random::GenerateSteps", backend_type);
				Event steps_event = rng.generate_gaussian(random_steps, mean, dev);
				steps_event.wait();
			}

			// Initialize walker positions to origin
			{
				PROFILE_RANGE("Kernel::InitializeWalkers", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple();
				auto outputs = std::make_tuple(std::ref(walker_positions));

				Event init_event = launch_kernel(resource,
												 NUM_WALKERS,
												 config,
												 inputs,
												 outputs,
												 InitializeWalkersKernel{});

				init_event.wait();
			}

			// Simulate random walk
			{
				PROFILE_RANGE("Kernel::RandomWalk", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(std::ref(random_steps));
				auto outputs = std::make_tuple(std::ref(walker_positions));

				Event walk_event = launch_kernel(resource,
												 NUM_WALKERS,
												 config,
												 inputs,
												 outputs,
												 RandomWalkKernel{NUM_STEPS, NUM_WALKERS});

				walk_event.wait();
			}

			// Calculate final distances from origin
			{
				PROFILE_RANGE("Kernel::CalculateDistances", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(std::ref(walker_positions));
				auto outputs = std::make_tuple(std::ref(final_distances));

				Event distance_event = launch_kernel(resource,
													 NUM_WALKERS,
													 config,
													 inputs,
													 outputs,
													 CalculateDistancesKernel{});

				distance_event.wait();
			}

			// Analyze results
			{
				PROFILE_RANGE("RandomWalk::Analysis", backend_type);
				std::vector<float> distances(NUM_WALKERS);
				final_distances.copy_to_host(distances);

				double mean_distance =
					std::accumulate(distances.begin(), distances.end(), 0.0) / NUM_WALKERS;
				double max_distance = *std::max_element(distances.begin(), distances.end());

				// For a 3D random walk, mean distance should scale roughly as sqrt(N)
				double expected_distance = std::sqrt(NUM_STEPS / NUM_WALKERS);

				REQUIRE(mean_distance > 0.0);
				REQUIRE(max_distance > mean_distance);

				LOGINFO("{} Random Walk: {} walkers, mean distance: {:.3f}, max distance: {:.3f}, "
						"expected: {:.3f}",
						backend_name,
						NUM_WALKERS,
						mean_distance,
						max_distance,
						expected_distance);
			}

			PROFILE_MARK("Random walk simulation completed", backend_type);
		}

		SECTION("Profiled noise generation pipeline on " + backend_name) {
			PROFILE_RANGE("NoiseGeneration::Pipeline", backend_type);

			constexpr size_t GRID_SIZE = 256;
			constexpr size_t TOTAL_POINTS = GRID_SIZE * GRID_SIZE;

			Random<Resource> rng(resource, 128);
			rng.init(161803, 0);

			DeviceBuffer<float> noise_values(TOTAL_POINTS);
			DeviceBuffer<float> smoothed_noise(TOTAL_POINTS);
			DeviceBuffer<float> gradient_magnitude(TOTAL_POINTS);

			// Generate base noise
			{
				PROFILE_RANGE("Random::GenerateNoise", backend_type);
				Event noise_event = rng.generate_gaussian(noise_values, 0.0f, 1.0f);
				noise_event.wait();
			}

			// Apply smoothing filter
			{
				PROFILE_RANGE("Kernel::SmoothingFilter", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(std::ref(noise_values));
				auto outputs = std::make_tuple(std::ref(smoothed_noise));

				Event smooth_event = launch_kernel(resource,
												   TOTAL_POINTS,
												   config,
												   inputs,
												   outputs,
												   SmoothingFilterKernel{GRID_SIZE});

				smooth_event.wait();
			}

			// Calculate gradient magnitude
			{
				PROFILE_RANGE("Kernel::GradientCalculation", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(std::ref(smoothed_noise));
				auto outputs = std::make_tuple(std::ref(gradient_magnitude));

				Event gradient_event = launch_kernel(resource,
													 TOTAL_POINTS,
													 config,
													 inputs,
													 outputs,
													 GradientCalculationKernel{GRID_SIZE});

				gradient_event.wait();
			}

			// Validate results
			{
				PROFILE_RANGE("NoiseGeneration::Validation", backend_type);
				std::vector<float> original_noise(TOTAL_POINTS);
				std::vector<float> smoothed(TOTAL_POINTS);
				std::vector<float> gradients(TOTAL_POINTS);

				noise_values.copy_to_host(original_noise);
				smoothed_noise.copy_to_host(smoothed);
				gradient_magnitude.copy_to_host(gradients);

				// Basic validation
				double original_variance = 0.0, smoothed_variance = 0.0;
				for (size_t i = 0; i < TOTAL_POINTS; ++i) {
					original_variance += original_noise[i] * original_noise[i];
					smoothed_variance += smoothed[i] * smoothed[i];
				}
				original_variance /= TOTAL_POINTS;
				smoothed_variance /= TOTAL_POINTS;

				// Smoothing should reduce variance
				REQUIRE(smoothed_variance <= original_variance);

				// All gradients should be non-negative
				bool all_gradients_valid =
					std::all_of(gradients.begin(), gradients.end(), [](float g) {
						return g >= 0.0f && std::isfinite(g);
					});
				REQUIRE(all_gradients_valid);

				double max_gradient = *std::max_element(gradients.begin(), gradients.end());

				LOGINFO("{} Noise Pipeline: {}x{} grid, original variance: {:.3f}, smoothed "
						"variance: {:.3f}, max gradient: {:.3f}",
						backend_name,
						GRID_SIZE,
						GRID_SIZE,
						original_variance,
						smoothed_variance,
						max_gradient);
			}

			PROFILE_MARK("Noise generation pipeline completed", backend_type);
		}
	};

	// Only test available backends - single backend usage as per project rules
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
// Comparative Performance Tests
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture,
				 "Random Generation Comparative Performance",
				 "[random][profiling][comparison]") {

	std::vector<std::pair<Resource, std::string>> available_backends;

	// Only use one backend at a time as per project rules
#ifdef USE_CUDA
	if (cuda_available)
		available_backends.emplace_back(cuda_resource, "CUDA");
#endif
#ifdef USE_SYCL
	if (sycl_available)
		available_backends.emplace_back(sycl_resource, "SYCL");
#endif
#ifdef USE_METAL
	if (metal_available)
		available_backends.emplace_back(metal_resource, "Metal");
#endif

	if (available_backends.empty()) {
		SKIP("No backends available for comparative testing");
	}

	SECTION("Debug resource type and backend compilation") {
// Print the actual resource type value
// Check which backends are compiled in
#ifdef USE_SYCL
		LOGDEBUG("USE_SYCL is defined");
#else
		LOGDEBUG("USE_SYCL is NOT defined");
#endif

#ifdef USE_CUDA
		LOGDEBUG("USE_CUDA is defined");
#else
		LOGDEBUG("USE_CUDA is NOT defined");
#endif

#ifdef USE_METAL
		LOGDEBUG("USE_METAL is defined");
#else
		LOGDEBUG("USE_METAL is NOT defined");
#endif

		// Check ResourceType enum values
		LOGDEBUG("ResourceType::SYCL = {}", static_cast<int>(ResourceType::SYCL));
		LOGDEBUG("ResourceType::CUDA = {}", static_cast<int>(ResourceType::CUDA));
		LOGDEBUG("ResourceType::CPU = {}", static_cast<int>(ResourceType::CPU));
		LOGDEBUG("ResourceType::METAL = {}", static_cast<int>(ResourceType::METAL));

		// Try to create the resource explicitly and test
		try {
			Resource sycl_resource(ResourceType::SYCL, 0);
			LOGDEBUG("Created SYCL resource successfully: type={}, id={}",
					 static_cast<int>(sycl_resource.type),
					 sycl_resource.id);

			// Try a simple kernel launch to see exactly where it fails
			const size_t test_size = 100;
			DeviceBuffer<float> test_buffer(test_size);
			auto inputs = std::make_tuple(std::ref(test_buffer));
			auto outputs = std::make_tuple(std::ref(test_buffer));

			KernelConfig config;

			Event event =
				launch_kernel(sycl_resource, test_size, config, inputs, outputs, SimpleKernel{});

			LOGINFO("About to call launch_kernel with SYCL resource...");

			// Use the struct functor instead of lambda

		} catch (const std::exception& e) {
			LOGINFO("Exception caught: {}", e.what());

			// Let's also check what the ResourceType comparison looks like
			Resource test_resource(ResourceType::SYCL, 0);
			switch (test_resource.type) {
			case ResourceType::SYCL:
				LOGINFO("Switch statement correctly identifies SYCL");
				break;
			case ResourceType::CUDA:
				LOGINFO("Switch statement thinks this is CUDA");
				break;
			case ResourceType::CPU:
				LOGINFO("Switch statement thinks this is CPU");
				break;
			case ResourceType::METAL:
				LOGINFO("Switch statement thinks this is METAL");
				break;
			default:
				LOGINFO("Switch statement hit default case - this is the problem!");
				LOGINFO("Actual resource.type value: {}", static_cast<int>(test_resource.type));
				break;
			}
		}
	}
	SECTION("Gaussian generation performance comparison") {
		constexpr size_t PERF_SIZE = 5000000; // 5M elements (Gaussian is typically slower)

		std::vector<std::pair<std::string, double>> performance_results;

		for (const auto& [resource, backend_name] : available_backends) {
			PROFILE_RANGE("Comparative::Gaussian::" + backend_name, resource.type);

			Random<Resource> rng(resource, 256);
			rng.init(654321, 0);

			DeviceBuffer<float> device_buffer(PERF_SIZE);

			auto start = std::chrono::high_resolution_clock::now();

			Event generation_event = rng.generate_gaussian(device_buffer, 0.0f, 1.0f);
			generation_event.wait();

			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			double time_ms = static_cast<double>(duration.count()) / 1000.0;

			performance_results.emplace_back(backend_name, time_ms);

			LOGINFO("{} generated {} Gaussian numbers in {:.3f} ms ({:.1f} M numbers/sec)",
					backend_name,
					PERF_SIZE,
					time_ms,
					(PERF_SIZE / 1000000.0) / (time_ms / 1000.0));
		}

		// Find fastest backend
		auto fastest =
			std::min_element(performance_results.begin(),
							 performance_results.end(),
							 [](const auto& a, const auto& b) { return a.second < b.second; });

		LOGINFO("Fastest backend for Gaussian generation: {} ({:.3f} ms)",
				fastest->first,
				fastest->second);
	}

	SECTION("Memory bandwidth test") {
		constexpr size_t BANDWIDTH_SIZE = 100000000; // 100M floats = ~400MB

		for (const auto& [resource, backend_name] : available_backends) {
			PROFILE_RANGE("Bandwidth::" + backend_name, resource.type);

			DeviceBuffer<float> device_buffer(BANDWIDTH_SIZE);
			std::vector<float> host_buffer(BANDWIDTH_SIZE, 1.0f);

			// Measure host-to-device bandwidth
			auto start = std::chrono::high_resolution_clock::now();
			device_buffer.copy_from_host(host_buffer);
			auto end = std::chrono::high_resolution_clock::now();

			auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			double h2d_time_ms = static_cast<double>(h2d_duration.count()) / 1000.0;
			double h2d_bandwidth_gbps =
				(BANDWIDTH_SIZE * sizeof(float)) / (h2d_time_ms * 1000000.0);

			// Measure device-to-host bandwidth
			start = std::chrono::high_resolution_clock::now();
			device_buffer.copy_to_host(host_buffer);
			end = std::chrono::high_resolution_clock::now();

			auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			double d2h_time_ms = static_cast<double>(d2h_duration.count()) / 1000.0;
			double d2h_bandwidth_gbps =
				(BANDWIDTH_SIZE * sizeof(float)) / (d2h_time_ms * 1000000.0);

			LOGINFO("{} Memory Bandwidth - H2D: {:.2f} GB/s, D2H: {:.2f} GB/s",
					backend_name,
					h2d_bandwidth_gbps,
					d2h_bandwidth_gbps);
		}
	}
}
TEST_CASE("SYCL Multi-device parallel random generation with cross-device analysis",
		  "[random][profiling][multi-device][sycl]") {
#ifdef USE_SYCL
	// Manually manage SYCL lifetime for this specific multi-device test
	try {
		SYCL::SYCLManager::init();
		SYCL::SYCLManager::load_info();
	} catch (const std::exception& e) {
		SKIP("SYCL backend initialization failed: " << e.what());
	}

	constexpr size_t NUMBERS_PER_DEVICE = 10000000; // 10M per device
	constexpr size_t SAMPLE_SIZE = 10000;			// Sample for analysis

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

		// Statistical data
		float mean = 0.0f;
		float variance = 0.0f;
		float min_val = 0.0f;
		float max_val = 0.0f;

		DeviceData(size_t device_id, size_t buffer_size)
			: resource(ResourceType::SYCL, device_id), sample(SAMPLE_SIZE) {
			// Set this device as current before allocating any memory
			SYCL::SYCLManager::use(static_cast<int>(device_id));

			// Now allocate the buffer on the correct device
			buffer = DeviceBuffer<float>(buffer_size);

			// Initialize the RNG for this device
			rng = std::make_unique<Random<Resource>>(resource, 256);
			rng->init(123456 + device_id * 1000, device_id);
		}

		// Destructor that ensures cleanup happens on the correct device
		~DeviceData() {
			try {
				// Synchronize any pending operations on this device first
				if (generation_event.is_valid()) {
					generation_event.wait();
				}

				// Set this device as current before deallocating
				SYCL::SYCLManager::use(static_cast<int>(resource.id));

				// Reset RNG first to free any device resources
				rng.reset();

				// Buffer will be automatically cleaned up by RAII
			} catch (const std::exception& e) {
				// Log but don't throw from destructor
				LOGWARN("Warning: Failed to cleanup device {} properly: {}", resource.id, e.what());
			}
		}
	};

	std::vector<std::unique_ptr<DeviceData>> devices;
	devices.reserve(num_devices); // Reserve to avoid reallocations

	for (size_t i = 0; i < num_devices; ++i) {
		LOGINFO("Initializing device {} with {} elements...", i, NUMBERS_PER_DEVICE);
		devices.push_back(std::make_unique<DeviceData>(i, NUMBERS_PER_DEVICE));
		LOGINFO("Successfully initialized device {} with seed {}", i, 323456 + i * 1000);
	}

	// Launch generation on all devices simultaneously
	PROFILE_RANGE("MultiDevice::ParallelGeneration", ResourceType::SYCL);

	auto global_start = std::chrono::high_resolution_clock::now();

	// Method 1: True async launches (recommended)
	std::vector<std::future<void>> launch_futures;
	launch_futures.reserve(num_devices);

	for (auto& device : devices) {
		// Launch each device's work in a separate thread
		auto future = std::async(std::launch::async, [&device, &global_start]() {
			// Set device context in this thread
			SYCL::SYCLManager::use(static_cast<int>(device->resource.id));

			// Record start time
			device->start_time = std::chrono::high_resolution_clock::now();

			// Launch the kernel
			device->generation_event = device->rng->generate_uniform(device->buffer, 0.0f, 1.0f);
		});

		launch_futures.push_back(std::move(future));
	}

	// Wait for all launches to complete
	LOGINFO("Waiting for all {} async launches to complete...", num_devices);
	for (auto& future : launch_futures) {
		future.wait();
	}

	LOGINFO("Launched generation on all {} devices simultaneously", num_devices);

	// Wait for all devices to complete and measure individual timings
	for (auto& device : devices) {
		device->generation_event.wait();
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end_time - device->start_time);
		device->timing_ms = static_cast<double>(duration.count()) / 1000.0;
	}

	auto global_end = std::chrono::high_resolution_clock::now();
	double total_time_ms =
		std::chrono::duration<double, std::milli>(global_end - global_start).count();

	// === PERFORMANCE ANALYSIS ===
	std::vector<double> timings;
	for (const auto& device : devices) {
		timings.push_back(device->timing_ms);
	}

	std::vector<double> individual_throughputs;

	for (const auto& device : devices) {
		timings.push_back(device->timing_ms);
		double device_throughput = (NUMBERS_PER_DEVICE / 1000000.0) / (device->timing_ms / 1000.0);
		individual_throughputs.push_back(device_throughput);
	}

	double min_time = *std::min_element(timings.begin(), timings.end());
	double max_time = *std::max_element(timings.begin(), timings.end());
	double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / num_devices;

	size_t total_numbers = NUMBERS_PER_DEVICE * num_devices;
	double actual_aggregate_throughput = (total_numbers / 1000000.0) / (total_time_ms / 1000.0);

	// 2. Sum of individual device throughputs (if they ran independently)
	double sum_individual_throughputs =
		std::accumulate(individual_throughputs.begin(), individual_throughputs.end(), 0.0);

	// 3. Theoretical perfect parallel throughput (if all devices matched the fastest)
	double perfect_parallel_throughput =
		(NUMBERS_PER_DEVICE / 1000000.0) / (min_time / 1000.0) * num_devices;

	// === MULTIPLE EFFICIENCY METRICS ===

	// Efficiency 1: How well we utilized the combined theoretical capacity of our devices
	double capacity_efficiency = (actual_aggregate_throughput / sum_individual_throughputs) * 100.0;

	// Efficiency 2: How close we got to perfect parallel execution (all devices = fastest)
	double parallel_efficiency =
		(actual_aggregate_throughput / perfect_parallel_throughput) * 100.0;

	// Efficiency 3: How much speedup we got vs single device (based on average performance)
	double speedup_factor =
		actual_aggregate_throughput / (individual_throughputs[0]); // vs device 0
	double ideal_speedup = num_devices;
	double speedup_efficiency = (speedup_factor / ideal_speedup) * 100.0;

	// === TIMING ANALYSIS ===
	double timing_variance = 0.0;
	for (double timing : timings) {
		double diff = timing - avg_time;
		timing_variance += diff * diff;
	}
	timing_variance /= num_devices;
	double timing_stddev = std::sqrt(timing_variance);
	double timing_coefficient_of_variation = (timing_stddev / avg_time) * 100.0;
	LOGINFO("=== CORRECTED PERFORMANCE RESULTS ===");
	LOGINFO("Total numbers generated: {} ({} per device)", total_numbers, NUMBERS_PER_DEVICE);
	LOGINFO("Wall-clock time: {:.3f} ms", total_time_ms);
	LOGINFO("Device timings - min: {:.3f}ms, max: {:.3f}ms, avg: {:.3f}ms, stddev: {:.3f}ms",
			min_time,
			max_time,
			avg_time,
			timing_stddev);
	LOGINFO("Timing variation coefficient: {:.1f}% (lower is better)",
			timing_coefficient_of_variation);

	LOGINFO("=== THROUGHPUT ANALYSIS ===");
	LOGINFO("Actual aggregate throughput: {:.1f} M numbers/sec", actual_aggregate_throughput);
	LOGINFO("Sum of individual throughputs: {:.1f} M numbers/sec", sum_individual_throughputs);
	LOGINFO("Perfect parallel throughput: {:.1f} M numbers/sec", perfect_parallel_throughput);

	LOGINFO("=== EFFICIENCY METRICS ===");
	LOGINFO("Capacity efficiency: {:.1f}% (actual vs sum of individual capacities)",
			capacity_efficiency);
	LOGINFO("Parallel efficiency: {:.1f}% (actual vs perfect parallel)", parallel_efficiency);
	LOGINFO("Speedup: {:.1f}x vs single device ({:.1f}% of ideal {}x)",
			speedup_factor,
			speedup_efficiency,
			ideal_speedup);

	// === BOTTLENECK ANALYSIS ===
	LOGINFO("=== BOTTLENECK ANALYSIS ===");
	if (timing_coefficient_of_variation > 20.0) {
		LOGINFO("⚠️  High timing variation ({:.1f}%) suggests load imbalance or resource contention",
				timing_coefficient_of_variation);
	}

	if (capacity_efficiency < 80.0) {
		LOGINFO("⚠️  Low capacity efficiency ({:.1f}%) suggests launch overhead or synchronization "
				"issues",
				capacity_efficiency);
	}

	// Per-device breakdown
	for (size_t i = 0; i < devices.size(); ++i) {
		double device_throughput =
			(NUMBERS_PER_DEVICE / 1000000.0) / (devices[i]->timing_ms / 1000.0);
		LOGINFO("Device {}: {:.3f}ms ({:.1f} M numbers/sec)",
				i,
				devices[i]->timing_ms,
				device_throughput);
	}

	// === CROSS-DEVICE STATISTICAL ANALYSIS ===
	LOGINFO("=== STATISTICAL ANALYSIS ===");
	LOGINFO("Sampling {} values from each device for analysis...", SAMPLE_SIZE);

	// Extract samples from each device and calculate statistics
	for (auto& device : devices) {
		// Ensure we're using the correct device for memory operations
		SYCL::SYCLManager::use(static_cast<int>(device->resource.id));

		device->buffer.copy_to_host(device->sample.data(), SAMPLE_SIZE);

		// Calculate statistics
		auto [min_it, max_it] = std::minmax_element(device->sample.begin(), device->sample.end());
		device->min_val = *min_it;
		device->max_val = *max_it;

		double sum = std::accumulate(device->sample.begin(), device->sample.end(), 0.0);
		device->mean = static_cast<float>(sum / SAMPLE_SIZE);

		double variance_sum = 0.0;
		for (float val : device->sample) {
			double diff = val - device->mean;
			variance_sum += diff * diff;
		}
		device->variance = static_cast<float>(variance_sum / SAMPLE_SIZE);
	}

	// Cross-device comparisons
	std::vector<float> all_mins, all_maxes, all_means, all_variances;
	for (const auto& device : devices) {
		all_mins.push_back(device->min_val);
		all_maxes.push_back(device->max_val);
		all_means.push_back(device->mean);
		all_variances.push_back(device->variance);
	}

	float global_min = *std::min_element(all_mins.begin(), all_mins.end());
	float global_max = *std::max_element(all_maxes.begin(), all_maxes.end());
	float mean_of_means = std::accumulate(all_means.begin(), all_means.end(), 0.0f) / num_devices;
	float mean_of_variances =
		std::accumulate(all_variances.begin(), all_variances.end(), 0.0f) / num_devices;

	LOGINFO("Global range across all devices: [{:.6f}, {:.6f}]", global_min, global_max);
	LOGINFO("Mean of device means: {:.6f} (target: ~0.500)", mean_of_means);
	LOGINFO("Mean of device variances: {:.6f} (target: ~0.083 for uniform[0,1])",
			mean_of_variances);

	// Device-by-device breakdown
	for (size_t i = 0; i < devices.size(); ++i) {
		LOGINFO("Device {}: mean={:.6f}, var={:.6f}, range=[{:.6f}, {:.6f}]",
				i,
				devices[i]->mean,
				devices[i]->variance,
				devices[i]->min_val,
				devices[i]->max_val);
	}

	// === CROSS-DEVICE QUALITY METRICS ===
	float max_mean_deviation = 0.0f;
	for (float mean : all_means) {
		max_mean_deviation = std::max(max_mean_deviation, std::abs(mean - 0.5f));
	}

	float min_variance = *std::min_element(all_variances.begin(), all_variances.end());
	float max_variance = *std::max_element(all_variances.begin(), all_variances.end());
	float variance_ratio = (min_variance > 0.0f) ? (max_variance / min_variance) : 1.0f;

	LOGINFO("=== QUALITY METRICS ===");
	LOGINFO("Max mean deviation from 0.5: {:.6f} (should be < 0.01)", max_mean_deviation);
	LOGINFO("Variance ratio (max/min): {:.3f} (should be < 1.5)", variance_ratio);

	// === CROSS-DEVICE CORRELATION TEST ===
	double max_correlation = 0.0;
	if (num_devices >= 2) {
		PROFILE_RANGE("CrossDevice::CorrelationTest", ResourceType::SYCL);

		LOGINFO("=== CROSS-DEVICE CORRELATION ANALYSIS ===");

		auto calculate_correlation = [](const std::vector<float>& x,
										const std::vector<float>& y) -> double {
			if (x.size() != y.size() || x.empty())
				return 0.0;

			double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
			double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
			double mean_x = sum_x / x.size();
			double mean_y = sum_y / y.size();

			double numerator = 0.0, sum_sq_x = 0.0, sum_sq_y = 0.0;

			for (size_t i = 0; i < x.size(); ++i) {
				double dx = x[i] - mean_x;
				double dy = y[i] - mean_y;
				numerator += dx * dy;
				sum_sq_x += dx * dx;
				sum_sq_y += dy * dy;
			}

			double denominator = std::sqrt(sum_sq_x * sum_sq_y);
			return (denominator > 1e-10) ? (numerator / denominator) : 0.0;
		};

		std::vector<double> correlations;

		for (size_t i = 0; i < num_devices - 1; ++i) {
			for (size_t j = i + 1; j < num_devices; ++j) {
				double corr = calculate_correlation(devices[i]->sample, devices[j]->sample);
				double abs_corr = std::abs(corr);
				correlations.push_back(abs_corr);
				max_correlation = std::max(max_correlation, abs_corr);

				LOGINFO("Correlation between device {} and {}: {:.6f}", i, j, corr);
			}
		}

		LOGINFO("Maximum absolute correlation: {:.6f} (should be < 0.05)", max_correlation);
	}

	// === FINAL QUALITY ASSERTIONS ===
	REQUIRE(global_min >= 0.0f);
	REQUIRE(global_max <= 1.0f);
	REQUIRE(max_mean_deviation < 0.01f);
	REQUIRE(variance_ratio < 1.5f);
	REQUIRE(mean_of_means > 0.45f);
	REQUIRE(mean_of_means < 0.55f);
	// REQUIRE(parallel_efficiency > 80.0); // Should be at least 80% efficient
	if (num_devices >= 2) {
		REQUIRE(max_correlation < 0.05); // Devices should be uncorrelated
	}

	// === SUMMARY ===
	LOGINFO("=== MULTI-DEVICE TEST SUMMARY ===");
	LOGINFO("✓ {} SYCL device(s) generated {:.0f}M numbers in {:.3f}ms",
			num_devices,
			total_numbers / 1000000.0,
			total_time_ms);
	LOGINFO("✓ Aggregate throughput: {:.1f} M numbers/sec", actual_aggregate_throughput);
	LOGINFO("✓ Parallel efficiency: {:.1f}%", parallel_efficiency);
	LOGINFO("✓ All statistical quality metrics passed");
	if (num_devices >= 2) {
		LOGINFO("✓ Cross-device correlation < {:.3f}", max_correlation);
	}

	PROFILE_MARK("MultiDevice test completed successfully", ResourceType::SYCL);

	// Ensure all devices are synchronized before cleanup
	LOGINFO("Synchronizing all devices before cleanup...");
	for (size_t i = 0; i < devices.size(); ++i) {
		try {
			SYCL::SYCLManager::sync(static_cast<int>(i));
		} catch (const std::exception& e) {
			LOGWARN("Warning: Failed to sync device {}: {}", i, e.what());
		}
	}

	// Clear devices explicitly in reverse order to avoid dependency issues
	LOGINFO("Cleaning up device data...");
	devices.clear();

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

TEST_CASE("CUDA Multi-device parallel random generation with cross-device analysis",
		  "[random][profiling][multi-device][cuda]") {
#ifdef USE_CUDA
	// Manually manage SYCL lifetime for this specific multi-device test
	try {
		CUDA::CUDAManager::init();
		CUDA::CUDAManager::load_info();
	} catch (const std::exception& e) {
		SKIP("SYCL backend initialization failed: " << e.what());
	}

	constexpr size_t NUMBERS_PER_DEVICE = 10000000; // 10M per device
	constexpr size_t SAMPLE_SIZE = 10000;			// Sample for analysis

	const auto& cuda_devices = CUDA::CUDAManager::devices();
	const size_t num_devices = cuda_devices.size();

	if (num_devices == 0) {
		SKIP("No CUDA devices available for multi-device test.");
	}

	LOGINFO("=== Multi-Device Parallel Random Generation Test ===");
	LOGINFO("Testing {} CUDA devices with {} numbers per device", num_devices, NUMBERS_PER_DEVICE);

	// This struct encapsulates all per-device data, preventing dangling references.
	struct DeviceData {
		Resource resource;
		std::unique_ptr<Random<Resource>> rng;
		DeviceBuffer<float> buffer;
		Event generation_event;
		std::chrono::high_resolution_clock::time_point start_time;
		double timing_ms = 0.0;
		std::vector<float> sample;

		// Statistical data
		float mean = 0.0f;
		float variance = 0.0f;
		float min_val = 0.0f;
		float max_val = 0.0f;

		DeviceData(size_t device_id, size_t buffer_size)
			: resource(ResourceType::CUDA, device_id), sample(SAMPLE_SIZE) {
			// Set this device as current before allocating any memory
			CUDA::CUDAManager::use(static_cast<int>(device_id));

			// Now allocate the buffer on the correct device
			buffer = DeviceBuffer<float>(buffer_size);

			// Initialize the RNG for this device
			rng = std::make_unique<Random<Resource>>(resource, 256);
			rng->init(123456 + device_id * 1000, device_id);
		}

		// Destructor that ensures cleanup happens on the correct device
		~DeviceData() {
			try {
				// Synchronize any pending operations on this device first
				if (generation_event.is_valid()) {
					generation_event.wait();
				}

				// Set this device as current before deallocating
				CUDA::CUDAManager::use(static_cast<int>(resource.id));

				// Reset RNG first to free any device resources
				rng.reset();

				// Buffer will be automatically cleaned up by RAII
			} catch (const std::exception& e) {
				// Log but don't throw from destructor
				LOGWARN("Warning: Failed to cleanup device {} properly: {}", resource.id, e.what());
			}
		}
	};

	std::vector<std::unique_ptr<DeviceData>> devices;
	devices.reserve(num_devices); // Reserve to avoid reallocations

	for (size_t i = 0; i < num_devices; ++i) {
		LOGINFO("Initializing device {} with {} elements...", i, NUMBERS_PER_DEVICE);
		devices.push_back(std::make_unique<DeviceData>(i, NUMBERS_PER_DEVICE));
		LOGINFO("Successfully initialized device {} with seed {}", i, 323456 + i * 1000);
	}

	// Launch generation on all devices simultaneously
	PROFILE_RANGE("MultiDevice::ParallelGeneration", ResourceType::CUDA);

	auto global_start = std::chrono::high_resolution_clock::now();

	// Method 1: True async launches (recommended)
	std::vector<std::future<void>> launch_futures;
	launch_futures.reserve(num_devices);

	for (auto& device : devices) {
		// Launch each device's work in a separate thread
		auto future = std::async(std::launch::async, [&device, &global_start]() {
			// Set device context in this thread
			CUDA::CUDAManager::use(static_cast<int>(device->resource.id));

			// Record start time
			device->start_time = std::chrono::high_resolution_clock::now();

			// Launch the kernel
			device->generation_event = device->rng->generate_uniform(device->buffer, 0.0f, 1.0f);
		});

		launch_futures.push_back(std::move(future));
	}

	// Wait for all launches to complete
	LOGINFO("Waiting for all {} async launches to complete...", num_devices);
	for (auto& future : launch_futures) {
		future.wait();
	}

	LOGINFO("Launched generation on all {} devices simultaneously", num_devices);

	// Wait for all devices to complete and measure individual timings
	for (auto& device : devices) {
		device->generation_event.wait();
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end_time - device->start_time);
		device->timing_ms = static_cast<double>(duration.count()) / 1000.0;
	}

	auto global_end = std::chrono::high_resolution_clock::now();
	double total_time_ms =
		std::chrono::duration<double, std::milli>(global_end - global_start).count();

	// === PERFORMANCE ANALYSIS ===
	std::vector<double> timings;
	for (const auto& device : devices) {
		timings.push_back(device->timing_ms);
	}

	std::vector<double> individual_throughputs;

	for (const auto& device : devices) {
		timings.push_back(device->timing_ms);
		double device_throughput = (NUMBERS_PER_DEVICE / 1000000.0) / (device->timing_ms / 1000.0);
		individual_throughputs.push_back(device_throughput);
	}

	double min_time = *std::min_element(timings.begin(), timings.end());
	double max_time = *std::max_element(timings.begin(), timings.end());
	double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / num_devices;

	size_t total_numbers = NUMBERS_PER_DEVICE * num_devices;
	double actual_aggregate_throughput = (total_numbers / 1000000.0) / (total_time_ms / 1000.0);

	// 2. Sum of individual device throughputs (if they ran independently)
	double sum_individual_throughputs =
		std::accumulate(individual_throughputs.begin(), individual_throughputs.end(), 0.0);

	// 3. Theoretical perfect parallel throughput (if all devices matched the fastest)
	double perfect_parallel_throughput =
		(NUMBERS_PER_DEVICE / 1000000.0) / (min_time / 1000.0) * num_devices;

	// === MULTIPLE EFFICIENCY METRICS ===

	// Efficiency 1: How well we utilized the combined theoretical capacity of our devices
	double capacity_efficiency = (actual_aggregate_throughput / sum_individual_throughputs) * 100.0;

	// Efficiency 2: How close we got to perfect parallel execution (all devices = fastest)
	double parallel_efficiency =
		(actual_aggregate_throughput / perfect_parallel_throughput) * 100.0;

	// Efficiency 3: How much speedup we got vs single device (based on average performance)
	double speedup_factor =
		actual_aggregate_throughput / (individual_throughputs[0]); // vs device 0
	double ideal_speedup = num_devices;
	double speedup_efficiency = (speedup_factor / ideal_speedup) * 100.0;

	// === TIMING ANALYSIS ===
	double timing_variance = 0.0;
	for (double timing : timings) {
		double diff = timing - avg_time;
		timing_variance += diff * diff;
	}
	timing_variance /= num_devices;
	double timing_stddev = std::sqrt(timing_variance);
	double timing_coefficient_of_variation = (timing_stddev / avg_time) * 100.0;
	LOGINFO("=== CORRECTED PERFORMANCE RESULTS ===");
	LOGINFO("Total numbers generated: {} ({} per device)", total_numbers, NUMBERS_PER_DEVICE);
	LOGINFO("Wall-clock time: {:.3f} ms", total_time_ms);
	LOGINFO("Device timings - min: {:.3f}ms, max: {:.3f}ms, avg: {:.3f}ms, stddev: {:.3f}ms",
			min_time,
			max_time,
			avg_time,
			timing_stddev);
	LOGINFO("Timing variation coefficient: {:.1f}% (lower is better)",
			timing_coefficient_of_variation);

	LOGINFO("=== THROUGHPUT ANALYSIS ===");
	LOGINFO("Actual aggregate throughput: {:.1f} M numbers/sec", actual_aggregate_throughput);
	LOGINFO("Sum of individual throughputs: {:.1f} M numbers/sec", sum_individual_throughputs);
	LOGINFO("Perfect parallel throughput: {:.1f} M numbers/sec", perfect_parallel_throughput);

	LOGINFO("=== EFFICIENCY METRICS ===");
	LOGINFO("Capacity efficiency: {:.1f}% (actual vs sum of individual capacities)",
			capacity_efficiency);
	LOGINFO("Parallel efficiency: {:.1f}% (actual vs perfect parallel)", parallel_efficiency);
	LOGINFO("Speedup: {:.1f}x vs single device ({:.1f}% of ideal {}x)",
			speedup_factor,
			speedup_efficiency,
			ideal_speedup);

	// === BOTTLENECK ANALYSIS ===
	LOGINFO("=== BOTTLENECK ANALYSIS ===");
	if (timing_coefficient_of_variation > 20.0) {
		LOGINFO("⚠️  High timing variation ({:.1f}%) suggests load imbalance or resource contention",
				timing_coefficient_of_variation);
	}

	if (capacity_efficiency < 80.0) {
		LOGINFO("⚠️  Low capacity efficiency ({:.1f}%) suggests launch overhead or synchronization "
				"issues",
				capacity_efficiency);
	}

	// Per-device breakdown
	for (size_t i = 0; i < devices.size(); ++i) {
		double device_throughput =
			(NUMBERS_PER_DEVICE / 1000000.0) / (devices[i]->timing_ms / 1000.0);
		LOGINFO("Device {}: {:.3f}ms ({:.1f} M numbers/sec)",
				i,
				devices[i]->timing_ms,
				device_throughput);
	}

	// === CROSS-DEVICE STATISTICAL ANALYSIS ===
	LOGINFO("=== STATISTICAL ANALYSIS ===");
	LOGINFO("Sampling {} values from each device for analysis...", SAMPLE_SIZE);

	// Extract samples from each device and calculate statistics
	for (auto& device : devices) {
		// Ensure we're using the correct device for memory operations
		CUDA::CUDAManager::use(static_cast<int>(device->resource.id));

		device->buffer.copy_to_host(device->sample.data(), SAMPLE_SIZE);

		// Calculate statistics
		auto [min_it, max_it] = std::minmax_element(device->sample.begin(), device->sample.end());
		device->min_val = *min_it;
		device->max_val = *max_it;

		double sum = std::accumulate(device->sample.begin(), device->sample.end(), 0.0);
		device->mean = static_cast<float>(sum / SAMPLE_SIZE);

		double variance_sum = 0.0;
		for (float val : device->sample) {
			double diff = val - device->mean;
			variance_sum += diff * diff;
		}
		device->variance = static_cast<float>(variance_sum / SAMPLE_SIZE);
	}

	// Cross-device comparisons
	std::vector<float> all_mins, all_maxes, all_means, all_variances;
	for (const auto& device : devices) {
		all_mins.push_back(device->min_val);
		all_maxes.push_back(device->max_val);
		all_means.push_back(device->mean);
		all_variances.push_back(device->variance);
	}

	float global_min = *std::min_element(all_mins.begin(), all_mins.end());
	float global_max = *std::max_element(all_maxes.begin(), all_maxes.end());
	float mean_of_means = std::accumulate(all_means.begin(), all_means.end(), 0.0f) / num_devices;
	float mean_of_variances =
		std::accumulate(all_variances.begin(), all_variances.end(), 0.0f) / num_devices;

	LOGINFO("Global range across all devices: [{:.6f}, {:.6f}]", global_min, global_max);
	LOGINFO("Mean of device means: {:.6f} (target: ~0.500)", mean_of_means);
	LOGINFO("Mean of device variances: {:.6f} (target: ~0.083 for uniform[0,1])",
			mean_of_variances);

	// Device-by-device breakdown
	for (size_t i = 0; i < devices.size(); ++i) {
		LOGINFO("Device {}: mean={:.6f}, var={:.6f}, range=[{:.6f}, {:.6f}]",
				i,
				devices[i]->mean,
				devices[i]->variance,
				devices[i]->min_val,
				devices[i]->max_val);
	}

	// === CROSS-DEVICE QUALITY METRICS ===
	float max_mean_deviation = 0.0f;
	for (float mean : all_means) {
		max_mean_deviation = std::max(max_mean_deviation, std::abs(mean - 0.5f));
	}

	float min_variance = *std::min_element(all_variances.begin(), all_variances.end());
	float max_variance = *std::max_element(all_variances.begin(), all_variances.end());
	float variance_ratio = (min_variance > 0.0f) ? (max_variance / min_variance) : 1.0f;

	LOGINFO("=== QUALITY METRICS ===");
	LOGINFO("Max mean deviation from 0.5: {:.6f} (should be < 0.01)", max_mean_deviation);
	LOGINFO("Variance ratio (max/min): {:.3f} (should be < 1.5)", variance_ratio);

	// === CROSS-DEVICE CORRELATION TEST ===
	double max_correlation = 0.0;
	if (num_devices >= 2) {
		PROFILE_RANGE("CrossDevice::CorrelationTest", ResourceType::CUDA);

		LOGINFO("=== CROSS-DEVICE CORRELATION ANALYSIS ===");

		auto calculate_correlation = [](const std::vector<float>& x,
										const std::vector<float>& y) -> double {
			if (x.size() != y.size() || x.empty())
				return 0.0;

			double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
			double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
			double mean_x = sum_x / x.size();
			double mean_y = sum_y / y.size();

			double numerator = 0.0, sum_sq_x = 0.0, sum_sq_y = 0.0;

			for (size_t i = 0; i < x.size(); ++i) {
				double dx = x[i] - mean_x;
				double dy = y[i] - mean_y;
				numerator += dx * dy;
				sum_sq_x += dx * dx;
				sum_sq_y += dy * dy;
			}

			double denominator = std::sqrt(sum_sq_x * sum_sq_y);
			return (denominator > 1e-10) ? (numerator / denominator) : 0.0;
		};

		std::vector<double> correlations;

		for (size_t i = 0; i < num_devices - 1; ++i) {
			for (size_t j = i + 1; j < num_devices; ++j) {
				double corr = calculate_correlation(devices[i]->sample, devices[j]->sample);
				double abs_corr = std::abs(corr);
				correlations.push_back(abs_corr);
				max_correlation = std::max(max_correlation, abs_corr);

				LOGINFO("Correlation between device {} and {}: {:.6f}", i, j, corr);
			}
		}

		LOGINFO("Maximum absolute correlation: {:.6f} (should be < 0.05)", max_correlation);
	}

	// === FINAL QUALITY ASSERTIONS ===
	REQUIRE(global_min >= 0.0f);
	REQUIRE(global_max <= 1.0f);
	REQUIRE(max_mean_deviation < 0.01f);
	REQUIRE(variance_ratio < 1.5f);
	REQUIRE(mean_of_means > 0.45f);
	REQUIRE(mean_of_means < 0.55f);
	// REQUIRE(parallel_efficiency > 80.0); // Should be at least 80% efficient
	if (num_devices >= 2) {
		REQUIRE(max_correlation < 0.05); // Devices should be uncorrelated
	}

	// === SUMMARY ===
	LOGINFO("=== MULTI-DEVICE TEST SUMMARY ===");
	LOGINFO("✓ {} CUDA device(s) generated {:.0f}M numbers in {:.3f}ms",
			num_devices,
			total_numbers / 1000000.0,
			total_time_ms);
	LOGINFO("✓ Aggregate throughput: {:.1f} M numbers/sec", actual_aggregate_throughput);
	LOGINFO("✓ Parallel efficiency: {:.1f}%", parallel_efficiency);
	LOGINFO("✓ All statistical quality metrics passed");
	if (num_devices >= 2) {
		LOGINFO("✓ Cross-device correlation < {:.3f}", max_correlation);
	}

	PROFILE_MARK("MultiDevice test completed successfully", ResourceType::CUDA);

	// Ensure all devices are synchronized before cleanup
	LOGINFO("Synchronizing all devices before cleanup...");
	for (size_t i = 0; i < devices.size(); ++i) {
		try {
			CUDA::CUDAManager::sync(static_cast<int>(i));
		} catch (const std::exception& e) {
			LOGWARN("Warning: Failed to sync device {}: {}", i, e.what());
		}
	}

	// Clear devices explicitly in reverse order to avoid dependency issues
	LOGINFO("Cleaning up device data...");
	devices.clear();

	// Manually finalize SYCL at the end of the test
	try {
		CUDA::CUDAManager::finalize();
	} catch (const std::exception& e) {
		FAIL("SYCL finalization failed: " << e.what());
	}
#else
	SKIP("Multi-device test requires CUDA backend to be enabled.");
#endif
}
