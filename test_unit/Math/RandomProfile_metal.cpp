#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/METAL/METALManager.h"
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

// Helper: Check if Metal is available, else skip
#define REQUIRE_METAL_OR_SKIP() \
    ARBD::METAL::Manager::load_info(); \
    if (!ARBD::METAL::Manager::get_library()) { \
        SKIP("Metal library not loaded - skipping Metal kernel test"); \
    }

// ============================================================================
// Profiled Random Test Fixture - Metal Only
// ============================================================================

struct MetalProfiledRandomTestFixture {
	Resource metal_resource;
	bool metal_available = false;

	MetalProfiledRandomTestFixture() {
		// Initialize profiling
		ProfilingConfig config;
		config.enable_timing = true;
		config.enable_memory_tracking = true;
		config.enable_kernel_profiling = true;
		config.enable_backend_markers = true;
		config.output_file = "metal_random_profile_test.json";
		ProfileManager::init(config);

		// Initialize Metal backend
		try {
#ifdef USE_METAL
			METAL::Manager::init();
			METAL::Manager::load_info();
			if (!METAL::Manager::devices().empty()) {
				METAL::Manager::use(0);
				metal_resource = Resource(ResourceType::METAL, 0);
				metal_available = true;
				LOGINFO("Metal backend available for profiled Random tests");
			}
#endif
		} catch (const std::exception& e) {
			LOGWARN("Metal backend initialization failed in MetalProfiledRandomTestFixture: {}", e.what());
		}
	}

	~MetalProfiledRandomTestFixture() {
		ProfileManager::print_summary();
		ProfileManager::finalize();

		try {
#ifdef USE_METAL
			if (metal_available)
				METAL::Manager::finalize();
#endif
		} catch (const std::exception& e) {
			std::cerr << "Error during MetalProfiledRandomTestFixture cleanup: " << e.what()
					  << std::endl;
		}
	}
};

// ============================================================================
// Profiled Random Generation Tests - Metal Backend
// ============================================================================

TEST_CASE_METHOD(MetalProfiledRandomTestFixture,
				 "Metal Profiled Random Generation Performance",
				 "[random][profiling][performance][metal]") {

	auto test_profiled_generation = [this](const Resource& resource,
										   const std::string& backend_name,
										   ResourceType backend_type) {
		if (!resource.is_device()) {
			SKIP("Backend " + backend_name + " not available");
		}

		SECTION("Profiled uniform generation on " + backend_name) {
			PROFILE_RANGE("MetalRandomGenerator::Creation", backend_type);

			Random<Resource> rng(resource, 128);
			rng.init(42, 0);

			// Profile memory allocation
			{
				PROFILE_RANGE("MetalDeviceBuffer::Allocation", backend_type);
				DeviceBuffer<float> device_buffer(100000); // 100K elements

				PROFILE_MEMORY(backend_type, nullptr);

				// Profile random number generation
				{
					PROFILE_RANGE("MetalRandom::GenerateUniform", backend_type);
					Event generation_event = rng.generate_uniform(device_buffer, 0.0f, 1.0f);
					generation_event.wait();
				}

				// Profile memory copy
				{
					PROFILE_RANGE("MetalBuffer::CopyToHost", backend_type);
					std::vector<float> host_results(100000);
					device_buffer.copy_to_host(host_results);
				}

				PROFILE_MARK("Metal uniform generation completed", backend_type);
			}

			LOGINFO("{} profiled uniform generation completed", backend_name);
		}

		SECTION("Profiled gaussian generation on " + backend_name) {
			PROFILE_RANGE("MetalRandomGenerator::GaussianTest", backend_type);

			Random<Resource> rng(resource, 128);
			rng.init(12345, 0);

			DeviceBuffer<float> device_buffer(50000);

			{
				PROFILE_RANGE("MetalRandom::GenerateGaussian", backend_type);
				Event generation_event = rng.generate_gaussian(device_buffer, 0.0f, 1.0f);
				generation_event.wait();
			}

			{
				PROFILE_RANGE("MetalStatistical::Validation", backend_type);
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

			PROFILE_MARK("Metal gaussian generation validated", backend_type);
			LOGINFO("{} profiled gaussian generation completed", backend_name);
		}

		SECTION("Profiled Vector3 generation on " + backend_name) {
			PROFILE_RANGE("MetalRandomGenerator::Vector3Test", backend_type);

			Random<Resource> rng(resource, 128);
			rng.init(9876, 0);

			DeviceBuffer<ARBD::Vector3_t<float>> device_buffer(25000);
			ARBD::Vector3_t<float> mean(0.0f, 0.0f, 0.0f);
			ARBD::Vector3_t<float> dev(1.0f, 1.0f, 1.0f);

			{
				PROFILE_RANGE("MetalRandom::GenerateVector3", backend_type);
				Event generation_event = rng.generate_gaussian(device_buffer, mean, dev);
				generation_event.wait();
			}

			{
				PROFILE_RANGE("MetalVector3::Validation", backend_type);
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

			PROFILE_MARK("Metal Vector3 generation validated", backend_type);
			LOGINFO("{} profiled Vector3 generation completed", backend_name);
		}
	};

#ifdef USE_METAL
	test_profiled_generation(metal_resource, "Metal", ResourceType::METAL);
#endif
}

// ============================================================================
// Profiled Random + Kernels Integration Tests - Metal Backend
// ============================================================================

TEST_CASE_METHOD(MetalProfiledRandomTestFixture,
				 "Metal Profiled Random-Kernel Integration",
				 "[random][kernels][profiling][integration][metal]") {
	auto test_profiled_integration = [this](const Resource& resource,
											const std::string& backend_name,
											ResourceType backend_type) {
		if (!resource.is_device()) {
			SKIP("Backend " + backend_name + " not available");
		}

		SECTION("Profiled Monte Carlo simulation on " + backend_name) {
			PROFILE_RANGE("MetalMonteCarlo::Simulation", backend_type);

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

			DeviceBuffer<float> x_coords(NUM_SAMPLES);
			DeviceBuffer<float> y_coords(NUM_SAMPLES);
			DeviceBuffer<int> inside_circle(NUM_SAMPLES);

			// Generate random coordinates
			{
				PROFILE_RANGE("MetalRandom::GenerateCoordinates", backend_type);
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
				PROFILE_RANGE("MetalKernel::CircleTest", backend_type);

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
				PROFILE_RANGE("MetalMonteCarlo::PiEstimation", backend_type);
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
				// For Monte Carlo with 100K samples, expect reasonable accuracy
				REQUIRE_THAT(pi_estimate, WithinAbs(3.14159, 0.1));

				LOGINFO("{} Monte Carlo π estimate: {:.5f} (error: {:.5f})",
						backend_name,
						pi_estimate,
						std::abs(pi_estimate - 3.14159));
			}

			PROFILE_MARK("Metal Monte Carlo simulation completed", backend_type);
		}

		SECTION("Profiled random walk simulation on " + backend_name) {
			PROFILE_RANGE("MetalRandomWalk::Simulation", backend_type);

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
				PROFILE_RANGE("MetalRandom::GenerateSteps", backend_type);
				Event steps_event = rng.generate_gaussian(random_steps, mean, dev);
				steps_event.wait();
			}

			// Initialize walker positions to origin
			{
				PROFILE_RANGE("MetalKernel::InitializeWalkers", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple();
				auto outputs = std::forward_as_tuple(walker_positions);

				Event init_event = launch_metal_kernel(
					resource,
					NUM_WALKERS,
					inputs,
					outputs,
					config,
					"initialize_walkers_kernel"
				);

				init_event.wait();
			}

			// Simulate random walk
			{
				PROFILE_RANGE("MetalKernel::RandomWalk", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(random_steps);
				auto outputs = std::forward_as_tuple(walker_positions);

				Event walk_event = launch_metal_kernel(
					resource,
					NUM_WALKERS,
					inputs,
					outputs,
					config,
					"random_walk_kernel"
				);

				walk_event.wait();
			}

			// Calculate final distances from origin
			{
				PROFILE_RANGE("MetalKernel::CalculateDistances", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(walker_positions);
				auto outputs = std::forward_as_tuple(final_distances);

				Event distance_event = launch_metal_kernel(
					resource,
					NUM_WALKERS,
					inputs,
					outputs,
					config,
					"calculate_distances_kernel"
				);

				distance_event.wait();
			}

			// Analyze results
			{
				PROFILE_RANGE("MetalRandomWalk::Analysis", backend_type);
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

			PROFILE_MARK("Metal random walk simulation completed", backend_type);
		}

		SECTION("Profiled noise generation pipeline on " + backend_name) {
			PROFILE_RANGE("MetalNoiseGeneration::Pipeline", backend_type);

			constexpr size_t GRID_SIZE = 256;
			constexpr size_t TOTAL_POINTS = GRID_SIZE * GRID_SIZE;

			Random<Resource> rng(resource, 128);
			rng.init(161803, 0);

			DeviceBuffer<float> noise_values(TOTAL_POINTS);
			DeviceBuffer<float> smoothed_noise(TOTAL_POINTS);
			DeviceBuffer<float> gradient_magnitude(TOTAL_POINTS);

			// Generate base noise
			{
				PROFILE_RANGE("MetalRandom::GenerateNoise", backend_type);
				Event noise_event = rng.generate_gaussian(noise_values, 0.0f, 1.0f);
				noise_event.wait();
			}

			// Apply smoothing filter
			{
				PROFILE_RANGE("MetalKernel::SmoothingFilter", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(noise_values);
				auto outputs = std::forward_as_tuple(smoothed_noise);

				Event smooth_event = launch_metal_kernel(
					resource,
					TOTAL_POINTS,
					inputs,
					outputs,
					config,
					"smoothing_filter_kernel"
				);

				smooth_event.wait();
			}

			// Calculate gradient magnitude
			{
				PROFILE_RANGE("MetalKernel::GradientCalculation", backend_type);
				KernelConfig config{.block_size = 256, .async = false};

				auto inputs = std::make_tuple(smoothed_noise);
				auto outputs = std::forward_as_tuple(gradient_magnitude);

				Event gradient_event = launch_metal_kernel(
					resource,
					TOTAL_POINTS,
					inputs,
					outputs,
					config,
					"gradient_calculation_kernel"
				);

				gradient_event.wait();
			}

			// Validate results
			{
				PROFILE_RANGE("MetalNoiseGeneration::Validation", backend_type);
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

			PROFILE_MARK("Metal noise generation pipeline completed", backend_type);
		}
	};

#ifdef USE_METAL
	test_profiled_integration(metal_resource, "Metal", ResourceType::METAL);
#endif
}

// ============================================================================
// Metal Performance Tests
// ============================================================================

TEST_CASE_METHOD(MetalProfiledRandomTestFixture,
				 "Metal Random Generation Performance Analysis",
				 "[random][profiling][performance][metal]") {

	if (!metal_available) {
		SKIP("Metal backend not available for performance testing");
	}

	SECTION("Gaussian generation performance analysis") {
		constexpr size_t PERF_SIZE = 5000000; // 5M elements

		PROFILE_RANGE("MetalPerformance::GaussianGeneration", ResourceType::METAL);

		Random<Resource> rng(metal_resource, 256);
		rng.init(654321, 0);

		DeviceBuffer<float> device_buffer(PERF_SIZE);

		auto start = std::chrono::high_resolution_clock::now();

		Event generation_event = rng.generate_gaussian(device_buffer, 0.0f, 1.0f);
		generation_event.wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		double time_ms = static_cast<double>(duration.count()) / 1000.0;

		LOGINFO("Metal generated {} Gaussian numbers in {:.3f} ms ({:.1f} M numbers/sec)",
				PERF_SIZE,
				time_ms,
				(PERF_SIZE / 1000000.0) / (time_ms / 1000.0));

		// Performance should be reasonable
		REQUIRE(time_ms < 10000.0); // Less than 10 seconds
	}

	SECTION("Memory bandwidth test") {
		constexpr size_t BANDWIDTH_SIZE = 10000000; // 10M floats = ~40MB

		PROFILE_RANGE("MetalBandwidth::Test", ResourceType::METAL);

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

		LOGINFO("Metal Memory Bandwidth - H2D: {:.2f} GB/s, D2H: {:.2f} GB/s",
				h2d_bandwidth_gbps,
				d2h_bandwidth_gbps);

		// Basic sanity check - should have some measurable bandwidth
		REQUIRE(h2d_bandwidth_gbps > 0.1);
		REQUIRE(d2h_bandwidth_gbps > 0.1);
	}
}