#include "../catch_boiler.h"

#ifdef USE_CUDA

#include "Backend/Buffer.h"
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Resource.h"
#include "Math/Vector3.h"
#include "Random/Random.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

using namespace ARBD;
using namespace ARBD::CUDA;

// Test fixture for CUDA Random tests
class CUDARandomTestFixture {
  public:
	Resource cuda_resource;

	CUDARandomTestFixture() {
		try {
			// Initialize CUDA backend
			CUDAManager::init();
			CUDAManager::load_info();

			if (CUDAManager::all_devices().empty()) {
				SKIP("No CUDA devices available");
				return;
			}

			// Use the first available CUDA device for all tests
			cuda_resource = Resource(ResourceType::CUDA, 0);
			CUDAManager::use(0);

		} catch (const std::exception& e) {
			FAIL("Failed to initialize CUDAManager in test fixture: " << e.what());
		}
	}

	~CUDARandomTestFixture() {
		try {
			CUDAManager::finalize();
		} catch (const std::exception& e) {
			std::cerr << "Error during CUDAManager finalization in test fixture: " << e.what()
					  << std::endl;
		}
	}
};

// Helper function to calculate basic statistics
struct Statistics {
	float mean;
	float variance;
	float min_val;
	float max_val;

	Statistics(const std::vector<float>& data) {
		if (data.empty()) {
			mean = variance = min_val = max_val = 0.0f;
			return;
		}

		// Calculate mean
		float sum = std::accumulate(data.begin(), data.end(), 0.0f);
		mean = sum / data.size();

		// Calculate variance
		float var_sum = 0.0f;
		for (float val : data) {
			float diff = val - mean;
			var_sum += diff * diff;
		}
		variance = var_sum / data.size();

		// Find min and max
		auto minmax = std::minmax_element(data.begin(), data.end());
		min_val = *minmax.first;
		max_val = *minmax.second;
	}
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(CUDARandomTestFixture,
				 "CUDA RandomDevice Basic Initialization",
				 "[cuda][random][init]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("RandomCUDA creation and initialization") {
		constexpr size_t num_states = 128;
		Random<Resource> rng(cuda_resource, num_states);
		// Initialize with a known seed
		rng.init(12345UL);

		REQUIRE(rng.is_initialized());
		REQUIRE(rng.get_resource() == cuda_resource);
	}

	SECTION("Multiple RandomCUDA instances") {
		constexpr size_t num_states1 = 64;
		constexpr size_t num_states2 = 256;

		Random<Resource> rng1(cuda_resource, num_states1);
		Random<Resource> rng2(cuda_resource, num_states2);

		rng1.init(111UL);
		rng2.init(222UL);
	}
}

TEST_CASE_METHOD(CUDARandomTestFixture,
				 "CUDA Random Uniform Float Generation",
				 "[cuda][random][uniform]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("Basic uniform float generation") {
		constexpr size_t num_states = 128;
		Random<Resource> rng(cuda_resource, num_states);
		rng.init(54321UL);

		const size_t count = 10000;
		DeviceBuffer<float> output(count, cuda_resource);

		// Generate uniform floats in default range [0, 1)
		Event event = rng.generate_uniform(output);
		event.wait();

		// Copy results to host for analysis
		std::vector<float> results(count);
		output.copy_to_host(results.data());

		Statistics stats(results);

		// Check that values are in [0, 1) range
		REQUIRE(stats.min_val >= 0.0f);
		REQUIRE(stats.max_val < 1.0f);

		// Check that mean is approximately 0.5 (within reasonable tolerance)
		REQUIRE_THAT(stats.mean, Catch::Matchers::WithinAbs(0.5f, 0.05f));

		// Check that we have reasonable spread (variance should be around 1/12 ≈ 0.083 for uniform
		// [0,1))
		REQUIRE_THAT(stats.variance, Catch::Matchers::WithinAbs(1.0f / 12.0f, 0.02f));
	}

	SECTION("Uniform float generation with custom range") {
		constexpr size_t num_states = 256;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(98765UL);

		const size_t count = 5000;
		const float min_val = -10.0f;
		const float max_val = 10.0f;

		DeviceBuffer<float> output(count, cuda_resource);

		Event event = rng.generate_uniform(output, min_val, max_val);
		event.wait();

		std::vector<float> results(count);
		output.copy_to_host(results.data());

		Statistics stats(results);

		// Check that values are in [min_val, max_val) range
		REQUIRE(stats.min_val >= min_val);
		REQUIRE(stats.max_val < max_val);

		// Check that mean is approximately (min_val + max_val) / 2
		float expected_mean = (min_val + max_val) / 2.0f;
		REQUIRE_THAT(stats.mean, Catch::Matchers::WithinAbs(expected_mean, 0.5f));
	}
}

TEST_CASE_METHOD(CUDARandomTestFixture,
				 "CUDA Random Gaussian Generation",
				 "[cuda][random][gaussian]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("Standard normal distribution") {
		constexpr size_t num_states = 128;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(13579UL);

		const size_t count = 10000;
		DeviceBuffer<float> output(count, cuda_resource);

		// Generate standard normal (mean=0, stddev=1)
		Event event = rng.generate_gaussian(output);
		event.wait();

		std::vector<float> results(count);
		output.copy_to_host(results.data());

		Statistics stats(results);

		// Check that mean is approximately 0 (standard normal distribution)
		REQUIRE_THAT(stats.mean, Catch::Matchers::WithinAbs(0.0f, 0.1f));

		// Check that variance is approximately 1 (standard normal distribution)
		REQUIRE_THAT(stats.variance, Catch::Matchers::WithinAbs(1.0f, 0.2f));

		// Check that most values are within reasonable bounds (99.7% should be within 3 standard
		// deviations)
		int outliers = 0;
		for (float val : results) {
			if (std::abs(val) > 4.0f) { // Being generous with 4 sigma
				outliers++;
			}
		}
		REQUIRE(outliers < count * 0.01); // Less than 1% outliers
	}

	SECTION("Custom normal distribution") {
		constexpr size_t num_states = 256;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(24680UL);

		const size_t count = 5000;
		const float mean = 5.0f;
		const float stddev = 2.0f;

		DeviceBuffer<float> output(count, cuda_resource);

		Event event = rng.generate_gaussian(output, mean, stddev);
		event.wait();

		std::vector<float> results(count);
		output.copy_to_host(results.data());

		Statistics stats(results);

		// Check that mean is approximately the specified mean
		REQUIRE_THAT(stats.mean, Catch::Matchers::WithinAbs(mean, 0.2f));

		// Check that variance is approximately stddev^2
		float expected_variance = stddev * stddev;
		REQUIRE_THAT(stats.variance, Catch::Matchers::WithinAbs(expected_variance, 0.8f));
	}
}

TEST_CASE_METHOD(CUDARandomTestFixture,
				 "CUDA Random Vector3 Generation",
				 "[cuda][random][vector3]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("Gaussian Vector3 generation") {
		constexpr size_t num_states = 128;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(11111UL);

		const size_t count = 1000;
		DeviceBuffer<Vector3> output(count, cuda_resource);

		Event event = rng.generate_gaussian_vector3(output);
		event.wait();

		// Copy results to host for analysis
		std::vector<Vector3> results(count);
		output.copy_to_host(results.data());

		// Extract components for statistical analysis
		std::vector<float> x_components, y_components, z_components;
		for (const auto& vec : results) {
			x_components.push_back(vec.x);
			y_components.push_back(vec.y);
			z_components.push_back(vec.z);
		}

		Statistics x_stats(x_components);
		Statistics y_stats(y_components);
		Statistics z_stats(z_components);

		// Each component should follow standard normal distribution
		REQUIRE_THAT(x_stats.mean, Catch::Matchers::WithinAbs(0.0f, 0.2f));
		REQUIRE_THAT(y_stats.mean, Catch::Matchers::WithinAbs(0.0f, 0.2f));
		REQUIRE_THAT(z_stats.mean, Catch::Matchers::WithinAbs(0.0f, 0.2f));

		REQUIRE_THAT(x_stats.variance, Catch::Matchers::WithinAbs(1.0f, 0.3f));
		REQUIRE_THAT(y_stats.variance, Catch::Matchers::WithinAbs(1.0f, 0.3f));
		REQUIRE_THAT(z_stats.variance, Catch::Matchers::WithinAbs(1.0f, 0.3f));
	}
}

TEST_CASE_METHOD(CUDARandomTestFixture,
				 "CUDA Random Integer Generation",
				 "[cuda][random][integer]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("Uniform integer generation") {
		constexpr size_t num_states = 128;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(55555UL);

		const size_t count = 10000;
		const int min_val = 1;
		const int max_val = 100;

		DeviceBuffer<int> output(count, cuda_resource);

		Event event = rng.generate_uniform_int(output, min_val, max_val);
		event.wait();

		std::vector<int> results(count);
		output.copy_to_host(results.data());

		// Check that all values are in range
		for (int val : results) {
			REQUIRE(val >= min_val);
			REQUIRE(val <= max_val);
		}

		// Check distribution uniformity by counting occurrences
		std::vector<int> counts(max_val - min_val + 1, 0);
		for (int val : results) {
			counts[val - min_val]++;
		}

		// Expected count per value
		float expected_count = static_cast<float>(count) / (max_val - min_val + 1);

		// Check that each value appears with reasonable frequency (within 20% of expected)
		for (int count_val : counts) {
			REQUIRE_THAT(static_cast<float>(count_val),
						 Catch::Matchers::WithinAbs(expected_count, expected_count * 0.2f));
		}
	}

	SECTION("Unsigned integer generation") {
		constexpr size_t num_states = 256;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(77777UL);

		const size_t count = 5000;
		const unsigned int min_val = 10;
		const unsigned int max_val = 50;

		DeviceBuffer<unsigned int> output(count, cuda_resource);

		Event event = rng.generate_uniform_uint(output, min_val, max_val);
		event.wait();

		std::vector<unsigned int> results(count);
		output.copy_to_host(results.data());

		// Check that all values are in range
		for (unsigned int val : results) {
			REQUIRE(val >= min_val);
			REQUIRE(val <= max_val);
		}
	}
}

TEST_CASE_METHOD(CUDARandomTestFixture,
				 "CUDA Random Seed Reproducibility",
				 "[cuda][random][seed]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("Same seed produces same sequence") {
		constexpr size_t num_states = 128;
		const unsigned long seed = 42UL;
		const size_t count = 1000;

		// First generator
		RandomCUDA<num_states> rng1(cuda_resource);
		rng1.init(seed);

		DeviceBuffer<float> output1(count, cuda_resource);
		Event event1 = rng1.generate_uniform(output1);
		event1.wait();

		std::vector<float> results1(count);
		output1.copy_to_host(results1.data());

		// Second generator with same seed
		RandomCUDA<num_states> rng2(cuda_resource);
		rng2.init(seed);

		DeviceBuffer<float> output2(count, cuda_resource);
		Event event2 = rng2.generate_uniform(output2);
		event2.wait();

		std::vector<float> results2(count);
		output2.copy_to_host(results2.data());

		// Results should be identical
		for (size_t i = 0; i < count; ++i) {
			REQUIRE_THAT(results1[i], Catch::Matchers::WithinAbs(results2[i], 1e-10f));
		}
	}

	SECTION("Different seeds produce different sequences") {
		constexpr size_t num_states = 128;
		const size_t count = 1000;

		// First generator
		RandomCUDA<num_states> rng1(cuda_resource);
		rng1.init(12345UL);

		DeviceBuffer<float> output1(count, cuda_resource);
		Event event1 = rng1.generate_uniform(output1);
		event1.wait();

		std::vector<float> results1(count);
		output1.copy_to_host(results1.data());

		// Second generator with different seed
		RandomCUDA<num_states> rng2(cuda_resource);
		rng2.init(54321UL);

		DeviceBuffer<float> output2(count, cuda_resource);
		Event event2 = rng2.generate_uniform(output2);
		event2.wait();

		std::vector<float> results2(count);
		output2.copy_to_host(results2.data());

		// Results should be different (at least some values)
		int different_count = 0;
		for (size_t i = 0; i < count; ++i) {
			if (std::abs(results1[i] - results2[i]) > 1e-10f) {
				different_count++;
			}
		}

		// Expect most values to be different
		REQUIRE(different_count > count * 0.9);
	}
}

TEST_CASE_METHOD(CUDARandomTestFixture, "CUDA Random Double Precision", "[cuda][random][double]") {
	if (CUDAManager::all_devices().empty()) {
		SKIP("No CUDA devices available");
	}

	SECTION("Double precision uniform generation") {
		constexpr size_t num_states = 128;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(99999UL);

		const size_t count = 5000;
		const double min_val = 0.0;
		const double max_val = 1.0;

		DeviceBuffer<double> output(count, cuda_resource);

		Event event = rng.generate_uniform(output, min_val, max_val);
		event.wait();

		std::vector<double> results(count);
		output.copy_to_host(results.data());

		// Check range
		for (double val : results) {
			REQUIRE(val >= min_val);
			REQUIRE(val < max_val);
		}

		// Check basic statistics
		double sum = std::accumulate(results.begin(), results.end(), 0.0);
		double mean = sum / results.size();
		REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.5, 0.05));
	}

	SECTION("Double precision gaussian generation") {
		constexpr size_t num_states = 256;
		RandomCUDA<num_states> rng(cuda_resource);
		rng.init(88888UL);

		const size_t count = 5000;
		const double mean = 0.0;
		const double stddev = 1.0;

		DeviceBuffer<double> output(count, cuda_resource);

		Event event = rng.generate_gaussian(output, mean, stddev);
		event.wait();

		std::vector<double> results(count);
		output.copy_to_host(results.data());

		// Check basic statistics
		double sum = std::accumulate(results.begin(), results.end(), 0.0);
		double actual_mean = sum / results.size();
		REQUIRE_THAT(actual_mean, Catch::Matchers::WithinAbs(mean, 0.1));

		// Check that most values are within reasonable bounds
		int outliers = 0;
		for (double val : results) {
			if (std::abs(val) > 4.0) {
				outliers++;
			}
		}
		REQUIRE(outliers < count * 0.01);
	}
}

TEST_CASE("Random Host-Only Tests", "[random][host]") {
	SECTION("RandomCPU basic functionality") {
		// Test CPU random number generation for comparison
		RandomCPU<> cpu_rng;
		cpu_rng.init(12345UL);

		// Generate some numbers to ensure it works
		std::vector<float> values;
		for (int i = 0; i < 100; ++i) {
			values.push_back(cpu_rng.uniform());
		}

		// Basic sanity checks
		REQUIRE(values.size() == 100);
		for (float val : values) {
			REQUIRE(val >= 0.0f);
			REQUIRE(val < 1.0f);
		}

		// Check that we get different values (not all the same)
		bool all_same = true;
		for (size_t i = 1; i < values.size(); ++i) {
			if (std::abs(values[i] - values[0]) > 1e-10f) {
				all_same = false;
				break;
			}
		}
		REQUIRE_FALSE(all_same);
	}
}

#endif // USE_CUDA
