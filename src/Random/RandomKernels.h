#pragma once
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Math/Vector3.h"
#include "openrand/philox.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
using namespace thrust;
#else
using namespace std;
#endif
#include <cmath>

// --- Functor for Uniform Float Generation ---
namespace ARBD {
HOST DEVICE inline float int2float(uint32_t i) {

	constexpr float factor = 1.0f / (4294967295.0f + 1.0f); // 1.0f / 2^32
	constexpr float halffactor = 0.5f * factor;
	return static_cast<float>(i) * factor + halffactor;
}

template<typename T>
struct UniformFunctor {
	T min_val;
	T max_val;
	uint64_t base_seed;
	uint32_t base_ctr;
	uint32_t global_seed;

	HOST DEVICE void operator()(size_t i, T* output) const {
		// Create a fresh Philox instance with deterministic parameters
		// Each thread gets a unique counter value based on its index
		openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i), global_seed);

		uint32_t random_int = rng.draw();

		float random_float_01 = int2float(random_int);
		output[i] = min_val + random_float_01 * (max_val - min_val);
	}
};

template<typename T>
struct GaussianFunctor {
	T mean;
	T stddev;
	size_t output_size;
	uint64_t base_seed;
	uint32_t base_ctr;
	uint32_t global_seed;

	HOST DEVICE void operator()(size_t i, T* output) const {
		if (i >= output_size)
			return;

		// Create a fresh Philox instance with deterministic parameters
		openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i), global_seed);
		uint32_t i1 = rng.draw();
		uint32_t i2 = rng.draw();

		float u1 = (int2float(i1) < 1e-7f) ? 1e-7f : int2float(i1);
		float u2 = (int2float(i2) < 1e-7f) ? 1e-7f : int2float(i2);

		// Box-Muller transform - generate one value per thread
		float r = sqrtf(-2.0f * logf(u1));
		float theta = 2.0f * 3.1415926535f * u2;
		float gaussian_val = r * cosf(theta);

		output[i] = mean + stddev * gaussian_val;
	}
};

// Specialization for Vector3_t types
template<typename T>
struct GaussianFunctor<ARBD::Vector3_t<T>> {
	ARBD::Vector3_t<T> mean;
	ARBD::Vector3_t<T> stddev;
	size_t output_size;
	uint64_t base_seed;
	uint32_t base_ctr;
	uint32_t global_seed;

	// The Box-Muller transform for generating pairs of Gaussian values
	HOST DEVICE ARBD::Vector3_t<T> box_muller(float u1, float u2) const {
		float r = sqrt(-2.0f * log(u1));
		float theta = 2.0f * 3.1415926535f * u2;
		return ARBD::Vector3_t<T>(r * cos(theta), r * sin(theta), 0.0f);
	}

	// Device code for Vector3_t types
	HOST DEVICE void operator()(size_t i, ARBD::Vector3_t<T>* output) const {
		if (i >= output_size)
			return;

		// Create a fresh Philox instance with deterministic parameters
		openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i), global_seed);

		uint32_t i1 = rng.draw();
		uint32_t i2 = rng.draw();
		uint32_t i3 = rng.draw();
		uint32_t i4 = rng.draw();
		uint32_t i5 = rng.draw();
		uint32_t i6 = rng.draw();

		// Generate three Gaussian values using Box-Muller (needs 3 uniform values)
		float u1_x = (int2float(i1) < 1e-7f) ? 1e-7f : int2float(i1);
		float u2_x = (int2float(i2) < 1e-7f) ? 1e-7f : int2float(i2);
		float u1_y = (int2float(i3) < 1e-7f) ? 1e-7f : int2float(i3);
		float u2_y = (int2float(i4) < 1e-7f) ? 1e-7f : int2float(i4);
		float u1_z = (int2float(i5) < 1e-7f) ? 1e-7f : int2float(i5);
		float u2_z = (int2float(i6) < 1e-7f) ? 1e-7f : int2float(i6);

		ARBD::Vector3_t<T> gauss_pair1 = box_muller(u1_x, u2_x);
		ARBD::Vector3_t<T> gauss_pair2 = box_muller(u1_y, u2_y);
		ARBD::Vector3_t<T> gauss_pair3 = box_muller(u1_z, u2_z);

		output[i] = ARBD::Vector3_t<T>(mean.x + stddev.x * gauss_pair1.x,
									   mean.y + stddev.y * gauss_pair2.x,
									   mean.z + stddev.z * gauss_pair3.x);
	}
};
} // namespace ARBD
