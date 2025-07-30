#pragma once
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "RandomKernels.h"
#include "openrand/philox.h"

namespace ARBD {

template<typename Resource>
class Random {
  private:
	const Resource& resource_;
	uint64_t seed_;
	uint32_t base_ctr_;
	uint32_t global_seed_;

  public:
	explicit Random(const Resource& resource, size_t /* num_states - not needed for stateless */)
		: resource_(resource), seed_(0), base_ctr_(0), global_seed_(42) {
		if (!resource.is_device()) {
			throw std::invalid_argument("Random generator requires a valid device resource");
		}
	}

	void init(unsigned long seed, size_t offset = 0) {
		seed_ = static_cast<uint64_t>(seed);
		base_ctr_ = static_cast<uint32_t>(offset);
		global_seed_ = 42; // Fixed global seed for consistency
	}

	// --- UNIFORM DISTRIBUTION ---
	template<typename T>
	Event generate_uniform(DeviceBuffer<T>& output, T min_val, T max_val) {
		ARBD::KernelConfig config;
		UniformFunctor<T> func{min_val, max_val, seed_, base_ctr_, global_seed_};

		auto inputs = std::make_tuple(); // Empty tuple for no inputs
		auto outputs = std::make_tuple(std::ref(output));

		return launch_kernel(resource_, output.size(), config, inputs, outputs, func);
	}

	// --- GAUSSIAN DISTRIBUTION ---
	template<typename T>
	Event generate_gaussian(DeviceBuffer<T>& output, T mean, T stddev) {
		ARBD::KernelConfig config;
		GaussianFunctor<T> func{mean, stddev, output.size(), seed_, base_ctr_, global_seed_};

		auto inputs = std::make_tuple(); // Empty tuple for no inputs
		auto outputs = std::make_tuple(std::ref(output));

		return launch_kernel(resource_, output.size(), config, inputs, outputs, func);
	}

	// --- GAUSSIAN DISTRIBUTION FOR VECTOR3 ---
	template<typename T>
	Event
	generate_gaussian(DeviceBuffer<Vector3_t<T>>& output, Vector3_t<T> mean, Vector3_t<T> stddev) {
		ARBD::KernelConfig config;
		GaussianFunctor<Vector3_t<T>> func{mean,
										   stddev,
										   output.size(),
										   seed_,
										   base_ctr_,
										   global_seed_};

		auto inputs = std::make_tuple(); // Empty tuple for no inputs
		auto outputs = std::make_tuple(std::ref(output));

		return launch_kernel(resource_, output.size(), config, inputs, outputs, func);
	}
};

} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>

template<typename T>
struct sycl::is_device_copyable<ARBD::GaussianFunctor<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<ARBD::UniformFunctor<T>> : std::true_type {};

#endif
