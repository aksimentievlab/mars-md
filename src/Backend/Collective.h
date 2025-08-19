// src/Backend/Collectives.h
#pragma once
#include "Buffer.h"
#include "Events.h"
#include "Resource.h"

#ifdef USE_NCCL
#include "CUDA/NCCLManager.h"
#endif
#include "MPIManager.h"

namespace ARBD {

class Collectives {
  public:
	enum class Backend { AUTO, MPI, NCCL, ONECCL };

  private:
	Resource resource_;
	Backend active_backend_;

	Backend select_backend(const Resource& res, Backend preferred) {
		if (preferred != Backend::AUTO) {
			return preferred;
		}

#ifdef USE_NCCL
		if (res.type == ResourceType::CUDA) {
			// Check if NCCLManager is initialized
			try {
				auto& nccl = NCCL::Manager::instance();
				if (nccl.is_initialized()) {
					return Backend::NCCL;
				}
			} catch (...) {
			}
		}
#endif

		// Default to MPI
		return Backend::MPI;
	}

  public:
	Collectives(const Resource& res, Backend preferred = Backend::AUTO)
		: resource_(res), active_backend_(select_backend(res, preferred)) {

		// Ensure MPI is initialized as fallback
		MPI::Manager::instance().init();

#ifdef USE_NCCL
		if (active_backend_ == Backend::NCCL) {
			NCCL::Manager::instance().init();
			LOGINFO("Using NCCL for collectives");
		} else
#endif
		{
			LOGINFO("Using MPI for collectives");
		}
	}

	template<typename T>
	Event allReduce(DeviceBuffer<T>& buffer, idx_t count) {
		switch (active_backend_) {
#ifdef USE_NCCL
		case Backend::NCCL:
			return NCCL::Manager::instance().allReduce(buffer, count, resource_);
#endif
		default:
			return MPI::Manager::instance().allReduce(buffer, count, resource_);
		}
	}

	template<typename T>
	Event broadcast(DeviceBuffer<T>& buffer, idx_t count, int root) {
		switch (active_backend_) {
#ifdef USE_NCCL
		case Backend::NCCL:
			return NCCL::Manager::instance().broadcast(buffer, count, root, resource_);
#endif
		default:
			return MPI::Manager::instance().broadcast(buffer, count, root, resource_);
		}
	}

	// Performance comparison helper
	void benchmark_backends() {
		DeviceBuffer<float> test_buffer(1024 * 1024); // 1M floats

		// Test MPI
		auto start = std::chrono::high_resolution_clock::now();
		MPI::Manager::instance().allReduce(test_buffer, test_buffer.size(), resource_);
		auto mpi_time = std::chrono::high_resolution_clock::now() - start;

#ifdef USE_NCCL
		if (resource_.type == ResourceType::CUDA) {
			start = std::chrono::high_resolution_clock::now();
			NCCL::Manager::instance().allReduce(test_buffer, test_buffer.size(), resource_);
			auto nccl_time = std::chrono::high_resolution_clock::now() - start;

			LOGINFO("MPI: {}ms, NCCL: {}ms",
					std::chrono::duration_cast<std::chrono::milliseconds>(mpi_time).count(),
					std::chrono::duration_cast<std::chrono::milliseconds>(nccl_time).count());
		}
#endif
	}
};

} // namespace ARBD
