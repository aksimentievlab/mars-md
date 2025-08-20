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

class DeviceMesh {
  private:
	std::vector<Resource> resources_;
	std::unordered_map<int, void*> queues_; // device_id -> queue
	bool peer_access_enabled_ = false;

  public:
	DeviceMesh() = default;

	explicit DeviceMesh(std::vector<Resource> resources) : resources_(resources) {
		for (const auto& res : resources_) {
#ifdef USE_CUDA
			if (res.type == ResourceType::CUDA) {
				cudaStream_t queue;
				CUDA_CHECK(cudaSetDevice(res.id));
				CUDA_CHECK(cudaStreamCreate(&queue));
				queues_[res.id] = queue;
			}
#endif

#ifdef USE_SYCL
			if (res.type == ResourceType::SYCL) {
				auto& device = SYCL::Manager::get_device(res.id);
				queues_[res.id] = &device.get_queue(0);
			}
#endif
		}

		enable_peer_access();
	}

	~DeviceMesh() {
#ifdef USE_CUDA
		for (const auto& res : resources_) {
			if (res.type == ResourceType::CUDA) {
				cudaSetDevice(res.id);
				cudaStreamDestroy(static_cast<cudaStream_t>(queues_[res.id]));
			}
		}
#endif
	}

	void enable_peer_access() {
#ifdef USE_CUDA
		for (size_t i = 0; i < resources_.size(); ++i) {
			if (resources_[i].type != ResourceType::CUDA)
				continue;

			cudaSetDevice(resources_[i].id);
			for (size_t j = 0; j < resources_.size(); ++j) {
				if (i != j && resources_[j].type == ResourceType::CUDA) {
					int can_access;
					cudaDeviceCanAccessPeer(&can_access, resources_[i].id, resources_[j].id);
					if (can_access) {
						cudaDeviceEnablePeerAccess(resources_[j].id, 0);
					}
				}
			}
		}
#endif
		peer_access_enabled_ = true;
	}

	void* get_queue(int device_id) {
		auto it = queues_.find(device_id);
		return (it != queues_.end()) ? it->second : nullptr;
	}
};
} // namespace ARBD
