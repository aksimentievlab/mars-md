#include "Backend/Resource.h"
#include <map>
#include <mutex>
#include <vector>
#include <unordered_map>

// Include backend-specific Stream types
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

namespace ARBD {

// Stream type enumeration
enum class StreamType {
	Compute = 0,
	Memory = 1,
	Default = 2
};

class StreamPool {
  private:
	// Forward declare Stream type based on backend
#ifdef USE_CUDA
	using Stream = CUDA::Stream;
#elif defined(USE_SYCL)
	using Stream = SYCL::Queue;
#elif defined(USE_METAL)
	using Stream = METAL::Queue;
#else
	using Stream = void; // Fallback for CPU-only builds
#endif

	std::unordered_map<Resource, std::vector<std::unique_ptr<Stream>>> streams_;
	std::mutex mutex_;

  public:
	void* acquire_stream(const Resource& res, StreamType type = StreamType::Compute) {
		std::lock_guard<std::mutex> lock(mutex_);
		
		auto& resource_streams = streams_[res];
		
		// Try to find an available stream
		for (auto& stream : resource_streams) {
			if (stream) {
				// For now, just return the first available stream
				// In a more sophisticated implementation, you might track stream usage
				return static_cast<void*>(stream.get());
			}
		}
		
		// Create new stream if none available
		std::unique_ptr<Stream> new_stream;
		
#ifdef USE_CUDA
		if (res.type == ResourceType::CUDA) {
			new_stream = std::make_unique<CUDA::Stream>(static_cast<int>(res.id));
		}
#endif

#ifdef USE_SYCL
		if (res.type == ResourceType::SYCL) {
			auto& device = SYCL::Manager::get_device(res.id);
			new_stream = std::make_unique<SYCL::Queue>(device.get_device());
		}
#endif

#ifdef USE_METAL
		if (res.type == ResourceType::METAL) {
			auto& device = METAL::Manager::get_device(res.id);
			new_stream = std::make_unique<METAL::Queue>(device.get_next_queue());
		}
#endif

		if (new_stream) {
			resource_streams.push_back(std::move(new_stream));
			return static_cast<void*>(resource_streams.back().get());
		}
		
		return nullptr;
	}

	void release_stream(void* stream, const Resource& res) {
		// The stream remains available for reuse
		// In a more sophisticated implementation, you might mark it as available
		(void)stream; // Suppress unused parameter warning
		(void)res;    // Suppress unused parameter warning
	}

	void sync_all_streams(const Resource& res) {
		std::lock_guard<std::mutex> lock(mutex_);
		
		auto it = streams_.find(res);
		if (it != streams_.end()) {
			for (auto& stream : it->second) {
				if (stream) {
#ifdef USE_CUDA
					if (res.type == ResourceType::CUDA) {
						static_cast<CUDA::Stream*>(stream.get())->synchronize();
					}
#endif

#ifdef USE_SYCL
					if (res.type == ResourceType::SYCL) {
						static_cast<SYCL::Queue*>(stream.get())->synchronize();
					}
#endif

#ifdef USE_METAL
					if (res.type == ResourceType::METAL) {
						static_cast<METAL::Queue*>(stream.get())->synchronize();
					}
#endif
				}
			}
		}
	}
};

template<typename T>
class MemoryPool {
  private:
	struct PoolBlock {
		void* ptr = nullptr;
		size_t size = 0;
		Resource resource;
		bool in_use = false;
	};

	std::vector<PoolBlock> blocks_;
	std::mutex mutex_;

  public:
	// Allocate from pool or create new block
	void* allocate(size_t bytes, const Resource& resource) {
		std::lock_guard<std::mutex> lock(mutex_);

		// Try to find existing free block
		for (auto& block : blocks_) {
			if (!block.in_use && block.size >= bytes && block.resource == resource) {
				block.in_use = true;
				return block.ptr;
			}
		}

		// Allocate new block
		PoolBlock new_block;
		new_block.size = bytes;
		new_block.resource = resource;
		new_block.in_use = true;

#ifdef USE_CUDA
		if (resource.type == ResourceType::CUDA) {
			cudaSetDevice(resource.id);
			cudaMalloc(&new_block.ptr, bytes);
		}
#endif

#ifdef USE_SYCL
		if (resource.type == ResourceType::SYCL) {
			auto& device = SYCL::Manager::get_device(resource.id);
			new_block.ptr = sycl::malloc_device(bytes, device.get_queue(0));
		}
#endif

		blocks_.push_back(new_block);
		return new_block.ptr;
	}

	void deallocate(void* ptr) {
		std::lock_guard<std::mutex> lock(mutex_);

		for (auto& block : blocks_) {
			if (block.ptr == ptr) {
				block.in_use = false;
				return;
			}
		}
	}

	void clear() {
		std::lock_guard<std::mutex> lock(mutex_);

		for (auto& block : blocks_) {
#ifdef USE_CUDA
			if (block.resource.type == ResourceType::CUDA) {
				cudaFree(block.ptr);
			}
#endif

#ifdef USE_SYCL
			if (block.resource.type == ResourceType::SYCL) {
				auto& device = SYCL::Manager::get_device(block.resource.id);
				sycl::free(block.ptr, device.get_queue(0));
			}
#endif
		}

		blocks_.clear();
	}

	~MemoryPool() {
		clear();
	}
};

// Global memory pool for temporary allocations
inline MemoryPool<idx_t>& get_temp_pool() {
	static MemoryPool<idx_t> pool;
	return pool;
}

namespace MemoryAdvise {
#ifdef USE_CUDA
constexpr int READ_MOSTLY = cudaMemAdviseSetReadMostly;
constexpr int PREFERRED_LOCATION = cudaMemAdviseSetPreferredLocation;
constexpr int ACCESSED_BY = cudaMemAdviseSetAccessedBy;
#else
// SYCL or other backends
constexpr int READ_MOSTLY = 0;
constexpr int PREFERRED_LOCATION = 1;
constexpr int ACCESSED_BY = 2;
#endif
} // namespace MemoryAdvise

} // namespace ARBD
