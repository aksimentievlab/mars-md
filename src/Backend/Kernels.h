#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Buffer.h"
#include "Events.h"
#include "Header.h"
#include "KernelConfig.h"
#include "Resource.h"
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef USE_SYCL
#include "SYCL/SYCLKernels.h"
#include "SYCL/SYCLManager.h"
#endif

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/tuple.h>
#ifdef __CUDACC__
#include "CUDA/KernelHelper.cuh"
#endif
#endif

namespace ARBD {

template<typename Functor, typename... Args>
Event launch_kernel(const Resource& resource,
					idx_t thread_count,
					const KernelConfig& config,
					Functor kernel_func,
					Args... args) {

#ifdef USE_CUDA
	if (resource.type == ResourceType::CUDA) {
		return launch_cuda_kernel(resource, thread_count, config, kernel_func, args...);
	}
#endif

#ifdef USE_SYCL
	if (resource.type == ResourceType::SYCL) {
		return launch_sycl_kernel(resource, thread_count, config, kernel_func, args...);
	}
#endif

#ifdef USE_METAL
	if (resource.type == ResourceType::METAL) {
		return launch_metal_kernel(resource, thread_count, config, kernel_func, args...);
	}
#endif

	// CPU fallback
	return launch_cpu_kernel(resource, thread_count, config, kernel_func, args...);
}

template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
std::enable_if_t<!is_device_buffer_v<InputTuple> && !is_device_buffer_v<OutputTuple>, Event>
launch_kernel(const Resource& resource,
			  idx_t thread_count,
			  const KernelConfig& config,
			  InputTuple& inputs,
			  OutputTuple& outputs,
			  Functor&& kernel_func,
			  Args&&... args) {
	try {
#ifdef USE_CUDA
		return launch_cuda_kernel(resource,
								  thread_count,
								  inputs,
								  outputs,
								  config,
								  std::forward<Functor>(kernel_func),
								  std::forward<Args>(args)...);
#elif defined(USE_SYCL)
		return launch_sycl_kernel(resource,
								  thread_count,
								  inputs,
								  outputs,
								  config,
								  std::forward<Functor>(kernel_func),
								  std::forward<Args>(args)...);
#elif defined(USE_METAL)
		throw_value_error("METAL backend requires a kernel name (string), not a functor. "
						  "Please use launch_metal_kernel.");
#else
		return launch_cpu_kernel(resource,
								 thread_count,
								 inputs,
								 outputs,
								 config,
								 std::forward<Functor>(kernel_func),
								 std::forward<Args>(args)...);
#endif
	} catch (const std::exception& e) {
		LOGERROR("Error in launch_kernel: {}", e.what());
		throw;
	}
}

/**
 * @brief Single output buffer (generators like Random)
 */
template<typename OutputBuffer, typename Functor, typename... Args>
std::enable_if_t<is_device_buffer_v<OutputBuffer> && !is_device_buffer_v<Functor> &&
					 !is_string_v<Functor>,
				 Event>
launch_kernel(const Resource& resource,
			  idx_t thread_count,
			  const KernelConfig& config,
			  OutputBuffer& output_buffer,
			  Functor&& kernel_func,
			  Args&&... args) {

	auto inputs = std::make_tuple();
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_kernel(resource,
						 thread_count,
						 config,
						 inputs,
						 outputs,
						 std::forward<Functor>(kernel_func),
						 std::forward<Args>(args)...);
}

/**
 * @brief Single input + single output buffers (transforms)
 */
template<typename InputBuffer, typename OutputBuffer, typename Functor, typename... Args>
std::enable_if_t<is_device_buffer_v<InputBuffer> && is_device_buffer_v<OutputBuffer> &&
					 !is_device_buffer_v<Functor> && !is_string_v<Functor>,
				 Event>
launch_kernel(const Resource& resource,
			  idx_t thread_count,
			  const KernelConfig& config,
			  InputBuffer& input_buffer,
			  OutputBuffer& output_buffer,
			  Functor&& kernel_func,
			  Args&&... args) {

	auto inputs = std::make_tuple(std::ref(input_buffer));
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_kernel(resource,
						 thread_count,
						 config,
						 inputs,
						 outputs,
						 std::forward<Functor>(kernel_func),
						 std::forward<Args>(args)...);
}

/**
 * @brief Dual input + single output buffers (binary operations)
 */
template<typename InputBuffer1,
		 typename InputBuffer2,
		 typename OutputBuffer,
		 typename Functor,
		 typename... Args>
std::enable_if_t<is_device_buffer_v<InputBuffer1> && is_device_buffer_v<InputBuffer2> &&
					 is_device_buffer_v<OutputBuffer> && !is_device_buffer_v<Functor> &&
					 !is_string_v<Functor>,
				 Event>
launch_kernel(const Resource& resource,
			  idx_t thread_count,
			  const KernelConfig& config,
			  InputBuffer1& input_buffer1,
			  InputBuffer2& input_buffer2,
			  OutputBuffer& output_buffer,
			  Functor&& kernel_func,
			  Args&&... args) {

	auto inputs = std::make_tuple(std::ref(input_buffer1), std::ref(input_buffer2));
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_kernel(resource,
						 thread_count,
						 config,
						 inputs,
						 outputs,
						 std::forward<Functor>(kernel_func),
						 std::forward<Args>(args)...);
}

// ============================================================================
// CPU Kernel Launcher (Host-only)
// ============================================================================

/**
 * @brief CPU kernel launcher - tuple-based interface
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						idx_t thread_count,
						const InputTuple& inputs,
						const OutputTuple& outputs,
						const KernelConfig& config,
						Functor&& kernel_func,
						Args... args) {

	config.dependencies.wait_all();

	auto input_ptrs = extract_buffer_pointers(inputs);
	auto output_ptrs = extract_buffer_pointers(outputs);

	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) {
		num_threads = 1;
	}

	std::vector<std::thread> threads;
	idx_t chunk_size = (thread_count + num_threads - 1) / num_threads;

	for (unsigned int t = 0; t < num_threads; ++t) {
		threads.emplace_back([=]() {
			idx_t start = t * chunk_size;
			idx_t end = std::min(start + chunk_size, thread_count);
			for (idx_t i = start; i < end; ++i) {
				auto all_args = std::tuple_cat(input_ptrs, output_ptrs);
				std::apply([&](auto&&... unpacked_args) { kernel_func(i, unpacked_args...); },
						   all_args);
			}
		});
	}

	for (auto& thread : threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}

	return Event(nullptr, resource);
}

/**
 * @brief CPU kernel launcher - single output buffer
 */
template<typename OutputBuffer, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						idx_t thread_count,
						const KernelConfig& config,
						OutputBuffer& output_buffer,
						Functor&& kernel_func,
						Args&&... args) {

	auto inputs = std::make_tuple();
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_cpu_kernel(resource,
							 thread_count,
							 inputs,
							 outputs,
							 config,
							 std::forward<Functor>(kernel_func),
							 std::forward<Args>(args)...);
}

/**
 * @brief CPU kernel launcher - single input, single output buffer
 */
template<typename InputBuffer, typename OutputBuffer, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						idx_t thread_count,
						const KernelConfig& config,
						InputBuffer& input_buffer,
						OutputBuffer& output_buffer,
						Functor&& kernel_func,
						Args&&... args) {

	auto inputs = std::make_tuple(std::ref(input_buffer));
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_cpu_kernel(resource,
							 thread_count,
							 inputs,
							 outputs,
							 config,
							 std::forward<Functor>(kernel_func),
							 std::forward<Args>(args)...);
}

/**
 * @brief Kernel chaining
 * Kernel chain on a single resource with same stream/Queue
 */
template<typename Backend>
class KernelChain {
  private:
	const Resource& resource_;
	EventList events_;

  public:
	explicit KernelChain(const Resource& resource) : resource_(resource) {}

	template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
	KernelChain& then(idx_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  Functor&& kernel,
					  const KernelConfig& config = {},
					  Args&&... args) {

		KernelConfig new_config = config;
		new_config.dependencies = events_;
		new_config.sync = false;

		Event completion_event = launch_kernel<Backend>(resource_,
														thread_count,
														inputs,
														outputs,
														new_config,
														std::forward<Functor>(kernel),
														std::forward<Args>(args)...);

		events_.clear();
		events_.add(completion_event);
		return *this;
	}

	template<typename InputTuple, typename OutputTuple, typename... Args>
	KernelChain& then(idx_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  const std::string& kernel_name,
					  const KernelConfig& config = {},
					  Args&&... args) {

		KernelConfig new_config = config;
		new_config.dependencies = events_;
		new_config.sync = false;

		Event completion_event = launch_kernel<Backend>(resource_,
														thread_count,
														inputs,
														outputs,
														new_config,
														kernel_name,
														std::forward<Args>(args)...);

		events_.clear();
		events_.add(completion_event);
		return *this;
	}

	void wait() {
		events_.wait_all();
	}
};

// ============================================================================
// Result Wrapper for Kernel Calls
// ============================================================================

template<typename T>
struct KernelResult {
	T result;
	Event completion_event;

	KernelResult(T&& res, Event&& event)
		: result(std::forward<T>(res)), completion_event(std::move(event)) {}

	void wait() {
		completion_event.wait();
	}

	bool is_ready() const {
		return completion_event.is_complete();
	}

	T get() {
		wait();
		return std::move(result);
	}
};

class KernelGraph {
  private:
	struct KernelNode {
		std::function<Event()> launcher;
		std::vector<size_t> dependencies;
		size_t node_id;
		std::string name;
		Event completion_event;
		bool executed{false};
	};

#ifdef USE_CUDA
	cudaGraph_t cuda_graph_{nullptr};
	cudaGraphExec_t cuda_graph_instance_{nullptr};
	bool is_recorded_{false};
#endif

#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
	using command_graph = sycl::ext::oneapi::experimental::command_graph<
		sycl::ext::oneapi::experimental::graph_state::modifiable>;
	using executable_graph = sycl::ext::oneapi::experimental::command_graph<
		sycl::ext::oneapi::experimental::graph_state::executable>;
	command_graph* sycl_graph_{nullptr};
	executable_graph* sycl_exec_graph_{nullptr};
	bool sycl_graph_recorded_{false};
#endif

	std::vector<KernelNode> nodes_;
	const Resource& resource_;

  public:
	explicit KernelGraph(const Resource& resource) : resource_(resource) {
#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
		if (resource.type == ResourceType::SYCL) {
			sycl::queue& q = *static_cast<sycl::queue*>(resource.get_stream());
			sycl_graph_ = new command_graph(q.get_context(), q.get_device());
		}
#endif
	}

	~KernelGraph() {
#ifdef USE_CUDA
		if (cuda_graph_instance_)
			cudaGraphExecDestroy(cuda_graph_instance_);
		if (cuda_graph_)
			cudaGraphDestroy(cuda_graph_);
#endif
#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
		if (sycl_exec_graph_)
			delete sycl_exec_graph_;
		if (sycl_graph_)
			delete sycl_graph_;
#endif
	}

	// Add kernel with direct arguments (zero overhead)
	template<typename Functor, typename... Args>
	size_t add_kernel(const std::string& name,
					  idx_t thread_count,
					  Functor kernel_func,
					  const KernelConfig& base_config,
					  Args... args) {

		size_t node_id = nodes_.size();

		// Zero-overhead launcher - capture by value, no std::forward
		auto launcher = [=, this]() -> Event {
			KernelConfig config = base_config;
			config.async = true;
			return launch_kernel(resource_, thread_count, config, kernel_func, args...);
		};

		nodes_.emplace_back(KernelNode{launcher, {}, node_id, name, Event{}, false});

		return node_id;
	}

	void add_dependency(size_t dependent, size_t dependency) {
		if (dependent < nodes_.size() && dependency < nodes_.size()) {
			nodes_[dependent].dependencies.push_back(dependency);
		}
	}

	EventList execute() {
#ifdef USE_CUDA
		if (resource_.type == ResourceType::CUDA && !nodes_.empty()) {
			if (!is_recorded_) {
				record_cuda_graph();
			}

			cudaStream_t stream = static_cast<cudaStream_t>(resource_.get_stream());
			CUDA_CHECK(cudaGraphLaunch(cuda_graph_instance_, stream));

			cudaEvent_t completion_event;
			CUDA_CHECK(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
			CUDA_CHECK(cudaEventRecord(completion_event, stream));

			EventList result;
			result.add(Event(completion_event, resource_));
			return result;
		}
#endif

#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
		if (resource_.type == ResourceType::SYCL && !nodes_.empty() && sycl_exec_graph_) {
			if (!sycl_graph_recorded_) {
				record_sycl_graph();
			}

			sycl::queue& q = *static_cast<sycl::queue*>(resource_.get_stream());
			auto sycl_event =
				q.submit([&](sycl::handler& h) { h.ext_oneapi_graph(*sycl_exec_graph_); });

			EventList result;
			result.add(Event(sycl_event, resource_));
			return result;
		}
#endif

		return execute_topologically();
	}

  private:
#ifdef USE_CUDA
	void record_cuda_graph() {
		cudaStream_t stream = static_cast<cudaStream_t>(resource_.get_stream());

		CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
		execute_topologically();
		CUDA_CHECK(cudaStreamEndCapture(stream, &cuda_graph_));
		CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_instance_, cuda_graph_, nullptr, nullptr, 0));
		is_recorded_ = true;
	}
#endif

#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
	void record_sycl_graph() {
		// Record SYCL graph nodes
		for (auto& node : nodes_) {
			sycl_graph_->add([&](sycl::handler& h) {
				// Add kernel to graph
				node.launcher();
			});
		}

		sycl_exec_graph_ = new executable_graph(sycl_graph_->finalize());
		sycl_graph_recorded_ = true;
	}
#endif

	EventList execute_topologically() {
		EventList all_events;
		std::vector<bool> visited(nodes_.size(), false);

		std::function<void(idx_t)> execute_node;
		execute_node = [&](idx_t node_id) {
			if (visited[node_id] || nodes_[node_id].executed)
				return;

			auto& node = nodes_[node_id];

			for (idx_t dep_id : node.dependencies) {
				execute_node(dep_id);
			}

			for (idx_t dep_id : node.dependencies) {
				nodes_[dep_id].completion_event.wait();
			}

			node.completion_event = node.launcher();
			node.executed = true;
			visited[node_id] = true;

			all_events.add(node.completion_event);
		};

		for (idx_t i = 0; i < nodes_.size(); ++i) {
			execute_node(i);
		}

		return all_events;
	}
};

/**
 * @brief Kernel Pipeline
 * A pipeline of kernels that are executed in order.
 * The pipeline is executed in order, and the output of each kernel is used as the input to the next
 * kernel.
 * @param resource The resource to use for the pipeline.
 * @param stream_id The stream ID to use for the pipeline.
 */
class KernelPipeline {
  private:
	const Resource& resource_;
	void* dedicated_queue_;
	EventList pipeline_events_;

  public:
	explicit KernelPipeline(const Resource& resource, int stream_id = 0) : resource_(resource) {
		dedicated_queue_ =
			(stream_id == 0) ? resource.get_stream() : resource.get_stream(stream_id);
	}

	template<typename Functor, typename... Args>
	KernelPipeline&
	then(idx_t thread_count, Functor kernel_func, const KernelConfig& base_config, Args... args) {

		KernelConfig config = base_config;
		config.explicit_queue = dedicated_queue_;
		config.dependencies = pipeline_events_;
		config.async = true;

		Event completion = launch_kernel(resource_, thread_count, config, kernel_func, args...);

		pipeline_events_.clear();
		pipeline_events_.add(completion);

		return *this;
	}

	void synchronize() {
		pipeline_events_.wait_all();
	}

	EventList get_events() const {
		return pipeline_events_;
	}
};

} // namespace ARBD
