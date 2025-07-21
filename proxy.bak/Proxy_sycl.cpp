#include "Backend/SYCL/SYCLManager.h"
#include "Backend/Proxy.h"
#include <sycl/sycl.hpp>
#include <functional>
#include <future>

namespace ARBD {
namespace ProxyImpl {

// Generic implementation for a synchronous SYCL call.
void* sycl_call_sync(void* addr, void* func_ptr, void* args, size_t args_size,
                    const Resource& location, size_t result_size) {
    if (!location.is_local()) {
        ARBD::throw_not_implemented("Non-local SYCL calls are not supported");
    }

    auto& device = SYCL::SYCLManager::get_current_device();
    auto& queue = device.get_next_queue();

    // 'args' is the std::function (lambda) that captures the real work.
    std::function<void*(void)>& func = *static_cast<std::function<void*(void)>*>(args);

    void* host_result = new char[result_size];
    void* device_result = sycl::malloc_device(result_size, queue);

    // This is a simplified execution model. A robust solution requires serializing
    // the lambda's captures, but for many cases, captures of simple types work.
    // The key is that the lambda 'func' contains the logic (e.g., array sum).
    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() {
            // Execute the captured function on the device.
            void* res = func();
            // Copy the result from its temporary location to the result buffer.
            if (res && result_size > 0) {
                memcpy(device_result, res, result_size);
            }
        });
    }).wait(); // wait() makes the call synchronous.

    queue.get().memcpy(host_result, device_result, result_size).wait();
    sycl::free(device_result, queue);

    return host_result;
}

// Generic implementation for an asynchronous SYCL call.
std::future<void*> sycl_call_async(void* addr, void* func_ptr, void* args,
                                   size_t args_size, const Resource& location,
                                   size_t result_size) {
    return std::async(std::launch::async, [=]() {
        // This lambda executes in a separate thread, returning the result when ready.
        return sycl_call_sync(addr, func_ptr, args, args_size, location, result_size);
    });
}

} // namespace ProxyImpl
} // namespace ARBD