#include <future>
#include <cstring>

#include "Backend/Proxy.h"

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

namespace ARBD {
namespace ProxyImpl {

// =============================================================================
// METAL Implementation Functions
// =============================================================================

#ifdef USE_METAL
void* metal_call_sync(void* addr, void* func_ptr, void* args, size_t args_size, 
                     const Resource& location, size_t result_size) {
    if (location.is_local()) {
        auto &device = ARBD::METAL::METALManager::get_current_device();

        // Create result storage
        void* result = new char[result_size];
        
        // For METAL, execute on device (simplified for now)
        // Note: Full METAL implementation would require compute shaders
        // This is a placeholder implementation
        std::memcpy(result, addr, std::min(result_size, sizeof(void*)));
        
        return result;
    } else {
        ARBD::throw_not_implemented("Non-local METAL calls not supported");
    }
}

std::future<void*> metal_call_async(void* addr, void* func_ptr, void* args, 
                                    size_t args_size, const Resource& location, 
                                    size_t result_size) {
    if (location.is_local()) {
        return std::async(std::launch::async, [=] {
            return metal_call_sync(addr, func_ptr, args, args_size, location, result_size);
        });
    } else {
        ARBD::throw_not_implemented("Proxy::callAsync() non-local METAL calls");
    }
}
#endif

} // namespace ProxyImpl
} // namespace ARBD 