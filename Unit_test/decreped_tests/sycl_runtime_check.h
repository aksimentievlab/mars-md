#pragma once

#ifdef USE_SYCL
#include <string>
#include <cstdlib>

/**
 * @brief Check if SYCL runtime is stable on this platform
 * 
 * AdaptiveCpp has known compatibility issues on macOS ARM64 with threading
 * and memory management. This function provides a way to detect and skip
 * tests that would crash the runtime.
 * 
 * @return true if SYCL runtime appears stable, false if unstable
 */
inline bool is_sycl_runtime_stable() {
    // Check for macOS ARM64 - known unstable with AdaptiveCpp
    #ifdef __APPLE__
        #ifdef __arm64__
            // AdaptiveCpp has threading issues on macOS ARM64
            const char* acpp_version = std::getenv("ACPP_VERSION");
            const char* acpp_targets = std::getenv("ACPP_TARGETS");
            
            // Check for AdaptiveCpp indicators
            if (acpp_version || acpp_targets) {
                return false; // Known unstable
            }
            
            // Also check if we're using AdaptiveCpp by default on macOS ARM64
            // Since AdaptiveCpp is the primary SYCL implementation for this platform
            // and has known threading issues, err on the side of caution
            return false;
        #endif
    #endif
    
    // Default to assuming stable
    return true;
}

/**
 * @brief Macro to skip SYCL tests on unstable platforms
 */
#define SKIP_IF_SYCL_UNSTABLE() \
    do { \
        if (!is_sycl_runtime_stable()) { \
            SKIP("SYCL runtime is unstable on this platform - skipping test to prevent crashes"); \
        } \
    } while(0)

#else

// No-op definitions when SYCL is not enabled
inline bool is_sycl_runtime_stable() { return false; }
#define SKIP_IF_SYCL_UNSTABLE() SKIP("SYCL not enabled")

#endif // USE_SYCL