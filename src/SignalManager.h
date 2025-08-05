#pragma once

#include <csignal> 

#ifdef SIGNAL
// These are POSIX specific, might need guards for other OS if portability is a concern
extern "C" { 
    #include <execinfo.h> 
    #ifndef __USE_GNU
        #define __USE_GNU
    #endif
    #include <ucontext.h>
}

#if __WORDSIZE == 64 
#define MY_REG_RIP REG_RIP
#else
#define MY_REG_RIP REG_EIP
#endif

#endif // SIGNAL

namespace ARBD {
/**
 * @namespace SignalManager
 * @brief Namespace containing signal handling and management functionality.
 * 
 * This namespace provides utilities for handling system signals, particularly
 * segmentation faults, and managing program shutdown requests.
 */
namespace SignalManager {
    
    /**
     * @brief Handles segmentation fault signals.
     * 
     * This function is called when a segmentation fault occurs. It prints
     * detailed information about the fault location and stack trace.
     * 
     * @param sig The signal number
     * @param info Pointer to siginfo_t structure containing signal information
     * @param secret Pointer to ucontext_t structure containing signal context
     */
    void segfault_handler(int sig, siginfo_t *info, void *secret);

    /**
     * @brief Sets up signal handling for segmentation faults.
     * 
     * This function configures the system to use the custom segfault_handler
     * for handling segmentation faults. It should be called during program
     * initialization.
     */
    void manage_segfault();

    /**
     * @brief Global flag indicating if a shutdown has been requested.
     * 
     * This atomic variable is used to coordinate program shutdown across
     * different threads. When set to non-zero, it indicates that the program
     * should begin shutdown procedures.
     */
    extern volatile sig_atomic_t shutdown_requested; 
    
    /**
     * @brief Checks if a shutdown has been requested.
     * 
     * @return true if shutdown has been requested, false otherwise
     */
    inline bool is_shutdown_requested() { return shutdown_requested != 0; }

} 
}