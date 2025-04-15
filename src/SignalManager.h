/*
 * Some function in namespace to handle the
 * signal of segmentation fault.
 */

#ifndef SIGNALMANAGER_H_
#define SIGNALMANAGER_H_

#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ "(" S2(__LINE__)")" /* mainly used by ARBDException */

#ifdef USE_LOGGER

/* #define ARBD_LOG_ACTIVE_LEVEL 0 */
/* #include "logger.h" */
//*
#ifdef __CUDA_ARCH__
// #define LOGHELPER(TYPE, FMT,...) printf("[%d,%d] [%s] [%s]: %s", blockIdx.x, threadIdx.x, TYPE, LOCATION, FMT);
#define LOGHELPER(TYPE, FMT,...) printf("[%s] [%s]: %s\n", TYPE, LOCATION, FMT);
#define LOGTRACE(...) LOGHELPER("trace",  __VA_ARGS__)
#define LOGDEBUG(...) LOGHELPER("debug",__VA_ARGS__)
// #define DEBUG(...) spdlog::debug(__VA_ARGS__)
#define LOGINFO(...) LOGHELPER("info",__VA_ARGS__)
#define LOGWARN(...) LOGHELPER("warn",__VA_ARGS__)
#define LOGERROR(...) LOGHELPER("error",__VA_ARGS__)
#define LOGCRITICAL(...) LOGHELPER("critical",__VA_ARGS__)
#else
#define FMT_HEADER_ONLY
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#define LOGTRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOGDEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
// #define DEBUG(...) spdlog::debug(__VA_ARGS__)
#define LOGINFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOGWARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOGERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOGCRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
// spdlog::set_level(spdlog::level::trace);
#endif
//*/

/*
#define TRACE(...) ::arbd::log_trace(__VA_ARGS__)
#define DEBUG(...) ::arbd::log_debug(__VA_ARGS__)
// #define DEBUG(...) spdlog::debug(__VA_ARGS__)
#define INFO(...) ::arbd::log_info(__VA_ARGS__)
#define WARN(...) ::arbd::log_warn(__VA_ARGS__)
#define ERROR(...) ::arbd::log_error(__VA_ARGS__)
#define CRITICAL(...) ::arbd::log_critical(__VA_ARGS__)
//*/

#else

// Disable logger macros
// NOTE to developers: only use the macros below for logging, only in host code
#define LOGTRACE(...)
#define LOGDEBUG(...)
#define LOGINFO(...)
#define LOGWARN(...)
#define LOGERROR(...)
#define LOGCRITICAL(...)

#endif

// see http://www.linuxjournal.com/files/linuxjournal.com/linuxjournal/articles/063/6391/6391l3.html
#include <csignal>
#include <execinfo.h>

#ifdef SIGNAL
/* get REG_EIP from ucontext.h */
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <ucontext.h>

#if __WORDSIZE == 64
#define MY_REG_RIP REG_RIP
#else
#define MY_REG_RIP REG_EIP
#endif
#endif

namespace SignalManager 
{

    void segfault_handler(int sig, siginfo_t *info, void *secret);
    void manage_segfault();
}

#endif /* SIGNALMANAGER_H_ */
