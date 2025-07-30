#pragma once

#include "ARBDException.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// spdlog includes
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifndef MIN_DEBUG_LEVEL
#define MIN_DEBUG_LEVEL 0
#endif
#ifndef MAX_DEBUG_LEVEL
#define MAX_DEBUG_LEVEL 10
#endif
#ifndef STDERR_LEVEL
#define STDERR_LEVEL 5
#endif

namespace ARBD {

enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, CRITICAL = 5 };

class Logger {
  public:
	static LogLevel current_level;
	static std::shared_ptr<spdlog::logger> logger_instance;

	// Initialize the logger with spdlog
	static void initialize() {
		if (!logger_instance) {
			// Create a console logger with color support
			logger_instance = spdlog::stdout_color_mt("ARBD");

			// Set custom pattern with timestamp, level, file:line, and message
			logger_instance->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%g:%#] %v");

			// Set the default logger
			spdlog::set_default_logger(logger_instance);
		}
	}

	// Convert ARBD LogLevel to spdlog level
	static spdlog::level::level_enum to_spdlog_level(LogLevel level) {
		switch (level) {
		case LogLevel::TRACE:
			return spdlog::level::trace;
		case LogLevel::DEBUG:
			return spdlog::level::debug;
		case LogLevel::INFO:
			return spdlog::level::info;
		case LogLevel::WARN:
			return spdlog::level::warn;
		case LogLevel::ERROR:
			return spdlog::level::err;
		case LogLevel::CRITICAL:
			return spdlog::level::critical;
		default:
			return spdlog::level::info;
		}
	}

	// Convert spdlog level to ARBD LogLevel
	static LogLevel from_spdlog_level(spdlog::level::level_enum level) {
		switch (level) {
		case spdlog::level::trace:
			return LogLevel::TRACE;
		case spdlog::level::debug:
			return LogLevel::DEBUG;
		case spdlog::level::info:
			return LogLevel::INFO;
		case spdlog::level::warn:
			return LogLevel::WARN;
		case spdlog::level::err:
			return LogLevel::ERROR;
		case spdlog::level::critical:
			return LogLevel::CRITICAL;
		default:
			return LogLevel::INFO;
		}
	}

	// Host-side logging methods
	static void log(LogLevel level, const SourceLocation& loc, std::string_view message) {
		if (level < current_level)
			return;

		// Ensure logger is initialized
		if (!logger_instance) {
			initialize();
		}

		// Create source location for spdlog
		spdlog::source_loc source_loc{loc.file_name, loc.line, loc.function_name};

		// Log the message
		logger_instance->log(source_loc, to_spdlog_level(level), message);
	}

	template<typename... Args>
	static void
	log(LogLevel level, const SourceLocation& loc, std::string_view fmt_str, Args&&... args) {
		if (level < current_level)
			return;

		// Ensure logger is initialized
		if (!logger_instance) {
			initialize();
		}

		try {
			// Create source location for spdlog
			spdlog::source_loc source_loc{loc.file_name, loc.line, loc.function_name};

			// Use spdlog's direct logging without fmt constexpr issues
			logger_instance->log(source_loc, to_spdlog_level(level), std::string(fmt_str));
		} catch (const std::exception& e) {
			// Fallback to simple logging if formatting fails
			std::string error_msg = "FORMAT_ERROR: " + std::string(fmt_str) + " (" + e.what() + ")";
			log(LogLevel::ERROR, loc, std::string_view(error_msg));
		}
	}

	static void set_level(LogLevel level) {
		current_level = level;
		if (logger_instance) {
			logger_instance->set_level(to_spdlog_level(level));
		}
	}

	// Convenience methods for direct spdlog usage
	static void trace(const std::string& message) {
		if (logger_instance) {
			logger_instance->trace(message);
		}
	}

	static void debug(const std::string& message) {
		if (logger_instance) {
			logger_instance->debug(message);
		}
	}

	static void info(const std::string& message) {
		if (logger_instance) {
			logger_instance->info(message);
		}
	}

	static void warn(const std::string& message) {
		if (logger_instance) {
			logger_instance->warn(message);
		}
	}

	static void error(const std::string& message) {
		if (logger_instance) {
			logger_instance->error(message);
		}
	}

	static void critical(const std::string& message) {
		if (logger_instance) {
			logger_instance->critical(message);
		}
	}

	// Template versions for formatted messages
	template<typename... Args>
	static void trace(const std::string& fmt, Args&&... args) {
		if (logger_instance) {
			logger_instance->trace(fmt, std::forward<Args>(args)...);
		}
	}

	template<typename... Args>
	static void debug(const std::string& fmt, Args&&... args) {
		if (logger_instance) {
			logger_instance->debug(fmt, std::forward<Args>(args)...);
		}
	}

	template<typename... Args>
	static void info(const std::string& fmt, Args&&... args) {
		if (logger_instance) {
			logger_instance->info(fmt, std::forward<Args>(args)...);
		}
	}

	template<typename... Args>
	static void warn(const std::string& fmt, Args&&... args) {
		if (logger_instance) {
			logger_instance->warn(fmt, std::forward<Args>(args)...);
		}
	}

	template<typename... Args>
	static void error(const std::string& fmt, Args&&... args) {
		if (logger_instance) {
			logger_instance->error(fmt, std::forward<Args>(args)...);
		}
	}

	template<typename... Args>
	static void critical(const std::string& fmt, Args&&... args) {
		if (logger_instance) {
			logger_instance->critical(fmt, std::forward<Args>(args)...);
		}
	}

  private:
	static const char* get_level_name(LogLevel level) {
		switch (level) {
		case LogLevel::TRACE:
			return "TRACE";
		case LogLevel::DEBUG:
			return "DEBUG";
		case LogLevel::INFO:
			return "INFO";
		case LogLevel::WARN:
			return "WARN";
		case LogLevel::ERROR:
			return "ERROR";
		case LogLevel::CRITICAL:
			return "CRITICAL";
		default:
			return "UNKNOWN";
		}
	}
};

inline LogLevel Logger::current_level = LogLevel::INFO;
inline std::shared_ptr<spdlog::logger> Logger::logger_instance = nullptr;

} // namespace ARBD

// Debug.h compatibility layer
#ifdef DEBUGMSG
#define Debug(x) (x)
#define DebugMsg(level, ...)                                           \
	ARBD::Logger::log(static_cast<ARBD::LogLevel>(std::min(level, 5)), \
					  ARBD::SourceLocation(),                          \
					  __VA_ARGS__)
#define DebugMessage(level, message)                                   \
	ARBD::Logger::log(static_cast<ARBD::LogLevel>(std::min(level, 5)), \
					  ARBD::SourceLocation(),                          \
					  std::string_view(message))
#else
#define Debug(x) static_cast<void>(0)
#define DebugMsg(level, ...) static_cast<void>(0)
#define DebugMessage(level, message) static_cast<void>(0)
#endif

// Host-side logging macros - only active when NOT compiling device code
#ifndef __CUDA_ARCH__
#define LOGTRACE(...) ARBD::Logger::log(ARBD::LogLevel::TRACE, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGDEBUG(...) ARBD::Logger::log(ARBD::LogLevel::DEBUG, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGINFO(...) ARBD::Logger::log(ARBD::LogLevel::INFO, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGWARN(...) ARBD::Logger::log(ARBD::LogLevel::WARN, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGERROR(...) ARBD::Logger::log(ARBD::LogLevel::ERROR, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGCRITICAL(...) \
	ARBD::Logger::log(ARBD::LogLevel::CRITICAL, ARBD::SourceLocation(), __VA_ARGS__)
#else
// Device-side: use simplified printf-based logging with proper format handling
namespace ARBD {
namespace DeviceLogger {
// Device-side printf wrapper that handles C-style formatting
template<typename... Args>
__device__ __forceinline__ void device_printf(const char* level, const char* fmt, Args... args) {
	printf("[%s]: ", level);
	printf(fmt, args...);
	printf("\n");
}

// Overload for string literals without arguments
__device__ __forceinline__ void device_printf(const char* level, const char* msg) {
	printf("[%s]: %s\n", level, msg);
}

// Helper to convert modern format strings to printf-style (simplified)
// Note: This is a basic fallback - use printf-style formats in device code
__device__ __forceinline__ void device_log_simple(const char* level, const char* msg) {
	printf("[%s]: %s\n", level, msg);
}
} // namespace DeviceLogger
} // namespace ARBD

// Device-side logging macros
// Note: Use printf-style format strings (%d, %s, %f) instead of {} in device code
#define LOGTRACE(...) ARBD::DeviceLogger::device_printf("TRACE", __VA_ARGS__)
#define LOGDEBUG(...) ARBD::DeviceLogger::device_printf("DEBUG", __VA_ARGS__)
#define LOGINFO(...) ARBD::DeviceLogger::device_printf("INFO", __VA_ARGS__)
#define LOGWARN(...) ARBD::DeviceLogger::device_printf("WARN", __VA_ARGS__)
#define LOGERROR(...) ARBD::DeviceLogger::device_printf("ERROR", __VA_ARGS__)
#define LOGCRITICAL(...) ARBD::DeviceLogger::device_printf("CRITICAL", __VA_ARGS__)
#endif
