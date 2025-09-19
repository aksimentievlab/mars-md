#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace ARBD {
/**
 * @brief RAII wrapper for FILE* handles
 *
 * This class provides a safe, RAII-style wrapper around C-style FILE* handles.
 * It automatically manages file opening and closing, preventing resource leaks.
 *
 * Features:
 * - Automatic file closing on destruction
 * - Move semantics support
 * - Copy prevention
 * - Exception safety
 *
 * @example Basic Usage:
 * ```cpp
 * // Open a file for reading
 * FileHandle input_file("data.txt", "r");
 *
 * // Open a file for writing
 * FileHandle output_file("output.txt", "w");
 *
 * // Use the underlying FILE* handle
 * FILE* fp = input_file.get();
 * fscanf(fp, "%d", &value);
 * ```
 *
 * @example Error Handling:
 * ```cpp
 * try {
 *     FileHandle file("nonexistent.txt", "r");
 * } catch (const std::runtime_error& e) {
 *     // Handle file open failure
 *     std::cerr << "Failed to open file: " << e.what() << std::endl;
 * }
 * ```
 *
 * @note The class prevents copying to avoid accidental double-closing.
 *       Use move semantics when transferring ownership.
 */

class FileHandle {
	FILE* m_file = nullptr;

  public:
	FileHandle(const char* filename, const char* mode) : m_file(std::fopen(filename, mode)) {
		if (!m_file) {
			// For write modes, try to create parent directories
			if (std::strchr(mode, 'w') || std::strchr(mode, 'a')) {
				std::string path(filename);
				size_t last_slash = path.find_last_of("/\\");
				if (last_slash != std::string::npos) {
					std::string dir_path = path.substr(0, last_slash);
					try {
						if (std::filesystem::create_directories(dir_path)) {
							// Try opening again after creating directories
							m_file = std::fopen(filename, mode);
						}
					} catch (const std::filesystem::filesystem_error&) {
						// If directory creation fails, don't retry - let the original error be
						// thrown
					}
				}
			}

			if (!m_file) {
				throw ARBD::Exception(
					ARBD::ExceptionType::FileOpenError,
					std::string("FileHandle: Failed to open file '") + filename + "' with mode '" +
						mode + "'.",
					ARBD::SourceLocation(__builtin_FILE(), __builtin_LINE(), __builtin_FUNCTION()));
			}
		}
	}
	~FileHandle() {
		if (m_file) {
			std::fclose(m_file);
			m_file = nullptr; // Good practice to nullify after closing
		}
	}
	// Delete copy constructor/assignment
	FileHandle(const FileHandle&) = delete;
	FileHandle& operator=(const FileHandle&) = delete;
	// Allow move
	FileHandle(FileHandle&& other) noexcept : m_file(other.m_file) {
		other.m_file = nullptr;
	}
	FileHandle& operator=(FileHandle&& other) noexcept {
		if (this != &other) {
			if (m_file)
				std::fclose(m_file);
			m_file = other.m_file;
			other.m_file = nullptr;
		}
		return *this;
	}

	FILE* get() const {
		return m_file;
	}
	// operator FILE*() const { return m_file; } // If implicit conversion is
	// desired
};
// Usage: FileHandle my_file("data.txt", "r"); // Automatically closes
} // namespace ARBD
