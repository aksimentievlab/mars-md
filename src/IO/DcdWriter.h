// Modified dcd reader from NAMD.
// Author: Jeff Comer <jcomer2@illinois.edu>
// Refactored for the arbd2/cpp20 branch with on 2025
// Author: Pin-Yi Li <pinyili2@illinois.edu> with Claude 4.0 sonnet

/**
***  Copyright (c) 1995-2025 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/
#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/Types.h"

#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// DCD file format constants
constexpr off_t NFILE_POS = 8;
constexpr off_t NPRIV_POS = 12;
constexpr off_t NSAVC_POS = 16;
constexpr off_t NSTEP_POS = 20;

#ifndef O_LARGEFILE
#define O_LARGEFILE 0x0
#endif

namespace ARBD {

enum class DcdError : int {
	Success = 0,
	FileDoesNotExist = -2,
	OpenFailed = -3,
	BadRead = -4,
	PrematureEOF = -5,
	BadFormat = -6,
	FileExists = -7,
	BadMalloc = -8
};

/**
 * @brief Modern C++20 DCD file writer
 *
 * Refactored for the arbd2/cpp20 branch with:
 * - Modern C++20 features
 * - ARBD2 exception handling
 * - Math/Types integration
 * - Resource management
 * - Backend compatibility
 * @section Usage
 *
 * Example: Writing a DCD trajectory file
 *
 * @example
 * #include "IO/DcdWriter.h"
 * #include "Math/Vector3.h"
 * #include <vector>
 *
 * using namespace ARBD;
 *
 * int main() {
 *     // Number of atoms and frames
 *     size_t numAtoms = 1000;
 *     size_t numFrames = 10;
 *
 *     // Create a DcdWriter (throws on failure)
 *     DcdWriter writer("trajectory.dcd");
 *
 *     // Prepare coordinates for each frame
 *     std::vector<Vector3> coords(numAtoms);
 *     for (size_t frame = 0; frame < numFrames; ++frame) {
 *         // Fill coords with your simulation data here
 *         // ...
 *         writer.writeFrame(coords);
 *     }
 *
 *     // File is closed automatically on destruction
 *     return 0;
 * }
 * @endcode
 *
 * @note Coordinates must be in Angstroms, as Vector3.
 */

class DcdWriter {
  public:
	/**
	 * @brief Constructor - opens DCD file for writing
	 * @param fileName Path to DCD file to create
	 * @param resource Backend resource for memory operations (optional)
	 * @throws ARBD::Exception on file open failure
	 */
	explicit DcdWriter(std::string_view fileName) : fileName_(fileName), fd_(-1) {

		fd_ = openDcd(fileName_);

		if (fd_ < 0) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"DcdWriter: Failed to open DCD file {}",
							fileName_);
		}

		LOGINFO("DcdWriter: Successfully opened DCD file '{}'", fileName_);
	}

	/**
	 * @brief Destructor - ensures file is properly closed
	 */
	~DcdWriter() noexcept {
		try {
			closeDcd();
		} catch (const std::exception& e) {
			LOGERROR("DcdWriter destructor error: {}", e.what());
		}
	}

	// Non-copyable but movable
	DcdWriter(const DcdWriter&) = delete;
	DcdWriter& operator=(const DcdWriter&) = delete;

	DcdWriter(DcdWriter&& other) noexcept : fileName_(std::move(other.fileName_)), fd_(other.fd_) {
		other.fd_ = -1;
	}

	DcdWriter& operator=(DcdWriter&& other) noexcept {
		if (this != &other) {
			closeDcd();
			fileName_ = std::move(other.fileName_);
			fd_ = other.fd_;
			other.fd_ = -1;
		}
		return *this;
	}

	/**
	 * @brief Write DCD header information
	 * @param N Number of atoms
	 * @param NFILE Number of coordinate sets (frames)
	 * @param NPRIV Starting timestep (typically 1, not 0)
	 * @param NSAVC Timesteps between saves
	 * @param NSTEP Total number of timesteps
	 * @param DELTA Timestep length in time units
	 * @param with_unitcell Whether unit cell information is included
	 * @return 0 on success
	 * @throws ARBD::Exception on write failure
	 */
	int writeHeader(int N,
					int NFILE = 1,
					int NPRIV = 1,
					int NSAVC = 1,
					int NSTEP = 0,
					float DELTA = 1.0f,
					bool with_unitcell = false) {

		try {
			return writeHeaderImpl(N, NFILE, NPRIV, NSAVC, NSTEP, DELTA, with_unitcell);
		} catch (const std::exception& e) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to write DCD header: %s",
							e.what());
		}
	}

	/**
	 * @brief Write a timestep of coordinates
	 * @param positions Vector of atom positions (ARBD::Vector3)
	 * @param unitcell Unit cell in CHARMM on-disk order (optional):
	 *        [A, cos(gamma), B, cos(beta), cos(alpha), C].
	 *        This array is written to the file verbatim, so it must use the
	 *        CHARMM ordering that VMD/MDAnalysis expect - not [a,b,c,alpha,beta,gamma].
	 *        Must be supplied whenever the header was written with
	 *        with_unitcell=true, otherwise every frame is missing its extra
	 *        block and the file misparses.
	 * @return 0 on success
	 * @throws ARBD::Exception on write failure
	 */
	int writeStep(const std::vector<Vector3>& positions, const std::vector<double>& unitcell = {}) {

		if (positions.empty()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Cannot write empty position vector");
		}

		const double* cell_ptr = unitcell.empty() ? nullptr : unitcell.data();

		try {
			// Use your Vector3 class directly - pass the vector for efficient processing
			return writeStepFromVector3(positions, cell_ptr);
		} catch (const std::exception& e) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to write DCD step: %s",
							e.what());
		}
	}

	/**
	 * @brief Write a timestep from DeviceBuffer (for GPU data)
	 * @param positions_buffer DeviceBuffer containing Vector3 positions
	 * @param unitcell Unit cell parameters (optional)
	 * @return 0 on success
	 */
	int writeStep(const DeviceBuffer<Vector3>& positions_buffer,
				  const std::vector<double>& unitcell = {}) {

		// Copy data from device to host
		std::vector<Vector3> host_positions;
		positions_buffer.copy_to_host(host_positions);

		return writeStep(host_positions, unitcell);
	}

	/**
	 * @brief Get the current file descriptor (for advanced usage)
	 * @return File descriptor or -1 if closed
	 */
	[[nodiscard]] int getFileDescriptor() const noexcept {
		return fd_;
	}

	/**
	 * @brief Check if the file is open and ready for writing
	 * @return true if file is open
	 */
	[[nodiscard]] bool isOpen() const noexcept {
		return fd_ >= 0;
	}

	/**
	 * @brief Get the filename
	 * @return Current filename
	 */
	[[nodiscard]] const std::string& getFileName() const noexcept {
		return fileName_;
	}

  private:
	std::string fileName_;
	int fd_;
	std::vector<float> coord_buffer_; // reused per-frame packing buffer for one X/Y/Z block

	/**
	 * @brief Open DCD file for writing with backup handling
	 * @param filename Path to DCD file
	 * @return File descriptor on success, negative DcdError code on failure
	 */
	int openDcd(std::string_view filename) {
		struct stat sbuf;

		// If file exists, rename it to .BAK
		if (stat(filename.data(), &sbuf) == 0) {
			std::string backup_name = std::string(filename) + ".BAK";

			if (rename(filename.data(), backup_name.c_str()) != 0) {
				LOGERROR("Failed to backup existing file {} to {}", filename, backup_name);
				return static_cast<int>(DcdError::OpenFailed);
			}

			LOGINFO("Backed up existing file {} to {}", filename, backup_name);
		}

		int fd = open(filename.data(),
					  O_RDWR | O_CREAT | O_EXCL | O_LARGEFILE,
					  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

		if (fd < 0) {
			LOGERROR("Failed to open DCD file {}: {}", filename, strerror(errno));
			return static_cast<int>(DcdError::OpenFailed);
		}

		return fd;
	}

	/**
	 * @brief Close DCD file
	 */
	void closeDcd() noexcept {
		if (fd_ >= 0) {
			close(fd_);
			fd_ = -1;
			LOGINFO("DcdWriter: Closed DCD file '{}'", fileName_);
		}
	}

	/**
	 * @brief Pad string to specified length with spaces
	 * @param s String to pad (modified in place)
	 * @param len Target length
	 */
	static void padString(char* s, int len) noexcept {
		int curlen = static_cast<int>(strlen(s));

		if (curlen > len) {
			s[len] = '\0';
			return;
		}

		for (int i = curlen; i < len; i++) {
			s[i] = ' ';
		}
		s[len] = '\0';
	}

	/**
	 * @brief Convert DcdError to human-readable string
	 * @param error DcdError code
	 * @return Error description
	 */
	static void getErrorString(DcdError error) {
		switch (error) {
		case DcdError::Success:
			LOGINFO("Success");
			break;
		case DcdError::FileDoesNotExist:
			LOGERROR("DCD file does not exist");
			break;
		case DcdError::OpenFailed:
			LOGERROR("Open of DCD file failed");
			break;
		case DcdError::BadRead:
			LOGERROR("Read call on DCD file failed");
			break;
		case DcdError::PrematureEOF:
			LOGERROR("Premature EOF found in DCD file");
			break;
		case DcdError::BadFormat:
			LOGERROR("Format of DCD file is wrong");
			break;
		case DcdError::FileExists:
			LOGERROR("Output file already exists");
			break;
		case DcdError::BadMalloc:
			LOGERROR("Memory allocation failed");
			break;
		default:
			LOGERROR("Unknown DCD error code: {}", static_cast<int>(error));
			break;
		}
	}

	/**
	 * @brief Implementation of header writing (FORTRAN binary format)
	 */
	int writeHeaderImpl(int N,
						int NFILE,
						int NPRIV,
						int NSAVC,
						int NSTEP,
						float DELTA,
						bool with_unitcell) {

		constexpr int HEADER_SIZE = 84;
		constexpr int TITLE_BLOCK_SIZE = 164;
		constexpr int TITLE_LINE_SIZE = 80;
		constexpr int CHARMM_VERSION = 24;

		// Write main header block (84 bytes)
		int out_integer = HEADER_SIZE;
		safeWrite(&out_integer, sizeof(int));

		char title_string[200];
		strcpy(title_string, "CORD");
		safeWrite(title_string, 4);

		// File count (set to 0, will be updated in writeStep)
		out_integer = 0;
		safeWrite(&out_integer, sizeof(int));

		// Starting timestep
		out_integer = NPRIV;
		safeWrite(&out_integer, sizeof(int));

		// Save frequency
		out_integer = NSAVC;
		safeWrite(&out_integer, sizeof(int));

		// Total timesteps (set to starting - frequency, will be updated)
		out_integer = NPRIV - NSAVC;
		safeWrite(&out_integer, sizeof(int));

		// Write zeros for unused fields
		out_integer = 0;
		for (int i = 0; i < 5; i++) {
			safeWrite(&out_integer, sizeof(int));
		}

		// Timestep size
		float out_float = DELTA;
		safeWrite(&out_float, sizeof(float));

		// Unit cell flag
		out_integer = with_unitcell ? 1 : 0;
		safeWrite(&out_integer, sizeof(int));

		// More unused fields
		out_integer = 0;
		for (int i = 0; i < 8; i++) {
			safeWrite(&out_integer, sizeof(int));
		}

		// CHARMM version
		out_integer = CHARMM_VERSION;
		safeWrite(&out_integer, sizeof(int));

		// Close header block
		out_integer = HEADER_SIZE;
		safeWrite(&out_integer, sizeof(int));

		// Write title block
		out_integer = TITLE_BLOCK_SIZE;
		safeWrite(&out_integer, sizeof(int));

		out_integer = 2; // Number of title lines
		safeWrite(&out_integer, sizeof(int));

		// First title line
		snprintf(title_string,
				 sizeof(title_string),
				 "REMARKS FILENAME=%s CREATED BY ARBD2",
				 fileName_.c_str());
		padString(title_string, TITLE_LINE_SIZE);
		safeWrite(title_string, TITLE_LINE_SIZE);

		// Second title line with timestamp
		time_t cur_time = time(nullptr);
		struct tm* tmbuf = localtime(&cur_time);
		char time_str[11];
		strftime(time_str, sizeof(time_str), "%m/%d/%y", tmbuf);

		snprintf(title_string,
				 sizeof(title_string),
				 "REMARKS DATE: %s CREATED BY USER: ARBD2",
				 time_str);
		padString(title_string, TITLE_LINE_SIZE);
		safeWrite(title_string, TITLE_LINE_SIZE);

		// Close title block
		out_integer = TITLE_BLOCK_SIZE;
		safeWrite(&out_integer, sizeof(int));

		// Write atom count
		out_integer = 4;
		safeWrite(&out_integer, sizeof(int));
		out_integer = N;
		safeWrite(&out_integer, sizeof(int));
		out_integer = 4;
		safeWrite(&out_integer, sizeof(int));

		LOGINFO("DcdWriter: Wrote header for {} atoms", N);
		return 0;
	}

	/**
	 * @brief Implementation of step writing from Vector3 array (most efficient)
	 * @param positions Vector of Vector3 positions
	 * @param cell Unit cell parameters (optional)
	 */
	int writeStepFromVector3(const std::vector<Vector3>& positions, const double* cell) {
		const int N = static_cast<int>(positions.size());

		// Write unit cell if provided
		if (cell) {
			int out_integer = 6 * sizeof(double);
			safeWrite(&out_integer, sizeof(int));
			safeWrite(cell, out_integer);
			safeWrite(&out_integer, sizeof(int));
		}

		// Write coordinates using the DCD block layout (X block, Y block, Z
		// block), one contiguous write() per block. Extracting each component
		// into a packed buffer first turns 3*N write() syscalls per frame into
		// 3 - for a 300k-atom system that is ~900k syscalls/frame eliminated,
		// which otherwise dominates wall-clock (the coordinate loops here were
		// the single largest cost in large-system runs).
		int coord_size = N * sizeof(float);
		const Vector3* pos_data = positions.data();

		coord_buffer_.resize(N);

		for (int i = 0; i < N; ++i) {
			coord_buffer_[i] = pos_data[i].x;
		}
		safeWrite(&coord_size, sizeof(int));
		safeWrite(coord_buffer_.data(), coord_size);
		safeWrite(&coord_size, sizeof(int));

		for (int i = 0; i < N; ++i) {
			coord_buffer_[i] = pos_data[i].y;
		}
		safeWrite(&coord_size, sizeof(int));
		safeWrite(coord_buffer_.data(), coord_size);
		safeWrite(&coord_size, sizeof(int));

		for (int i = 0; i < N; ++i) {
			coord_buffer_[i] = pos_data[i].z;
		}
		safeWrite(&coord_size, sizeof(int));
		safeWrite(coord_buffer_.data(), coord_size);
		safeWrite(&coord_size, sizeof(int));

		// Update header counters
		updateHeader();

		LOGTRACE("DcdWriter: Wrote step with {} atoms using Vector3 layout", N);
		return 0;
	}

	/**
	 * @brief Update header counters after writing a step
	 */
	void updateHeader() {
		int NSAVC, NSTEP, NFILE;

		// Read current values
		lseek(fd_, NSAVC_POS, SEEK_SET);
		read(fd_, &NSAVC, sizeof(int));
		lseek(fd_, NSTEP_POS, SEEK_SET);
		read(fd_, &NSTEP, sizeof(int));
		lseek(fd_, NFILE_POS, SEEK_SET);
		read(fd_, &NFILE, sizeof(int));

		// Update values
		NSTEP += NSAVC;
		NFILE += 1;

		// Write updated values
		lseek(fd_, NSTEP_POS, SEEK_SET);
		safeWrite(&NSTEP, sizeof(int));
		lseek(fd_, NFILE_POS, SEEK_SET);
		safeWrite(&NFILE, sizeof(int));

		// Return to end of file
		lseek(fd_, 0, SEEK_END);
	}

	/**
	 * @brief Safe write with error checking
	 * @param data Data to write
	 * @param size Size in bytes
	 * @throws std::runtime_error on write failure
	 */
	void safeWrite(const void* data, size_t size) {
		ssize_t written = write(fd_, data, size);
		if (written != static_cast<ssize_t>(size)) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Write failed: expected {} bytes, wrote {} bytes",
							size,
							written);
		}
	}
};

} // namespace ARBD
