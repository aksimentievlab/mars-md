#pragma once

#include "MARSException.h"
#include "MARSLogger.h"
#include "Types/Types.h"
#include <cstdio>
#include <string>
#include <vector>

namespace MARS {

/**
 * @brief DCD file reader with stride support for MARS2
 *
 * This class provides functionality to read DCD trajectory files
 * with configurable stride (frame skipping) for post-processing
 * and analysis workflows.
 */
class DcdReader {
  public:
	/**
	 * @brief Constructor for DCD reader
	 * @param filename Path to DCD file
	 * @param stride Stride for reading frames (1 = every frame, 2 = every other frame, etc.)
	 */
	DcdReader(const std::string& filename, int stride = 1)
		: filename_(filename), stride_(stride), current_frame_(0), file_(nullptr) {

		if (stride <= 0) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid stride value: %d (must be > 0)",
							stride);
		}

		file_ = fopen(filename.c_str(), "rb");
		if (!file_) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to open DCD file: %s",
							filename.c_str());
		}

		// Read DCD header
		readHeader();
		LOGINFO("DcdReader: Opened DCD file '%s' with stride %d", filename.c_str(), stride_);
	}

	/**
	 * @brief Destructor
	 */
	~DcdReader() {
		if (file_) {
			fclose(file_);
		}
	}

	// Non-copyable but movable
	DcdReader(const DcdReader&) = delete;
	DcdReader& operator=(const DcdReader&) = delete;

	DcdReader(DcdReader&& other) noexcept
		: filename_(std::move(other.filename_)), stride_(other.stride_),
		  current_frame_(other.current_frame_), file_(other.file_), num_atoms_(other.num_atoms_),
		  num_frames_(other.num_frames_) {
		other.file_ = nullptr;
	}

	DcdReader& operator=(DcdReader&& other) noexcept {
		if (this != &other) {
			if (file_)
				fclose(file_);
			filename_ = std::move(other.filename_);
			stride_ = other.stride_;
			current_frame_ = other.current_frame_;
			file_ = other.file_;
			num_atoms_ = other.num_atoms_;
			num_frames_ = other.num_frames_;
			other.file_ = nullptr;
		}
		return *this;
	}

	/**
	 * @brief Read the next frame according to stride
	 * @return Vector3 array of coordinates, or nullptr if EOF
	 */
	Vector3* read_step() {
		if (!file_)
			return nullptr;

		// Skip frames according to stride
		for (int i = 0; i < stride_ - 1; ++i) {
			if (!skip_step()) {
				return nullptr; // EOF
			}
		}

		// Read the actual frame
		return read_frame();
	}

	/**
	 * @brief Skip a single frame
	 * @return true if frame was skipped, false if EOF
	 */
	bool skip_step() {
		if (!file_)
			return false;

		// Read frame header
		int frame_size;
		if (fread(&frame_size, sizeof(int), 1, file_) != 1) {
			return false; // EOF
		}

		// Skip the frame data
		if (fseek(file_, frame_size, SEEK_CUR) != 0) {
			return false; // Error or EOF
		}

		// Read frame footer
		int footer_size;
		if (fread(&footer_size, sizeof(int), 1, file_) != 1) {
			return false; // EOF
		}

		if (frame_size != footer_size) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"DCD frame size mismatch: header=%d, footer=%d",
							frame_size,
							footer_size);
		}

		current_frame_++;
		return true;
	}

	/**
	 * @brief Get number of atoms in the trajectory
	 * @return Number of atoms
	 */
	int get_num_atoms() const {
		return num_atoms_;
	}

	/**
	 * @brief Get total number of frames in the trajectory
	 * @return Number of frames
	 */
	int get_num_frames() const {
		return num_frames_;
	}

	/**
	 * @brief Get current frame number
	 * @return Current frame number
	 */
	int get_current_frame() const {
		return current_frame_;
	}

	/**
	 * @brief Check if we're at end of file
	 * @return true if at EOF
	 */
	bool is_eof() const {
		if (!file_)
			return true;
		return feof(file_) != 0;
	}

  private:
	std::string filename_;
	int stride_;
	int current_frame_;
	FILE* file_;
	int num_atoms_;
	int num_frames_;

	/**
	 * @brief Read DCD file header
	 */
	void readHeader() {
		// Read magic number
		int magic;
		if (fread(&magic, sizeof(int), 1, file_) != 1) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to read DCD magic number");
		}

		// Check if we need to swap bytes (DCD files can be big/little endian)
		bool swap_bytes = false;
		if (magic != 84) { // 84 is the magic number for DCD files
			swap_bytes = true;
			// TODO: Implement byte swapping if needed
		}

		// Read number of frames
		if (fread(&num_frames_, sizeof(int), 1, file_) != 1) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to read number of frames");
		}

		// Read timestep info (skip for now)
		float timestep;
		if (fread(&timestep, sizeof(float), 1, file_) != 1) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to read timestep");
		}

		// Read number of atoms
		if (fread(&num_atoms_, sizeof(int), 1, file_) != 1) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to read number of atoms");
		}

		LOGINFO("DcdReader: DCD file has %d atoms, %d frames", num_atoms_, num_frames_);
	}

	/**
	 * @brief Read a single frame of coordinates
	 * @return Vector3 array of coordinates, or nullptr if error
	 */
	Vector3* read_frame() {
		if (!file_)
			return nullptr;

		// Read frame header
		int frame_size;
		if (fread(&frame_size, sizeof(int), 1, file_) != 1) {
			return nullptr; // EOF
		}

		// Allocate coordinate array
		Vector3* coords = new Vector3[num_atoms_];

		// Read X coordinates
		for (int i = 0; i < num_atoms_; ++i) {
			if (fread(&coords[i].x, sizeof(float), 1, file_) != 1) {
				delete[] coords;
				return nullptr;
			}
		}

		// Read Y coordinates
		for (int i = 0; i < num_atoms_; ++i) {
			if (fread(&coords[i].y, sizeof(float), 1, file_) != 1) {
				delete[] coords;
				return nullptr;
			}
		}

		// Read Z coordinates
		for (int i = 0; i < num_atoms_; ++i) {
			if (fread(&coords[i].z, sizeof(float), 1, file_) != 1) {
				delete[] coords;
				return nullptr;
			}
		}

		// Read frame footer
		int footer_size;
		if (fread(&footer_size, sizeof(int), 1, file_) != 1) {
			delete[] coords;
			return nullptr;
		}

		if (frame_size != footer_size) {
			delete[] coords;
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"DCD frame size mismatch: header=%d, footer=%d",
							frame_size,
							footer_size);
		}

		current_frame_++;
		return coords;
	}
};

} // namespace MARS
