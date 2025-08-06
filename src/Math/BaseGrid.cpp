/*********************************************************************
 * @file  BaseGrid.cpp
 *
 * @brief Implementation of BaseGrid I/O operations and host-specific methods
 *        Following the pattern of Vector3.h (keeping device code in header)
 *
 * @author Original: Jeff Comer <jcomer2@illinois.edu>
 * @author V2 Port: Pin-Yi Li with Claude 4.0 Sonnet
 *********************************************************************/
#if !defined(__METAL_VERSION__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
#include "BaseGrid.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/FileHandle.h"
#include <array>
#include <cctype>
#include <cstdio>
#include <string>
#include <string_view>
#include <sstream>

namespace ARBD {

namespace {
constexpr size_t BUFFER_SIZE = 512;

/**
 * @brief Check if character represents start of integer
 */
bool is_int_start(char c) {
	return std::isdigit(c) || c == '-' || c == '+';
}

/**
 * @brief Find first space in string
 */
size_t find_first_space(const char* str, size_t max_len) {
	for (size_t i = 0; i < max_len && str[i] != '\0'; ++i) {
		if (std::isspace(str[i]))
			return i;
	}
	return max_len;
}

/**
 * @brief Parse OpenDX header line for grid counts
 */
bool parse_grid_counts(const std::string& line, size_t& nx, size_t& ny, size_t& nz) {
	if (line.find("object 1 class gridpositions counts") == std::string::npos) {
		return false;
	}

	std::istringstream iss(line);
	std::string word;
	// Skip "object 1 class gridpositions counts"
	while (iss >> word && word != "counts") {
	}

	if (!(iss >> nx >> ny >> nz)) {
		return false;
	}

	return nx > 0 && ny > 0 && nz > 0;
}
} // namespace

/*===================*\
|  I/O OPERATIONS     |
\*===================*/

template<typename T>
void BaseGrid<T>::write(std::string_view filename) const {
	write(filename, "");
}

template<typename T>
void BaseGrid<T>::write(std::string_view filename, std::string_view comments) const {
	try {
		FileHandle file(filename.data(), "w");
		write_dx_format(file, comments);
		LOGINFO("BaseGrid::write - Successfully wrote {} values to '{}'", values_.size(), filename);
	} catch (const Exception& e) {
		LOGERROR("BaseGrid::write - Failed to write to '{}': {}", filename, e.what());
		throw;
	}
}

template<typename T>
void BaseGrid<T>::write_dx_format(const FileHandle& file, std::string_view comments) const {
	FILE* fp = file.get();

	// Write header
	std::fprintf(fp, "# %.*s\n", static_cast<int>(comments.size()), comments.data());
	std::fprintf(fp, "object 1 class gridpositions counts %zu %zu %zu\n", nx(), ny(), nz());

	// Write origin
	const auto& origin = config_.origin;
	std::fprintf(fp,
				 "origin %.12g %.12g %.12g\n",
				 static_cast<double>(origin.x),
				 static_cast<double>(origin.y),
				 static_cast<double>(origin.z));

	// Write basis vectors (delta lines)
	const auto& basis = config_.basis;
	std::fprintf(fp,
				 "delta %.12g %.12g %.12g\n",
				 static_cast<double>(basis.ex().x),
				 static_cast<double>(basis.ex().y),
				 static_cast<double>(basis.ex().z));
	std::fprintf(fp,
				 "delta %.12g %.12g %.12g\n",
				 static_cast<double>(basis.ey().x),
				 static_cast<double>(basis.ey().y),
				 static_cast<double>(basis.ey().z));
	std::fprintf(fp,
				 "delta %.12g %.12g %.12g\n",
				 static_cast<double>(basis.ez().x),
				 static_cast<double>(basis.ez().y),
				 static_cast<double>(basis.ez().z));

	// Write grid connections and data header
	std::fprintf(fp, "object 2 class gridconnections counts %zu %zu %zu\n", nx(), ny(), nz());
	std::fprintf(fp, "object 3 class array type float rank 0 items %zu data follows\n", size());

	// Write data in groups of 3 for readability
	const size_t total_values = values_.size();
	const size_t complete_triplets = total_values / 3;
	const size_t remainder = total_values % 3;

	for (size_t i = 0; i < complete_triplets; ++i) {
		const size_t base_idx = i * 3;
		std::fprintf(fp,
					 "%.12g %.12g %.12g\n",
					 static_cast<double>(values_[base_idx]),
					 static_cast<double>(values_[base_idx + 1]),
					 static_cast<double>(values_[base_idx + 2]));
	}

	// Handle remainder
	if (remainder == 1) {
		std::fprintf(fp, "%.12g\n", static_cast<double>(values_[total_values - 1]));
	} else if (remainder == 2) {
		std::fprintf(fp,
					 "%.12g %.12g\n",
					 static_cast<double>(values_[total_values - 2]),
					 static_cast<double>(values_[total_values - 1]));
	}
}

template<typename T>
void BaseGrid<T>::write_data_format(const FileHandle& file) const {
	FILE* fp = file.get();

	// Write grid dimensions
	std::fprintf(fp, "%zu\n%zu\n%zu\n", nx(), ny(), nz());

	// Write origin
	const auto& origin = config_.origin;
	std::fprintf(fp,
				 "%.12g\n%.12g\n%.12g\n",
				 static_cast<double>(origin.x),
				 static_cast<double>(origin.y),
				 static_cast<double>(origin.z));

	// Write basis matrix (column by column)
	const auto& basis = config_.basis;
	const auto ex = basis.ex();
	const auto ey = basis.ey();
	const auto ez = basis.ez();

	std::fprintf(fp,
				 "%.12g\n%.12g\n%.12g\n",
				 static_cast<double>(ex.x),
				 static_cast<double>(ex.y),
				 static_cast<double>(ex.z));
	std::fprintf(fp,
				 "%.12g\n%.12g\n%.12g\n",
				 static_cast<double>(ey.x),
				 static_cast<double>(ey.y),
				 static_cast<double>(ey.z));
	std::fprintf(fp,
				 "%.12g\n%.12g\n%.12g\n",
				 static_cast<double>(ez.x),
				 static_cast<double>(ez.y),
				 static_cast<double>(ez.z));

	// Write all values
	for (const auto& value : values_) {
		std::fprintf(fp, "%.12g\n", static_cast<double>(value));
	}
}

template<typename T>
BaseGrid<T> BaseGrid<T>::read_from_file(std::string_view filename) {
	try {
		FileHandle file(filename.data(), "r");
		BaseGrid<T> grid;
		grid.read_dx_format(file);

		LOGINFO("BaseGrid::read_from_file - Successfully read {} values from '{}'",
				grid.size(),
				filename);
		return grid;
	} catch (const Exception& e) {
		LOGERROR("BaseGrid::read_from_file - Failed to read from '{}': {}", filename, e.what());
		throw;
	}
}

template<typename T>
void BaseGrid<T>::read_dx_format(const FileHandle& file) {
	FILE* fp = file.get();

	// Initialize parsing state
	size_t nx = 0, ny = 0, nz = 0;
	Vector3 origin{0, 0, 0};
	std::array<Vector3, 3> basis_vectors{};
	size_t delta_count = 0;
	size_t values_read = 0;
	bool dimensions_parsed = false;

	std::array<char, BUFFER_SIZE> line_buffer{};

	while (std::fgets(line_buffer.data(), BUFFER_SIZE, fp)) {
		std::string line(line_buffer.data());

		// Skip comments and empty lines
		if (line.empty() || line[0] == '#' || line.size() < 2) {
			continue;
		}

		// Parse data values
		if (dimensions_parsed && is_int_start(line[0]) && values_read < nx * ny * nz) {
			std::istringstream iss(line);
			T value;

			while (iss >> value && values_read < values_.size()) {
				values_[values_read++] = value;
			}
			continue;
		}

		// Parse header information
		if (line.size() >= 6) {
			std::string prefix = line.substr(0, 5);

			if (prefix == "origi") {
				// Parse origin line
				std::istringstream iss(line);
				std::string word;
				iss >> word; // skip "origin"

				float x, y, z;
				if (iss >> x >> y >> z) {
					origin = Vector3(static_cast<T>(x), static_cast<T>(y), static_cast<T>(z));
				}
			} else if (prefix == "delta") {
				// Parse delta (basis vector) line
				std::istringstream iss(line);
				std::string word;
				iss >> word; // skip "delta"

				float x, y, z;
				if (iss >> x >> y >> z && delta_count < 3) {
					basis_vectors[delta_count] =
						Vector3(static_cast<T>(x), static_cast<T>(y), static_cast<T>(z));
					++delta_count;
				}
			} else if (prefix == "objec") {
				// Parse object line for grid dimensions
				if (line.find("object 1 class gridpositions") != std::string::npos) {
					if (parse_grid_counts(line, nx, ny, nz)) {
						dimensions_parsed = true;

						// Initialize grid with parsed dimensions
						config_.dimensions = Vector3_t<size_t>(nx, ny, nz);
						const size_t total_size = nx * ny * nz;
						values_.resize(total_size, T{0});
						values_read = 0;
					}
				}
			}
		}
	}

	// Validate parsing results
	if (!dimensions_parsed || values_read != values_.size()) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"BaseGrid::read_dx_format - Invalid file format: declared size {}, read {}",
						values_.size(),
						values_read);
	}

	// Set final configuration
	config_.origin = origin;
	if (delta_count == 3) {
		config_.basis = Matrix3(basis_vectors[0], basis_vectors[1], basis_vectors[2]);
	} else {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"BaseGrid::read_dx_format - Missing or invalid basis vectors");
	}

	update_derived_quantities();
}

template<typename T>
void BaseGrid<T>::write_average_profile(std::string_view filename, int axis) const {
	if (axis < 0 || axis >= 3) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"BaseGrid::write_average_profile - Invalid axis: {}",
						axis);
	}

	try {
		FileHandle file(filename.data(), "w");
		FILE* fp = file.get();

		const int dir0 = axis;
		const int dir1 = (axis + 1) % 3;
		const int dir2 = (axis + 2) % 3;

		// Jump values for different directions
		const size_t jump[3] = {ny() * nz(), nz(), 1};
		const size_t dims[3] = {nx(), ny(), nz()};

		// Calculate profile
		for (size_t i0 = 0; i0 < dims[dir0]; ++i0) {
			T sum = T{0};
			size_t count = 0;

			for (size_t i1 = 0; i1 < dims[dir1]; ++i1) {
				for (size_t i2 = 0; i2 < dims[dir2]; ++i2) {
					const size_t linear_idx = i0 * jump[dir0] + i1 * jump[dir1] + i2 * jump[dir2];
					sum += values_[linear_idx];
					++count;
				}
			}

			const T average = sum / static_cast<T>(count);

			// Calculate world coordinate along axis
			T world_coord = T{0};
			switch (dir0) {
			case 0:
				world_coord = config_.origin.x + static_cast<T>(i0) * config_.basis.ex().x;
				break;
			case 1:
				world_coord = config_.origin.y + static_cast<T>(i0) * config_.basis.ey().y;
				break;
			case 2:
				world_coord = config_.origin.z + static_cast<T>(i0) * config_.basis.ez().z;
				break;
			}

			std::fprintf(fp,
						 "%.10g %.10g\n",
						 static_cast<double>(world_coord),
						 static_cast<double>(average));
		}

		LOGINFO("BaseGrid::write_average_profile - Wrote profile along axis {} to '{}'",
				axis,
				filename);
	} catch (const Exception& e) {
		LOGERROR("BaseGrid::write_average_profile - Failed: {}", e.what());
		throw;
	}
}

// Explicit template instantiations for common types
template class BaseGrid<float>;

} // namespace ARBD
#endif