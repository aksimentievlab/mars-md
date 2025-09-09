/*********************************************************************
 * @file  dxreader.cpp
 *
 * @brief OpenDX format reader/writer implementation for BaseGrid
 *        Host-only I/O operations separated from BaseGrid core
 *
 * @author Original: Jeff Comer <jcomer2@illinois.edu>
 * @author V2 Port: Pin-Yi Li with Claude 4.0 Sonnet
 *********************************************************************/
#include "DxIO.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/FileHandle.h"
#include "Types/BaseGrid.h"
#include <array>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>

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
||  I/O OPERATIONS     |
\*===================*/

namespace DXReader {

template<typename T>
void write_grid(const BaseGrid<T>& grid, std::string_view filename) {
	write_grid(grid, filename, "");
}

template<typename T>
void write_grid(const BaseGrid<T>& grid, std::string_view filename, std::string_view comments) {
	try {
		FileHandle file(filename.data(), "w");
		write_dx_format(grid, file, comments);
		LOGINFO("DXReader::write_grid - Successfully wrote {} values to '{}'",
				grid.size(),
				filename);
	} catch (const Exception& e) {
		LOGERROR("DXReader::write_grid - Failed to write to '{}': {}", filename, e.what());
		throw;
	}
}

template<typename T>
void write_dx_format(const BaseGrid<T>& grid, const FileHandle& file, std::string_view comments) {
	FILE* fp = file.get();

	// Write header
	std::fprintf(fp, "# %.*s\n", static_cast<int>(comments.size()), comments.data());
	std::fprintf(fp,
				 "object 1 class gridpositions counts %zu %zu %zu\n",
				 grid.nx(),
				 grid.ny(),
				 grid.nz());

	// Write origin
	const auto& origin = grid.origin();
	std::fprintf(fp,
				 "origin %.12g %.12g %.12g\n",
				 static_cast<double>(origin.x),
				 static_cast<double>(origin.y),
				 static_cast<double>(origin.z));

	// Write basis vectors (delta lines)
	const auto& basis = grid.basis();
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
	std::fprintf(fp,
				 "object 2 class gridconnections counts %zu %zu %zu\n",
				 grid.nx(),
				 grid.ny(),
				 grid.nz());
	std::fprintf(fp,
				 "object 3 class array type float rank 0 items %zu data follows\n",
				 grid.size());

	// Write data in groups of 3 for readability
	const size_t total_values = grid.size();
	const size_t complete_triplets = total_values / 3;
	const size_t remainder = total_values % 3;

	for (size_t i = 0; i < complete_triplets; ++i) {
		const size_t base_idx = i * 3;
		std::fprintf(fp,
					 "%.12g %.12g %.12g\n",
					 static_cast<double>(grid[base_idx]),
					 static_cast<double>(grid[base_idx + 1]),
					 static_cast<double>(grid[base_idx + 2]));
	}

	// Handle remainder
	if (remainder == 1) {
		std::fprintf(fp, "%.12g\n", static_cast<double>(grid[total_values - 1]));
	} else if (remainder == 2) {
		std::fprintf(fp,
					 "%.12g %.12g\n",
					 static_cast<double>(grid[total_values - 2]),
					 static_cast<double>(grid[total_values - 1]));
	}
}

template<typename T>
void write_data_format(const BaseGrid<T>& grid, const FileHandle& file) {
	FILE* fp = file.get();

	// Write grid dimensions
	std::fprintf(fp, "%zu\n%zu\n%zu\n", grid.nx(), grid.ny(), grid.nz());

	// Write origin
	const auto& origin = grid.origin();
	std::fprintf(fp,
				 "%.12g\n%.12g\n%.12g\n",
				 static_cast<double>(origin.x),
				 static_cast<double>(origin.y),
				 static_cast<double>(origin.z));

	// Write basis matrix (column by column)
	const auto& basis = grid.basis();
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
	for (size_t i = 0; i < grid.size(); ++i) {
		std::fprintf(fp, "%.12g\n", static_cast<double>(grid[i]));
	}
}

template<typename T>
BaseGrid<T> read_from_file(std::string_view filename) {
	try {
		FileHandle file(filename.data(), "r");
		BaseGrid<T> grid;
		read_dx_format(grid, file);

		LOGINFO("DXReader::read_from_file - Successfully read {} values from '{}'",
				grid.size(),
				filename);
		return grid;
	} catch (const Exception& e) {
		LOGERROR("DXReader::read_from_file - Failed to read from '{}': {}", filename, e.what());
		throw;
	}
}

template<typename T>
void read_dx_format(BaseGrid<T>& grid, const FileHandle& file) {
	FILE* fp = file.get();

	// Initialize parsing state
	size_t nx = 0, ny = 0, nz = 0;
	typename BaseGrid<T>::Vector3 origin{0, 0, 0};
	std::array<typename BaseGrid<T>::Vector3, 3> basis_vectors{};
	size_t delta_count = 0;
	size_t values_read = 0;
	bool dimensions_parsed = false;

	std::array<char, BUFFER_SIZE> line_buffer{};

	// First, we need to parse the dimensions to create the grid
	BaseGrid<T> temp_grid;
	std::vector<T> temp_values;

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

			while (iss >> value && values_read < temp_values.size()) {
				temp_values[values_read++] = value;
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
					origin = typename BaseGrid<T>::Vector3(static_cast<T>(x),
														   static_cast<T>(y),
														   static_cast<T>(z));
				}
			} else if (prefix == "delta") {
				// Parse delta (basis vector) line
				std::istringstream iss(line);
				std::string word;
				iss >> word; // skip "delta"

				float x, y, z;
				if (iss >> x >> y >> z && delta_count < 3) {
					basis_vectors[delta_count] = typename BaseGrid<T>::Vector3(static_cast<T>(x),
																			   static_cast<T>(y),
																			   static_cast<T>(z));
					++delta_count;
				}
			} else if (prefix == "objec") {
				// Parse object line for grid dimensions
				if (line.find("object 1 class gridpositions") != std::string::npos) {
					if (parse_grid_counts(line, nx, ny, nz)) {
						dimensions_parsed = true;

						// Initialize temporary storage with parsed dimensions
						const size_t total_size = nx * ny * nz;
						temp_values.resize(total_size, T{0});
						values_read = 0;
					}
				}
			}
		}
	}

	// Validate parsing results
	if (!dimensions_parsed || values_read != temp_values.size()) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"DXReader::read_dx_format - Invalid file format: declared size {}, read {}",
						temp_values.size(),
						values_read);
	}

	// Validate basis vectors
	if (delta_count != 3) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"DXReader::read_dx_format - Missing or invalid basis vectors");
	}

	// Create final grid with parsed configuration
	typename BaseGrid<T>::Matrix3 basis_matrix(basis_vectors[0],
											   basis_vectors[1],
											   basis_vectors[2]);
	grid = BaseGrid<T>(basis_matrix,
					   origin,
					   static_cast<idx_t>(nx),
					   static_cast<idx_t>(ny),
					   static_cast<idx_t>(nz));

	// Copy values to the grid
	for (size_t i = 0; i < temp_values.size(); ++i) {
		grid[i] = temp_values[i];
	}
}

template<typename T>
void write_average_profile(const BaseGrid<T>& grid, std::string_view filename, int axis) {
	if (axis < 0 || axis >= 3) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"DXReader::write_average_profile - Invalid axis: {}",
						axis);
	}

	try {
		FileHandle file(filename.data(), "w");
		FILE* fp = file.get();

		const int dir0 = axis;
		const int dir1 = (axis + 1) % 3;
		const int dir2 = (axis + 2) % 3;

		// Jump values for different directions
		const size_t jump[3] = {grid.ny() * grid.nz(), grid.nz(), 1};
		const size_t dims[3] = {grid.nx(), grid.ny(), grid.nz()};

		// Calculate profile
		for (size_t i0 = 0; i0 < dims[dir0]; ++i0) {
			T sum = T{0};
			size_t count = 0;

			for (size_t i1 = 0; i1 < dims[dir1]; ++i1) {
				for (size_t i2 = 0; i2 < dims[dir2]; ++i2) {
					const size_t linear_idx = i0 * jump[dir0] + i1 * jump[dir1] + i2 * jump[dir2];
					sum += grid[linear_idx];
					++count;
				}
			}

			const T average = sum / static_cast<T>(count);

			// Calculate world coordinate along axis
			T world_coord = T{0};
			const auto& basis = grid.basis();
			const auto& origin = grid.origin();
			switch (dir0) {
			case 0:
				world_coord = origin.x + static_cast<T>(i0) * basis.ex().x;
				break;
			case 1:
				world_coord = origin.y + static_cast<T>(i0) * basis.ey().y;
				break;
			case 2:
				world_coord = origin.z + static_cast<T>(i0) * basis.ez().z;
				break;
			}

			std::fprintf(fp,
						 "%.10g %.10g\n",
						 static_cast<double>(world_coord),
						 static_cast<double>(average));
		}

		LOGINFO("DXReader::write_average_profile - Wrote profile along axis {} to '{}'",
				axis,
				filename);
	} catch (const Exception& e) {
		LOGERROR("DXReader::write_average_profile - Failed: {}", e.what());
		throw;
	}
}

// Explicit template instantiations for common types
template void write_grid<float>(const BaseGrid<float>&, std::string_view);
template void write_grid<float>(const BaseGrid<float>&, std::string_view, std::string_view);
template void write_dx_format<float>(const BaseGrid<float>&, const FileHandle&, std::string_view);
template void write_data_format<float>(const BaseGrid<float>&, const FileHandle&);
template BaseGrid<float> read_from_file<float>(std::string_view);
template void read_dx_format<float>(BaseGrid<float>&, const FileHandle&);
template void write_average_profile<float>(const BaseGrid<float>&, std::string_view, int);

} // namespace DXReader

} // namespace ARBD
