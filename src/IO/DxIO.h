/*********************************************************************
 * @file  dxreader.h
 *
 * @brief OpenDX format reader/writer for BaseGrid
 *        Host-only I/O operations separated from BaseGrid core
 *
 * @author Original: Jeff Comer <jcomer2@illinois.edu>
 * @author V2 Port: Pin-Yi Li with Claude 4.0 Sonnet
 *********************************************************************/
#pragma once

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

// Forward declaration
template<typename T>
class BaseGrid;

/**
 * @brief OpenDX format I/O operations for BaseGrid
 *        These functions are host-only and separated from the main BaseGrid class
 *        to maintain clean separation between core grid operations and I/O
 */
namespace DXReader {

/**
 * @brief Write BaseGrid to file with optional comments
 * @param grid The grid to write
 * @param filename Output filename
 * @param comments Optional comments to include in file header
 */
template<typename T>
void write_grid(const BaseGrid<T>& grid, std::string_view filename);

template<typename T>
void write_grid(const BaseGrid<T>& grid, std::string_view filename, std::string_view comments);

/**
 * @brief Write BaseGrid in OpenDX format to an open file handle
 * @param grid The grid to write
 * @param file Open file handle for writing
 * @param comments Optional comments to include in file header
 */
template<typename T>
void write_dx_format(const BaseGrid<T>& grid, const FileHandle& file, std::string_view comments);

/**
 * @brief Write BaseGrid in simple data format to an open file handle
 * @param grid The grid to write
 * @param file Open file handle for writing
 */
template<typename T>
void write_data_format(const BaseGrid<T>& grid, const FileHandle& file);

/**
 * @brief Read BaseGrid from OpenDX format file
 * @param filename Input filename
 * @return BaseGrid loaded from file
 * @throws Exception on file I/O errors or invalid format
 */
template<typename T=float>
BaseGrid<T> read_from_file(std::string_view filename);

/**
 * @brief Read BaseGrid from OpenDX format from an open file handle
 * @param grid Grid object to populate with data
 * @param file Open file handle for reading
 * @throws Exception on file I/O errors or invalid format
 */
template<typename T=float>
void read_dx_format(BaseGrid<T>& grid, const FileHandle& file);

/**
 * @brief Write average profile along specified axis
 * @param grid The grid to analyze
 * @param filename Output filename for profile data
 * @param axis Axis along which to compute average (0=x, 1=y, 2=z)
 * @throws Exception if axis is invalid or file I/O fails
 */
template<typename T>
void write_average_profile(const BaseGrid<T>& grid, std::string_view filename, int axis);

} // namespace DXReader

} // namespace ARBD
