/*********************************************************************
 * @file  BaseGrid.h
 *
 * @brief Modern C++20 BaseGrid class for mars2/cpp20 branch
 *        Host-side BaseGrid Manager.
 *        Multi-backend support (CUDA, SYCL, CPU) with clean separation
 *
 * @author Original: Jeff Comer <jcomer2@illinois.edu>
 *********************************************************************/
#pragma once

#include "IO/FileHandle.h"
#include "MARSException.h"
#include "MARSLogger.h"
#include <span>
#include <string_view>
#include <vector>

#include "Backend/Buffer.h"
#include "BaseGridDevice.h"
#include "Header.h"
#include "IndexList.h"
#include "Matrix3.h"
#include "Types.h"
#include "Vector3.h"
#include <cmath>

namespace MARS {

// Forward declarations
class FileHandle;
template<typename T>
class BaseGrid;

namespace DXReader {
template<typename T>
void write_grid(const BaseGrid<T>&, std::string_view);
template<typename T>
void write_grid(const BaseGrid<T>&, std::string_view, std::string_view);
template<typename T>
void write_dx_format(const BaseGrid<T>&, const FileHandle&, std::string_view);
template<typename T>
void write_data_format(const BaseGrid<T>&, const FileHandle&);
template<typename T>
BaseGrid<T> read_from_file(std::string_view);
template<typename T>
void read_dx_format(BaseGrid<T>&, const FileHandle&);
template<typename T>
void write_average_profile(const BaseGrid<T>&, std::string_view, int);
} // namespace DXReader

/**
 * @brief Template-based for BaseGrid class
 * @tparam T Data type stored in grid (typically float or double)
 * @param Config Configuration for the grid
 * @param values Values stored in the grid
 * @param device_ptr Device pointer for the grid
 * @param device_dirty Track if the device memory needs to be synced
 * @param basis_inv Inverse of the basis matrix
 * @param config Configuration for the grid
 * @param values Values stored in the grid
 */
template<typename T = mars_real>
class BaseGrid {
  public:
	using value_type = T;
	using Vector3 = Vector3_t<T>;
	using Matrix3 = Matrix3_t<T>;
	using IndexType = idx_t;
	/**
	 * @brief Grid configuration structure for initialization
	 */

	using BoundaryCondition = GridBoundaryCondition;
	/**
	 * @brief Grid configuration
	 * @param origin Origin point of the grid
	 * @param basis Basis vectors defining grid spacing
	 * @param dimensions Grid dimensions (nx, ny, nz)
	 * @param boundary Boundary condition
	 */
	struct Config {
		Vector3_t<T> origin{0, 0, 0};		  ///< Origin point of the grid
		Matrix3_t<T> basis{T(1)};			  ///< Basis vectors defining grid spacing
		Vector3_t<idx_t> dimensions{1, 1, 1}; ///< Grid dimensions (nx, ny, nz)
		BoundaryCondition boundary = BoundaryCondition::Dirichlet;

		/// Convert boundary condition to int for device use
		constexpr int boundary_as_int() const noexcept {
			return static_cast<int>(boundary);
		}

		idx_t total_size() const noexcept {
			return dimensions.x * dimensions.y * dimensions.z;
		}

		bool is_valid() const noexcept {
			return dimensions.x > 0 && dimensions.y > 0 && dimensions.z > 0;
		}
	};

  private:
	Config config_;
	Matrix3 basis_inv_;		///< Inverse of basis matrix (cached for performance)
	std::vector<T> values_; ///< Grid values in contiguous memory (host-only)
	mutable std::unique_ptr<DeviceBuffer<T>>
		device_buffer_; ///< Device memory managed via Buffer.h system (lazy initialization)
	void update_derived_quantities() {
		basis_inv_ = config_.basis.inverse();
	}

	/**
	 * @brief Throw if @p other has a different element count
	 * @param other Other grid
	 * @param op Operation name
	 */
	void require_same_size(const BaseGrid& other, const char* op) const {
		if (values_.size() != other.values_.size()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::%s: Size mismatch (%zu vs %zu)",
							op,
							values_.size(),
							other.values_.size());
		}
	}

  public:
	/*=============================*\
	|  CONSTRUCTORS & DESTRUCTORS   |
	\*=============================*/

	/**
	 * @brief Default constructor - creates unit grid
	 */
	BaseGrid() : config_{}, values_(1, T{0}) {
		update_derived_quantities();
	}

	/**
	 * @brief Primary constructor with full specification
	 */
	BaseGrid(const Matrix3& basis, const Vector3& origin, idx_t nx, idx_t ny, idx_t nz)
		: config_{origin, basis, Vector3_t<idx_t>(nx, ny, nz)} {

		if (!config_.is_valid()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid: Invalid dimensions (%zu, %zu, %zu)",
							nx,
							ny,
							nz);
		}

		values_.resize(config_.total_size(), T{0});
		update_derived_quantities();
		LOGDEBUG("BaseGrid({}, {}, {})", nx, ny, nz);
	}

	/**
	 * @brief Orthogonal grid constructor from box size and resolution
	 */
	BaseGrid(const Vector3& box_size, T dx) {
		T abs_dx = std::abs(dx);
		Vector3 abs_box{std::abs(box_size.x), std::abs(box_size.y), std::abs(box_size.z)};

		// Calculate grid dimensions ensuring dx is upper bound for spacing
		auto nx = static_cast<idx_t>(std::max(1.0f, std::ceil(abs_box.x / abs_dx)));
		auto ny = static_cast<idx_t>(std::max(1.0f, std::ceil(abs_box.y / abs_dx)));
		auto nz = static_cast<idx_t>(std::max(1.0f, std::ceil(abs_box.z / abs_dx)));

		config_.dimensions = Vector3_t<idx_t>(nx, ny, nz);
		config_.basis = Matrix3(Vector3(abs_box.x / nx, 0, 0),
								Vector3(0, abs_box.y / ny, 0),
								Vector3(0, 0, abs_box.z / nz));
		config_.origin = -T(0.5) * abs_box;

		values_.resize(config_.total_size(), T{0});

		update_derived_quantities();
		LOGDEBUG("BaseGrid(box={}, dx={}) - Orthogonal constructor", box_size.length(), dx);
	}

	/**
	 * @brief Copy constructor
	 */
	BaseGrid(const BaseGrid& other) : config_(other.config_), basis_inv_(other.basis_inv_) {
		values_ = other.values_;
	}

	/**
	 * @brief Move constructor
	 */
	BaseGrid(BaseGrid&& other) noexcept
		: config_(std::move(other.config_)), basis_inv_(std::move(other.basis_inv_)) {
		values_ = std::move(other.values_);
		device_buffer_ = std::move(other.device_buffer_);
	}

	/**
	 * @brief Copy assignment
	 */
	BaseGrid& operator=(const BaseGrid& other) {
		if (this != &other) {
			config_ = other.config_;
			basis_inv_ = other.basis_inv_;
			values_ = other.values_;
			device_buffer_.reset(); // Force reallocation on device
		}
		return *this;
	}

	/**
	 * @brief Move assignment
	 */
	BaseGrid& operator=(BaseGrid&& other) noexcept {
		if (this != &other) {
			config_ = std::move(other.config_);
			basis_inv_ = std::move(other.basis_inv_);
			values_ = std::move(other.values_);
			device_buffer_ = std::move(other.device_buffer_);
		}
		return *this;
	}

	/**
	 * @brief Destructor
	 */
	virtual ~BaseGrid() = default;

	/*=======================*\
	|  CORE GRID OPERATIONS   |
	\*=======================*/

	/**
	 * @brief Get grid dimensions
	 */
	const Vector3_t<idx_t>& dimensions() const noexcept {
		return config_.dimensions;
	}

	idx_t nx() const noexcept {
		return config_.dimensions.x;
	}
	idx_t ny() const noexcept {
		return config_.dimensions.y;
	}
	idx_t nz() const noexcept {
		return config_.dimensions.z;
	}
	idx_t size() const noexcept {
		return config_.total_size();
	}

	/**
	 * @brief Get grid configuration
	 */
	const Config& config() const noexcept {
		return config_;
	}

	/**
	 * @brief Boundary condition applied when sampling outside the grid
	 */
	BoundaryCondition boundary() const noexcept {
		return config_.boundary;
	}

	/**
	 * @brief Set the boundary condition (default is Dirichlet)
	 */
	void set_boundary(BoundaryCondition bc) noexcept {
		config_.boundary = bc;
	}
	const Vector3& origin() const noexcept {
		return config_.origin;
	}
	const Matrix3& basis() const noexcept {
		return config_.basis;
	}
	const Matrix3& basis_inverse() const noexcept {
		return basis_inv_;
	}

	/*=======================*\
	|  HOST-ONLY OPERATIONS   |
	\*=======================*/

	/**
	 * @brief Access grid values (host-only - uses std::vector)
	 */
	T& operator[](idx_t index) {
		return values_[index];
	}
	const T& operator[](idx_t index) const {
		return values_[index];
	}

	T& at(idx_t index) {
		if (index >= values_.size()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::at: Index %zu out of range [0, %zu)",
							index,
							values_.size());
		}
		return values_[index];
	}

	const T& at(idx_t index) const {
		if (index >= values_.size()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::at: Index %zu out of range [0, %zu)",
							index,
							values_.size());
		}
		return values_[index];
	}

	/**
	 * @brief Get raw data pointer for interfacing with backends (host-only)
	 */
	T* data() noexcept {
		return values_.data();
	}
	const T* data() const noexcept {
		return values_.data();
	}

	/**
	 * @brief Get span view of data (C++20, host-only)
	 */
	std::span<T> span() noexcept {
		return std::span<T>(values_);
	}
	std::span<const T> span() const noexcept {
		return std::span<const T>(values_);
	}

	/**
	 * @brief Device memory management methods (host-only)
	 * @param resource Resource to allocate device memory on
	 * @return Reference to DeviceBuffer (lazy initialization)
	 */
	DeviceBuffer<T>& get_device_buffer(const Resource& resource = Resource{}) const {
		if (!device_buffer_) {
			Resource target_resource = (resource == Resource{}) ? get_default_resource() : resource;
			device_buffer_ =
				std::make_unique<DeviceBuffer<T>>(config_.total_size(), target_resource);
			// Copy host data to device
			device_buffer_->copy_from_host(values_.data(), values_.size());
		}
		return *device_buffer_;
	}

	/**
	 * @brief Sync host data to device
	 * @param resource Resource to sync to (optional)
	 */
	void sync_to_device(const Resource& resource = Resource{}) {
		auto& buffer = get_device_buffer(resource);
		buffer.copy_from_host(values_.data(), values_.size());
	}

	/**
	 * @brief Sync device data to host
	 * @param resource Resource to sync from (optional)
	 */
	void sync_from_device(const Resource& resource = Resource{}) {
		if (device_buffer_) {
			device_buffer_->copy_to_host(values_.data(), values_.size());
		}
	}

	/**
	 * @brief Get device data pointer (for kernel use)
	 * @param resource Resource to get pointer from (optional)
	 * @return Raw device pointer
	 */
	T* get_device_pointer(const Resource& resource = Resource{}) const {
		auto& buffer = get_device_buffer(resource);
		return buffer.data();
	}

	/// @brief Geometry half of a device view, shared by both view flavors.
	GridGeometry<T> device_geometry() const {
		GridGeometry<T> geom;
		geom.origin = config_.origin;
		geom.basis = config_.basis;
		geom.basis_inv = basis_inv_;
		geom.dimensions = config_.dimensions;
		geom.grid_id = -1; // Can be set by caller if needed
		geom.boundary_condition = config_.boundary_as_int();
		return geom;
	}

	/**
	 * @brief Create read-only device view for kernel use
	 * @param resource Resource to get device data from (optional)
	 * @return BaseGridView for device kernels
	 */
	BaseGridView<T> get_device_view(const Resource& resource = Resource{}) const {
		auto& buffer = get_device_buffer(resource);
		BaseGridView<T> view;
		static_cast<GridGeometry<T>&>(view) = device_geometry();
		view.data = buffer.data();
		return view;
	}

	/**
	 * @brief Create writable device view, for grid-mutating kernels
	 * @param resource Resource to get device data from (optional)
	 * @return BaseGridMutableView for device kernels that write to the grid
	 * @note Host values_ are stale until sync_from_device().
	 */
	BaseGridMutableView<T> get_mutable_device_view(const Resource& resource = Resource{}) const {
		auto& buffer = get_device_buffer(resource);
		BaseGridMutableView<T> view;
		static_cast<GridGeometry<T>&>(view) = device_geometry();
		view.data = buffer.data();
		return view;
	}

  private:
	/**
	 * @brief Get default resource for device allocation
	 */
	Resource get_default_resource() const {
		// Use default device resource (same logic as Buffer.h)
		return Resource{};
	}

	/*===================*\
	|  INDEX OPERATIONS   |
	\*===================*/

	/**
	 * @brief Convert 3D indices to linear index
	 */
	idx_t index(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return iz + iy * nz() + ix * ny() * nz();
	}

	/**
	 * @brief Convert linear index to 3D indices using device-safe IndexList
	 */
	IndexList<idx_t, 3> index_to_ijk(idx_t linear_index) const {
		IndexList<idx_t, 3> result;
		result.add(linear_index / (ny() * nz())); // ix
		result.add((linear_index / nz()) % ny()); // iy
		result.add(linear_index % nz());		  // iz
		return result;
	}

	/**
	 * @brief Get position in space from linear index
	 */
	Vector3 get_position(idx_t linear_index) const {
		auto ijk = index_to_ijk(linear_index);
		Vector3 grid_coords(static_cast<T>(ijk[0]), static_cast<T>(ijk[1]), static_cast<T>(ijk[2]));
		return basis().transform(grid_coords) + origin();
	}

	/**
	 * @brief Transform world position to grid coordinates
	 */
	Vector3 transform_to_grid(const Vector3& world_pos) const {
		return basis_inverse().transform(world_pos - origin());
	}

	/**
	 * @brief Transform grid coordinates to world position
	 */
	Vector3 transform_to_world(const Vector3& grid_pos) const {
		return basis().transform(grid_pos) + origin();
	}

	/**
	 * @brief Check if position is within grid bounds
	 */
	bool in_bounds(const Vector3& world_pos) const {
		Vector3 grid_pos = transform_to_grid(world_pos);
		return grid_pos.x >= 0 && grid_pos.x < nx() && grid_pos.y >= 0 && grid_pos.y < ny() &&
			   grid_pos.z >= 0 && grid_pos.z < nz();
	}

	/**
	 * @brief Check if position is within interpolation bounds (with safety margin)
	 */
	bool in_interpolation_bounds(const Vector3& world_pos) const {
		Vector3 grid_pos = transform_to_grid(world_pos);
		return grid_pos.x >= 1 && grid_pos.x < nx() - 1 && grid_pos.y >= 1 &&
			   grid_pos.y < ny() - 1 && grid_pos.z >= 1 && grid_pos.z < nz() - 1;
	}

	/*====================*\
	|  UTILITY OPERATIONS  |
	\*====================*/
  public:
	/**
	 * @brief Zero all grid values (host-only)
	 */
	void zero() {
		std::fill(values_.begin(), values_.end(), T{0});
	}

	/**
	 * @brief Add constant to all grid values (host-only)
	 */
	void shift(T value) {
		for (auto& v : values_)
			v += value;
	}

	/**
	 * @brief Scale all grid values by constant (host-only)
	 */
	void scale(T factor) {
		for (auto& v : values_)
			v *= factor;
	}

	/**
	 * @brief Compute mean of all grid values
	 */
	T mean() const {
		T sum = T{0};
		for (const auto& v : values_)
			sum += v;
		return sum / static_cast<T>(values_.size());
	}

	/**
	 * @brief Element-wise multiplication with another grid (host-only)
	 */
	BaseGrid& multiply(const BaseGrid& other) {
		require_same_size(other, "multiply");
		for (idx_t i = 0; i < values_.size(); ++i) {
			values_[i] *= other.values_[i];
		}
		return *this;
	}

	/**
	 * @brief Element-wise addition with another grid (host-only)
	 */
	BaseGrid& add(const BaseGrid& other) {
		require_same_size(other, "add");
		for (idx_t i = 0; i < values_.size(); ++i) {
			values_[i] += other.values_[i];
		}
		return *this;
	}

	/**
	 * @brief Element-wise subtraction of another grid (host-only)
	 */
	BaseGrid& subtract(const BaseGrid& other) {
		require_same_size(other, "subtract");
		for (idx_t i = 0; i < values_.size(); ++i) {
			values_[i] -= other.values_[i];
		}
		return *this;
	}

	/**
	 * @brief Volume integral, sum(v) * cell volume
	 */
	T integrate() const {
		T sum = T{0};
		for (const auto& v : values_)
			sum += v;
		return sum * get_cell_volume();
	}

	/**
	 * @brief Smallest grid value
	 */
	T min() const {
		return *std::min_element(values_.begin(), values_.end());
	}

	/**
	 * @brief Largest grid value
	 */
	T max() const {
		return *std::max_element(values_.begin(), values_.end());
	}

	/**
	 * @brief True if any value is NaN or infinite
	 */
	bool has_non_finite() const {
		for (const auto& v : values_) {
			if (!std::isfinite(static_cast<double>(v)))
				return true;
		}
		return false;
	}

	/**
	 * @brief Trilinear resample onto new dimensions, preserving world extent
	 * @param new_dims Target (nx, ny, nz), each >= 1
	 * @note Edge taps clamp (Neumann) regardless of this grid's boundary
	 *       condition, so a constant field resamples to the same constant.
	 */
	BaseGrid resample(const Vector3_t<idx_t>& new_dims) const {
		if (new_dims.x < 1 || new_dims.y < 1 || new_dims.z < 1) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::resample: Invalid dimensions (%zu, %zu, %zu)",
							new_dims.x,
							new_dims.y,
							new_dims.z);
		}

		Matrix3 new_basis(basis().ex() * (static_cast<T>(nx()) / static_cast<T>(new_dims.x)),
						  basis().ey() * (static_cast<T>(ny()) / static_cast<T>(new_dims.y)),
						  basis().ez() * (static_cast<T>(nz()) / static_cast<T>(new_dims.z)));
		BaseGrid out(new_basis, origin(), new_dims.x, new_dims.y, new_dims.z);

		for (idx_t ix = 0; ix < new_dims.x; ++ix) {
			for (idx_t iy = 0; iy < new_dims.y; ++iy) {
				for (idx_t iz = 0; iz < new_dims.z; ++iz) {
					const Vector3 world = out.transform_to_world(
						Vector3(static_cast<T>(ix), static_cast<T>(iy), static_cast<T>(iz)));
					out.values_[out.index(ix, iy, iz)] =
						interpolate_grid_point(values_.data(),
											   world,
											   config_.origin,
											   basis_inv_,
											   config_.dimensions,
											   static_cast<int>(GridBoundaryCondition::Neumann));
				}
			}
		}
		return out;
	}

	/**
	 * @brief Copy into a larger grid, padding with a constant
	 * @param pad_lo Cells added on the low side of each axis
	 * @param pad_hi Cells added on the high side of each axis
	 * @param fill Value written to the added cells
	 */
	BaseGrid
	pad(const Vector3_t<idx_t>& pad_lo, const Vector3_t<idx_t>& pad_hi, T fill = T{0}) const {
		BaseGrid out(basis(),
					 transform_to_world(Vector3(-static_cast<T>(pad_lo.x),
												-static_cast<T>(pad_lo.y),
												-static_cast<T>(pad_lo.z))),
					 nx() + pad_lo.x + pad_hi.x,
					 ny() + pad_lo.y + pad_hi.y,
					 nz() + pad_lo.z + pad_hi.z);
		std::fill(out.values_.begin(), out.values_.end(), fill);

		for (idx_t ix = 0; ix < nx(); ++ix) {
			for (idx_t iy = 0; iy < ny(); ++iy) {
				for (idx_t iz = 0; iz < nz(); ++iz) {
					out.values_[out.index(ix + pad_lo.x, iy + pad_lo.y, iz + pad_lo.z)] =
						values_[index(ix, iy, iz)];
				}
			}
		}
		return out;
	}

	/**
	 * @brief Extract a sub-box
	 * @param lo Inclusive lower corner in grid indices
	 * @param dims Extent of the crop; lo + dims must not exceed the grid
	 */
	BaseGrid crop(const Vector3_t<idx_t>& lo, const Vector3_t<idx_t>& dims) const {
		if (dims.x < 1 || dims.y < 1 || dims.z < 1 || lo.x + dims.x > nx() ||
			lo.y + dims.y > ny() || lo.z + dims.z > nz()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::crop: (%zu,%zu,%zu)+(%zu,%zu,%zu) exceeds (%zu,%zu,%zu)",
							lo.x,
							lo.y,
							lo.z,
							dims.x,
							dims.y,
							dims.z,
							nx(),
							ny(),
							nz());
		}

		BaseGrid out(basis(),
					 transform_to_world(
						 Vector3(static_cast<T>(lo.x), static_cast<T>(lo.y), static_cast<T>(lo.z))),
					 dims.x,
					 dims.y,
					 dims.z);

		for (idx_t ix = 0; ix < dims.x; ++ix) {
			for (idx_t iy = 0; iy < dims.y; ++iy) {
				for (idx_t iz = 0; iz < dims.z; ++iz) {
					out.values_[out.index(ix, iy, iz)] =
						values_[index(ix + lo.x, iy + lo.y, iz + lo.z)];
				}
			}
		}
		return out;
	}

	/**
	 * @brief Get system box as Matrix3
	 */
	Matrix3 get_box() const {
		return Matrix3(static_cast<T>(nx()) * basis().ex(),
					   static_cast<T>(ny()) * basis().ey(),
					   static_cast<T>(nz()) * basis().ez());
	}

	/**
	 * @brief Get grid center position
	 */
	Vector3 get_center() const {
		Vector3 center_grid(T(0.5) * nx(), T(0.5) * ny(), T(0.5) * nz());
		return transform_to_world(center_grid);
	}

	/**
	 * @brief Get volume of single grid cell
	 */
	T get_cell_volume() const {
		return std::abs(basis().det());
	}

	/**
	 * @brief Get total grid volume
	 */
	T get_total_volume() const {
		return get_cell_volume() * static_cast<T>(size());
	}

	/*===============================*\
	|  PERIODIC BOUNDARY CONDITIONS   |
	\*===============================*/

	/**
	 * @brief Wrap scalar distance for minimum image convention
	 * @param x Distance component
	 * @param l Box length in that dimension
	 * @return Wrapped distance: -0.5*l <= x < 0.5*l
	 */
	static inline T wrapDiff(T x, T l) {
#ifdef USE_CUDA
		int image = int(floorf(x / l));
#else
		int image = int(floor(x / l));
#endif
		x -= image * l;
		if (x >= T(0.5) * l)
			x -= l;
		return x;
	}

	/**
	 * @brief Apply minimum image convention to vector difference
	 * @param dr Vector difference between two points
	 * @return Minimum image vector difference
	 */
	Vector3 wrapDiff(const Vector3& dr) const {
		// Only apply wrapping if we have periodic boundaries
		if (config_.boundary != BoundaryCondition::Periodic) {
			return dr;
		}

		// Get box lengths from basis vectors
		Vector3 box_lengths = Vector3(config_.basis.ex().length(),
									  config_.basis.ey().length(),
									  config_.basis.ez().length());

		// Apply minimum image convention to each component
		Vector3 wrapped_dr = dr;
		wrapped_dr.x = wrapDiff(dr.x, box_lengths.x);
		wrapped_dr.y = wrapDiff(dr.y, box_lengths.y);
		wrapped_dr.z = wrapDiff(dr.z, box_lengths.z);

		return wrapped_dr;
	}

	/*===========================*\
	|  INTERPOLATION & SAMPLING   |
	\*===========================*/

	/**
	 * @brief Interpolate value at world position using trilinear interpolation (host-only)
	 * @param world_pos World position to interpolate at
	 * @return Interpolated value
	 */
	T interpolate(const Vector3& world_pos) const {
		return interpolate_grid_point(values_.data(),
									  world_pos,
									  config_.origin,
									  basis_inv_,
									  config_.dimensions,
									  config_.boundary_as_int());
	}

	/**
	 * @brief Interpolate value at world position using explicit data pointer (device-compatible)
	 * @param data_ptr Pointer to grid data (host or device)
	 * @param world_pos World position to interpolate at
	 * @return Interpolated value
	 */
	T interpolate(const T* data_ptr, const Vector3& world_pos) const {
		return interpolate_grid_point(data_ptr,
									  world_pos,
									  config_.origin,
									  basis_inv_,
									  config_.dimensions,
									  config_.boundary_as_int());
	}

	/**
	 * @brief Get value at nearest grid point (host-only)
	 * @param world_pos World position to sample at
	 * @return Nearest grid point value
	 */
	T get_value(const Vector3& world_pos) const {
		return get_value_nearest(values_.data(),
								 world_pos,
								 config_.origin,
								 basis_inv_,
								 config_.dimensions,
								 config_.boundary_as_int());
	}

	/**
	 * @brief Get value at nearest grid point using explicit data pointer (device-compatible)
	 * @param data_ptr Pointer to grid data (host or device)
	 * @param world_pos World position to sample at
	 * @return Nearest grid point value
	 */
	T get_value(const T* data_ptr, const Vector3& world_pos) const {
		return get_value_nearest(data_ptr,
								 world_pos,
								 config_.origin,
								 basis_inv_,
								 config_.dimensions,
								 config_.boundary_as_int());
	}

	/**
	 * @brief Compute gradient at world position using finite differences (host-only)
	 * @param world_pos World position to compute gradient at
	 * @return Gradient vector
	 */
	Vector3 compute_gradient(const Vector3& world_pos) const {
		return MARS::compute_gradient<T>(values_.data(),
										 world_pos,
										 config_.origin,
										 config_.basis,
										 basis_inv_,
										 config_.dimensions,
										 config_.boundary_as_int());
	}

	/**
	 * @brief Compute gradient at world position using explicit data pointer (device-compatible)
	 * @param data_ptr Pointer to grid data (host or device)
	 * @param world_pos World position to compute gradient at
	 * @return Gradient vector
	 */
	Vector3 compute_gradient(const T* data_ptr, const Vector3& world_pos) const {
		return MARS::compute_gradient<T>(data_ptr,
										 world_pos,
										 config_.origin,
										 config_.basis,
										 basis_inv_,
										 config_.dimensions,
										 config_.boundary_as_int());
	}

	/*====================*\
	|  NEIGHBOR OPERATIONS |
	\*====================*/

	/**
	 * @brief Device-safe 3x3x3 neighbor structure
	 */
	template<typename U = T>
	struct alignas(16) NeighborList {
		U v[3][3][3]; ///< 3x3x3 neighborhood values

		constexpr NeighborList() {
			// Initialize to zero
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					for (int k = 0; k < 3; ++k) {
						v[i][j][k] = U{0};
					}
				}
			}
		}

		constexpr U& operator()(int i, int j, int k) noexcept {
			return v[i + 1][j + 1][k + 1];
		}

		constexpr const U& operator()(int i, int j, int k) const noexcept {
			return v[i + 1][j + 1][k + 1];
		}

		constexpr U& center() noexcept {
			return v[1][1][1];
		}
		constexpr const U& center() const noexcept {
			return v[1][1][1];
		}
	};

	/**
	 * @brief Get 3x3x3 neighborhood around a grid point (host-only)
	 */
	NeighborList<T> get_neighbor_list(idx_t ix, idx_t iy, idx_t iz) const {
		return get_neighbor_list_from_grid(values_.data(), ix, iy, iz, config_.dimensions);
	}

	/**
	 * @brief Get 3x3x3 neighborhood using explicit data pointer (device-compatible)
	 */
	NeighborList<T> get_neighbor_list(const T* data_ptr, idx_t ix, idx_t iy, idx_t iz) const {
		return get_neighbor_list_from_grid(data_ptr, ix, iy, iz, config_.dimensions);
	}

	/**
	 * @brief Get neighbor value at relative offset (host-only)
	 */
	T get_neighbor(idx_t ix, idx_t iy, idx_t iz, int di, int dj, int dk) const {
		LOGDEBUG("get_neighbor on HOST: ix={}, iy={}, iz={}, di={}, dj={}, dk={}",
				 ix,
				 iy,
				 iz,
				 di,
				 dj,
				 dk);
		return get_neighbor_from_grid(values_.data(),
									  ix,
									  iy,
									  iz,
									  di,
									  dj,
									  dk,
									  config_.dimensions,
									  config_.boundary_as_int());
	}

	/**
	 * @brief Get neighbor value using explicit data pointer (device-compatible)
	 */
	T get_neighbor(const T* data_ptr, idx_t ix, idx_t iy, idx_t iz, int di, int dj, int dk) const {
		return get_neighbor_from_grid(data_ptr,
									  ix,
									  iy,
									  iz,
									  di,
									  dj,
									  dk,
									  config_.dimensions,
									  config_.boundary_as_int());
	}

	/**
	 * @brief Get neighbor at world position with offset (host-only)
	 */
	T get_neighbor_at_position(const Vector3& world_pos, int di, int dj, int dk) const {
		const Vector3 grid_pos = transform_to_grid(world_pos);

		const idx_t ix = static_cast<idx_t>(grid_pos.x);
		const idx_t iy = static_cast<idx_t>(grid_pos.y);
		const idx_t iz = static_cast<idx_t>(grid_pos.z);

		return get_neighbor(ix, iy, iz, di, dj, dk);
	}

	/**
	 * @brief Get neighbor at world position using explicit data pointer (device-compatible)
	 */
	T get_neighbor_at_position(const T* data_ptr,
							   const Vector3& world_pos,
							   int di,
							   int dj,
							   int dk) const {
		const Vector3 grid_pos = transform_to_grid(world_pos);

		const idx_t ix = static_cast<idx_t>(grid_pos.x);
		const idx_t iy = static_cast<idx_t>(grid_pos.y);
		const idx_t iz = static_cast<idx_t>(grid_pos.z);

		return get_neighbor(data_ptr, ix, iy, iz, di, dj, dk);
	}

  public:
	/**
	 * @brief Friend declarations for DXReader I/O functions
	 */
	template<typename U>
	friend void DXReader::write_grid(const BaseGrid<U>&, std::string_view);
	template<typename U>
	friend void DXReader::write_grid(const BaseGrid<U>&, std::string_view, std::string_view);
	template<typename U>
	friend void DXReader::write_dx_format(const BaseGrid<U>&, const FileHandle&, std::string_view);
	template<typename U>
	friend void DXReader::write_data_format(const BaseGrid<U>&, const FileHandle&);
	template<typename U>
	friend BaseGrid<U> DXReader::read_from_file(std::string_view);
	template<typename U>
	friend void DXReader::read_dx_format(BaseGrid<U>&, const FileHandle&);
	template<typename U>
	friend void DXReader::write_average_profile(const BaseGrid<U>&, std::string_view, int);
};

using BaseGridf = BaseGrid<float>;
using BaseGridd = BaseGrid<double>;

} // namespace MARS
