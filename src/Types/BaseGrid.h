/*********************************************************************
 * @file  BaseGrid.h
 *
 * @brief Modern C++20 BaseGrid class for arbd2/cpp20 branch
 *        Multi-backend support (CUDA, SYCL, CPU) with clean separation
 *
 * @author Original: Jeff Comer <jcomer2@illinois.edu>
 * @author V2 Port: Pin-Yi Li with Claude 4.0 Sonnet
 *********************************************************************/
#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/FileHandle.h"
#include <span>
#include <string_view>
#include <vector>

#include "Header.h"
#include "IndexList.h"
#include "Matrix3.h"
#include "Types.h"
#include "Vector3.h"
#include <cmath>

#ifdef HOST_GUARD
#include "Backend/Buffer.h"
#endif

namespace ARBD {

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
template<typename T = float>
class BaseGrid {
  public:
	using value_type = T;
	using Vector3 = Vector3_t<T>;
	using Matrix3 = Matrix3_t<T>;
	using IndexType = idx_t;
	/**
	 * @brief Grid configuration structure for initialization
	 */

	enum class BoundaryCondition : int {
		Dirichlet = 0, ///< Fixed value at boundary
		Neumann = 1,   ///< Fixed derivative at boundary
		Periodic = 2   ///< Periodic boundary conditions
	};
	struct Config {
		/**
		 * @brief Boundary condition types for grid operations
		 */
		Vector3_t<T> origin{0, 0, 0};		  ///< Origin point of the grid
		Matrix3_t<T> basis{T(1)};			  ///< Basis vectors defining grid spacing
		Vector3_t<idx_t> dimensions{1, 1, 1}; ///< Grid dimensions (nx, ny, nz)
		BoundaryCondition boundary = BoundaryCondition::Periodic;

		HOST DEVICE constexpr idx_t total_size() const noexcept {
			return dimensions.x * dimensions.y * dimensions.z;
		}

		HOST DEVICE constexpr bool is_valid() const noexcept {
			return dimensions.x > 0 && dimensions.y > 0 && dimensions.z > 0;
		}
	};

  private:
	// Core grid data
	Config config_;
	Matrix3 basis_inv_; ///< Inverse of basis matrix (cached for performance)

// Host-side storage and device memory management
#ifdef HOST_GUARD
	std::vector<T> values_; ///< Grid values in contiguous memory (host-only)
	mutable std::unique_ptr<DeviceBuffer<T>>
		device_buffer_; ///< Device memory managed via Buffer.h system (lazy initialization)
#endif

	HOST DEVICE void update_derived_quantities() {
		basis_inv_ = config_.basis.inverse();
	}

  public:
	/*=============================*\
	|  CONSTRUCTORS & DESTRUCTORS   |
	\*=============================*/

	/**
	 * @brief Default constructor - creates unit grid
	 * @usage BaseGrid grid;
	 * @usage BaseGrid grid(basis, origin, nx, ny, nz);
	 * @usage BaseGrid grid(box_size, dx);
	 * @usage BaseGrid grid(other);
	 * @usage BaseGrid grid(std::move(other));
	 * @usage BaseGrid grid(config);
	 * @usage BaseGrid grid(std::move(config));
	 * @usage BaseGrid grid(config, values);
	 */
	BaseGrid()
		: config_{}
#ifdef HOST_GUARD
		  ,
		  values_(1, T{0})
#endif
	{
		update_derived_quantities();
		LOGINFO("BaseGrid() - Default constructor");
	}

	/**
	 * @brief Primary constructor with full specification
	 */
	HOST BaseGrid(const Matrix3& basis, const Vector3& origin, idx_t nx, idx_t ny, idx_t nz)
		: config_{origin, basis, Vector3_t<idx_t>(nx, ny, nz)} {

		if (!config_.is_valid()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid: Invalid dimensions (%zu, %zu, %zu)",
							nx,
							ny,
							nz);
		}

#ifdef HOST_GUARD
		values_.resize(config_.total_size(), T{0});
#endif

		update_derived_quantities();
		LOGINFO("BaseGrid({}, {}, {})", nx, ny, nz);
	}

	/**
	 * @brief Orthogonal grid constructor from box size and resolution
	 */
	HOST BaseGrid(const Vector3& box_size, T dx) {
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

#ifdef HOST_GUARD
		values_.resize(config_.total_size(), T{0});
#endif

		update_derived_quantities();
		LOGINFO("BaseGrid(box={}, dx={}) - Orthogonal constructor", box_size.length(), dx);
	}

	/**
	 * @brief Copy constructor
	 */
	HOST BaseGrid(const BaseGrid& other) : config_(other.config_), basis_inv_(other.basis_inv_) {
#ifdef HOST_GUARD
		values_ = other.values_;
#endif
		LOGINFO("BaseGrid - Copy");
	}

	/**
	 * @brief Move constructor
	 */
	HOST BaseGrid(BaseGrid&& other) noexcept
		: config_(std::move(other.config_)), basis_inv_(std::move(other.basis_inv_)) {
#ifdef HOST_GUARD
		values_ = std::move(other.values_);
		device_buffer_ = std::move(other.device_buffer_);
#endif
	}

	/**
	 * @brief Copy assignment
	 */
	HOST BaseGrid& operator=(const BaseGrid& other) {
		if (this != &other) {
			config_ = other.config_;
			basis_inv_ = other.basis_inv_;
#ifdef HOST_GUARD
			values_ = other.values_;
			device_buffer_.reset(); // Force reallocation on device
#endif
		}
		return *this;
	}

	/**
	 * @brief Move assignment
	 */
	HOST BaseGrid& operator=(BaseGrid&& other) noexcept {
		if (this != &other) {
			config_ = std::move(other.config_);
			basis_inv_ = std::move(other.basis_inv_);
#ifdef HOST_GUARD
			values_ = std::move(other.values_);
			device_buffer_ = std::move(other.device_buffer_);
#endif
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
	// Note: All methods below are device-compatible (HOST DEVICE)

	/**
	 * @brief Get grid dimensions
	 */
	HOST DEVICE const Vector3_t<idx_t>& dimensions() const noexcept {
		return config_.dimensions;
	}

	HOST DEVICE idx_t nx() const noexcept {
		return config_.dimensions.x;
	}
	HOST DEVICE idx_t ny() const noexcept {
		return config_.dimensions.y;
	}
	HOST DEVICE idx_t nz() const noexcept {
		return config_.dimensions.z;
	}
	HOST DEVICE idx_t size() const noexcept {
		return config_.total_size();
	}

	/**
	 * @brief Get grid configuration
	 */
	HOST DEVICE const Config& config() const noexcept {
		return config_;
	}
	HOST DEVICE const Vector3& origin() const noexcept {
		return config_.origin;
	}
	HOST DEVICE const Matrix3& basis() const noexcept {
		return config_.basis;
	}
	HOST DEVICE const Matrix3& basis_inverse() const noexcept {
		return basis_inv_;
	}

	/*=======================*\
	|  HOST-ONLY OPERATIONS   |
	\*=======================*/
	// Note: All methods below are host-only (require HOST_GUARD)

/**
 * @brief Access grid values (host-only - uses std::vector)
 */
#ifdef HOST_GUARD
	HOST T& operator[](idx_t index) {
		return values_[index];
	}
	HOST const T& operator[](idx_t index) const {
		return values_[index];
	}

	HOST T& at(idx_t index) {
		if (index >= values_.size()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::at: Index %zu out of range [0, %zu)",
							index,
							values_.size());
		}
		return values_[index];
	}

	HOST const T& at(idx_t index) const {
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
	HOST T* data() noexcept {
		return values_.data();
	}
	HOST const T* data() const noexcept {
		return values_.data();
	}

	/**
	 * @brief Get span view of data (C++20, host-only)
	 */
	HOST std::span<T> span() noexcept {
		return std::span<T>(values_);
	}
	HOST std::span<const T> span() const noexcept {
		return std::span<const T>(values_);
	}
#endif

/**
 * @brief Device memory management methods (host-only)
 */
#ifdef HOST_GUARD
	/**
	 * @brief Get device buffer for backend operations
	 * @param resource Resource to allocate device memory on
	 * @return Reference to DeviceBuffer (lazy initialization)
	 */
	HOST DeviceBuffer<T>& get_device_buffer(const Resource& resource = Resource{}) const {
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
	HOST void sync_to_device(const Resource& resource = Resource{}) {
		auto& buffer = get_device_buffer(resource);
		buffer.copy_from_host(values_.data(), values_.size());
	}

	/**
	 * @brief Sync device data to host
	 * @param resource Resource to sync from (optional)
	 */
	HOST void sync_from_device(const Resource& resource = Resource{}) {
		if (device_buffer_) {
			device_buffer_->copy_to_host(values_.data(), values_.size());
		}
	}

	/**
	 * @brief Get device data pointer (for kernel use)
	 * @param resource Resource to get pointer from (optional)
	 * @return Raw device pointer
	 */
	HOST T* get_device_pointer(const Resource& resource = Resource{}) const {
		auto& buffer = get_device_buffer(resource);
		return buffer.data();
	}

  private:
	/**
	 * @brief Get default resource for device allocation
	 */
	HOST Resource get_default_resource() const {
		// Use default device resource (same logic as Buffer.h)
		return Resource{};
	}
#endif

	/*=======================*\
	|  DEVICE-COMPATIBLE OPS  |
	\*=======================*/
	// Note: All methods below are device-compatible (HOST DEVICE)

	/*===================*\
	|  INDEX OPERATIONS   |
	\*===================*/

	/**
	 * @brief Convert 3D indices to linear index
	 */
	HOST DEVICE idx_t index(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return iz + iy * nz() + ix * ny() * nz();
	}

	/**
	 * @brief Convert linear index to 3D indices using device-safe IndexList
	 */
	HOST DEVICE IndexList<idx_t, 3> index_to_ijk(idx_t linear_index) const {
		IndexList<idx_t, 3> result;
		result.add(linear_index / (ny() * nz())); // ix
		result.add((linear_index / nz()) % ny()); // iy
		result.add(linear_index % nz());		  // iz
		return result;
	}

	/**
	 * @brief Get position in space from linear index
	 */
	HOST DEVICE Vector3 get_position(idx_t linear_index) const {
		auto ijk = index_to_ijk(linear_index);
		Vector3 grid_coords(static_cast<T>(ijk[0]), static_cast<T>(ijk[1]), static_cast<T>(ijk[2]));
		return basis().transform(grid_coords) + origin();
	}

	/**
	 * @brief Transform world position to grid coordinates
	 */
	HOST DEVICE Vector3 transform_to_grid(const Vector3& world_pos) const {
		return basis_inverse().transform(world_pos - origin());
	}

	/**
	 * @brief Transform grid coordinates to world position
	 */
	HOST DEVICE Vector3 transform_to_world(const Vector3& grid_pos) const {
		return basis().transform(grid_pos) + origin();
	}

	/**
	 * @brief Check if position is within grid bounds
	 */
	HOST DEVICE bool in_bounds(const Vector3& world_pos) const {
		Vector3 grid_pos = transform_to_grid(world_pos);
		return grid_pos.x >= 0 && grid_pos.x < nx() && grid_pos.y >= 0 && grid_pos.y < ny() &&
			   grid_pos.z >= 0 && grid_pos.z < nz();
	}

	/**
	 * @brief Check if position is within interpolation bounds (with safety margin)
	 */
	HOST DEVICE bool in_interpolation_bounds(const Vector3& world_pos) const {
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
#ifdef HOST_GUARD
	HOST void zero() {
		std::fill(values_.begin(), values_.end(), T{0});
	}
#endif

/**
 * @brief Add constant to all grid values (host-only)
 */
#ifdef HOST_GUARD
	HOST void shift(T value) {
		for (auto& v : values_)
			v += value;
	}
#endif

/**
 * @brief Scale all grid values by constant (host-only)
 */
#ifdef HOST_GUARD
	HOST void scale(T factor) {
		for (auto& v : values_)
			v *= factor;
	}
#endif

	/**
	 * @brief Compute mean of all grid values
	 */
#ifdef HOST_GUARD
	HOST T mean() const {
		T sum = T{0};
		for (const auto& v : values_)
			sum += v;
		return sum / static_cast<T>(values_.size());
	}
#endif

/**
 * @brief Element-wise multiplication with another grid (host-only)
 */
#ifdef HOST_GUARD
	HOST BaseGrid& multiply(const BaseGrid& other) {
		if (values_.size() != other.values_.size()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"BaseGrid::multiply: Size mismatch (%zu vs %zu)",
							values_.size(),
							other.values_.size());
		}

		for (idx_t i = 0; i < values_.size(); ++i) {
			values_[i] *= other.values_[i];
		}
		return *this;
	}
#endif

	/**
	 * @brief Get system box as Matrix3
	 */
	HOST DEVICE Matrix3 get_box() const {
		return Matrix3(static_cast<T>(nx()) * basis().ex(),
					   static_cast<T>(ny()) * basis().ey(),
					   static_cast<T>(nz()) * basis().ez());
	}

	/**
	 * @brief Get grid center position
	 */
	HOST DEVICE Vector3 get_center() const {
		Vector3 center_grid(T(0.5) * nx(), T(0.5) * ny(), T(0.5) * nz());
		return transform_to_world(center_grid);
	}

	/**
	 * @brief Get volume of single grid cell
	 */
	HOST DEVICE T get_cell_volume() const {
		return std::abs(basis().det());
	}

	/**
	 * @brief Get total grid volume
	 */
	HOST DEVICE T get_total_volume() const {
		return get_cell_volume() * static_cast<T>(size());
	}

	/*===========================*\
	|  INTERPOLATION & SAMPLING   |
	\*===========================*/

	/**
	 * @brief Interpolate value at world position using trilinear interpolation (host-only)
	 * @param world_pos World position to interpolate at
	 * @return Interpolated value
	 */
	HOST T interpolate(const Vector3& world_pos) const {
#ifdef HOST_GUARD
		return interpolate_grid_point(values_.data(),
									  world_pos,
									  config_.origin,
									  basis_inv_,
									  config_.dimensions);
#else
		return T{0}; // Should not be called on device
#endif
	}

	/**
	 * @brief Interpolate value at world position using explicit data pointer (device-compatible)
	 * @param data_ptr Pointer to grid data (host or device)
	 * @param world_pos World position to interpolate at
	 * @return Interpolated value
	 */
	HOST DEVICE T interpolate(const T* data_ptr, const Vector3& world_pos) const {
		return interpolate_grid_point(data_ptr,
									  world_pos,
									  config_.origin,
									  basis_inv_,
									  config_.dimensions);
	}

	/**
	 * @brief Get value at nearest grid point (host-only)
	 * @param world_pos World position to sample at
	 * @return Nearest grid point value
	 */
	HOST T get_value(const Vector3& world_pos) const {
#ifdef HOST_GUARD
		return get_value_nearest(values_.data(),
								 world_pos,
								 config_.origin,
								 basis_inv_,
								 config_.dimensions);
#else
		return T{0}; // Should not be called on device
#endif
	}

	/**
	 * @brief Get value at nearest grid point using explicit data pointer (device-compatible)
	 * @param data_ptr Pointer to grid data (host or device)
	 * @param world_pos World position to sample at
	 * @return Nearest grid point value
	 */
	HOST DEVICE T get_value(const T* data_ptr, const Vector3& world_pos) const {
		return get_value_nearest(data_ptr,
								 world_pos,
								 config_.origin,
								 basis_inv_,
								 config_.dimensions);
	}

	/**
	 * @brief Compute gradient at world position using finite differences (host-only)
	 * @param world_pos World position to compute gradient at
	 * @return Gradient vector
	 */
	HOST Vector3 compute_gradient(const Vector3& world_pos) const {
#ifdef HOST_GUARD
		return compute_gradient<T>(values_.data(),
								   world_pos,
								   config_.origin,
								   config_.basis,
								   basis_inv_,
								   config_.dimensions);
#else
		return Vector3{T{0}, T{0}, T{0}}; // Should not be called on device
#endif
	}

	/**
	 * @brief Compute gradient at world position using explicit data pointer (device-compatible)
	 * @param data_ptr Pointer to grid data (host or device)
	 * @param world_pos World position to compute gradient at
	 * @return Gradient vector
	 */
	HOST DEVICE Vector3 compute_gradient(const T* data_ptr, const Vector3& world_pos) const {
		return compute_gradient<T>(data_ptr,
								   world_pos,
								   config_.origin,
								   config_.basis,
								   basis_inv_,
								   config_.dimensions);
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

		HOST DEVICE constexpr NeighborList() {
			// Initialize to zero
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					for (int k = 0; k < 3; ++k) {
						v[i][j][k] = U{0};
					}
				}
			}
		}

		HOST DEVICE constexpr U& operator()(int i, int j, int k) noexcept {
			return v[i + 1][j + 1][k + 1];
		}

		HOST DEVICE constexpr const U& operator()(int i, int j, int k) const noexcept {
			return v[i + 1][j + 1][k + 1];
		}

		HOST DEVICE constexpr U& center() noexcept {
			return v[1][1][1];
		}
		HOST DEVICE constexpr const U& center() const noexcept {
			return v[1][1][1];
		}
	};

	/**
	 * @brief Get 3x3x3 neighborhood around a grid point (host-only)
	 */
	HOST NeighborList<T> get_neighbor_list(idx_t ix, idx_t iy, idx_t iz) const {
#ifdef HOST_GUARD
		LOGINFO("get_neighbor_list: calling get_neighbor_list_from_grid");
		return get_neighbor_list_from_grid(values_.data(), ix, iy, iz, config_.dimensions);
#else
		return NeighborList<T>{}; // Should not be called on device
#endif
	}

	/**
	 * @brief Get 3x3x3 neighborhood using explicit data pointer (device-compatible)
	 */
	HOST DEVICE NeighborList<T>
	get_neighbor_list(const T* data_ptr, idx_t ix, idx_t iy, idx_t iz) const {
		return get_neighbor_list_from_grid(data_ptr, ix, iy, iz, config_.dimensions);
	}

	/**
	 * @brief Get neighbor value at relative offset (host-only)
	 */
	HOST T get_neighbor(idx_t ix, idx_t iy, idx_t iz, int di, int dj, int dk) const {
#ifdef HOST_GUARD
		LOGINFO("get_neighbor on HOST: ix={}, iy={}, iz={}, di={}, dj={}, dk={}",
				ix,
				iy,
				iz,
				di,
				dj,
				dk);
		return get_neighbor_from_grid(values_.data(), ix, iy, iz, di, dj, dk, config_.dimensions);
#else
		return T{0}; // Should not be called on device
#endif
	}

	/**
	 * @brief Get neighbor value using explicit data pointer (device-compatible)
	 */
	HOST DEVICE T
	get_neighbor(const T* data_ptr, idx_t ix, idx_t iy, idx_t iz, int di, int dj, int dk) const {
		return get_neighbor_from_grid(data_ptr, ix, iy, iz, di, dj, dk, config_.dimensions);
	}

	/**
	 * @brief Get neighbor at world position with offset (host-only)
	 */
	HOST T get_neighbor_at_position(const Vector3& world_pos, int di, int dj, int dk) const {
		const Vector3 grid_pos = transform_to_grid(world_pos);

		const idx_t ix = static_cast<idx_t>(grid_pos.x);
		const idx_t iy = static_cast<idx_t>(grid_pos.y);
		const idx_t iz = static_cast<idx_t>(grid_pos.z);

		return get_neighbor(ix, iy, iz, di, dj, dk);
	}

	/**
	 * @brief Get neighbor at world position using explicit data pointer (device-compatible)
	 */
	HOST DEVICE T get_neighbor_at_position(const T* data_ptr,
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

	/*================================*\
	|  DEVICE-SAFE HELPER FUNCTIONS   |
	\*================================*/

  public:
	/**
	 * @brief Device-safe interpolation function (CUDA/SYCL compatible)
	 */
	template<typename U>
	friend HOST DEVICE U interpolate_grid_point(const U* grid_values,
												const Vector3_t<U>& world_pos,
												const Vector3_t<U>& origin,
												const Matrix3_t<U>& basis_inv,
												const Vector3_t<idx_t>& dimensions);

	/**
	 * @brief Device-safe nearest point sampling
	 */
	template<typename U>
	friend HOST DEVICE U get_value_nearest(const U* grid_values,
										   const Vector3_t<U>& world_pos,
										   const Vector3_t<U>& origin,
										   const Matrix3_t<U>& basis_inv,
										   const Vector3_t<idx_t>& dimensions);

	/**
	 * @brief Device-safe gradient computation
	 */
	template<typename U>
	friend HOST DEVICE Vector3_t<U> compute_gradient(const U* grid_values,
													 const Vector3_t<U>& world_pos,
													 const Vector3_t<U>& origin,
													 const Matrix3_t<U>& basis,
													 const Matrix3_t<U>& basis_inv,
													 const Vector3_t<idx_t>& dimensions);

	/**
	 * @brief Device-safe neighbor list extraction
	 */
	template<typename U>
	friend HOST DEVICE auto get_neighbor_list_from_grid(const U* grid_values,
														idx_t ix,
														idx_t iy,
														idx_t iz,
														const Vector3_t<idx_t>& dimensions);

	/**
	 * @brief Device-safe single neighbor access
	 */
	template<typename U>
	friend HOST DEVICE U get_neighbor_from_grid(const U* grid_values,
												idx_t ix,
												idx_t iy,
												idx_t iz,
												int di,
												int dj,
												int dk,
												const Vector3_t<idx_t>& dimensions);

	/**
	 * @brief Device-safe index wrapping for periodic boundaries
	 */
	friend HOST DEVICE idx_t wrap_index(int index, idx_t size);

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

	/*===================*\
	|  I/O OPERATIONS     |
	\*===================*/

	// I/O operations have been moved to IO/dxreader.h for better separation
	// Use DXReader::write_grid(), DXReader::read_from_file(), etc.
	// Example usage:
	//   DXReader::write_grid(my_grid, "output.dx");
	//   auto grid = DXReader::read_from_file<float>("input.dx");
};

// Type aliases for common usage
using BaseGridf = BaseGrid<float>;
using BaseGridd = BaseGrid<double>;

/*==========================*\
|  DEVICE-SAFE FUNCTIONS     |
\*==========================*/

/**
 * @brief Device-safe interpolation function (CUDA/SYCL compatible)
 */
template<typename T>
HOST DEVICE T interpolate_grid_point(const T* grid_values,
									 const Vector3_t<T>& world_pos,
									 const Vector3_t<T>& origin,
									 const Matrix3_t<T>& basis_inv,
									 const Vector3_t<idx_t>& dimensions) {
	// Transform world position to grid coordinates
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	const idx_t nx = dimensions.x;
	const idx_t ny = dimensions.y;
	const idx_t nz = dimensions.z;

	// Check bounds
	if (grid_pos.x < 0 || grid_pos.x >= nx || grid_pos.y < 0 || grid_pos.y >= ny ||
		grid_pos.z < 0 || grid_pos.z >= nz) {
		return T{0};
	}

	// Linear interpolation
	const idx_t i0 = static_cast<idx_t>(grid_pos.x);
	const idx_t j0 = static_cast<idx_t>(grid_pos.y);
	const idx_t k0 = static_cast<idx_t>(grid_pos.z);

	const idx_t i1 = i0 + 1;
	const idx_t j1 = j0 + 1;
	const idx_t k1 = k0 + 1;

	const T fx = grid_pos.x - static_cast<T>(i0);
	const T fy = grid_pos.y - static_cast<T>(j0);
	const T fz = grid_pos.z - static_cast<T>(k0);

	// Get grid indices using consistent indexing
	const idx_t idx000 = k0 + j0 * nz + i0 * ny * nz;
	const idx_t idx001 = k1 + j0 * nz + i0 * ny * nz;
	const idx_t idx010 = k0 + j1 * nz + i0 * ny * nz;
	const idx_t idx011 = k1 + j1 * nz + i0 * ny * nz;
	const idx_t idx100 = k0 + j0 * nz + i1 * ny * nz;
	const idx_t idx101 = k1 + j0 * nz + i1 * ny * nz;
	const idx_t idx110 = k0 + j1 * nz + i1 * ny * nz;
	const idx_t idx111 = k1 + j1 * nz + i1 * ny * nz;

	// Trilinear interpolation
	const T v000 = grid_values[idx000];
	const T v001 = grid_values[idx001];
	const T v010 = grid_values[idx010];
	const T v011 = grid_values[idx011];
	const T v100 = grid_values[idx100];
	const T v101 = grid_values[idx101];
	const T v110 = grid_values[idx110];
	const T v111 = grid_values[idx111];

	const T v00 = v000 * (T{1} - fx) + v100 * fx;
	const T v01 = v001 * (T{1} - fx) + v101 * fx;
	const T v10 = v010 * (T{1} - fx) + v110 * fx;
	const T v11 = v011 * (T{1} - fx) + v111 * fx;

	const T v0 = v00 * (T{1} - fy) + v10 * fy;
	const T v1 = v01 * (T{1} - fy) + v11 * fy;

	return v0 * (T{1} - fz) + v1 * fz;
}

/**
 * @brief Get value at nearest grid point (device-safe)
 */
template<typename T>
HOST DEVICE T get_value_nearest(const T* grid_values,
								const Vector3_t<T>& world_pos,
								const Vector3_t<T>& origin,
								const Matrix3_t<T>& basis_inv,
								const Vector3_t<idx_t>& dimensions) {
	// Transform to grid coordinates
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	// Find nearest grid point
	const idx_t ix = static_cast<idx_t>(grid_pos.x + T{0.5});
	const idx_t iy = static_cast<idx_t>(grid_pos.y + T{0.5});
	const idx_t iz = static_cast<idx_t>(grid_pos.z + T{0.5});

	// Wrap for periodic boundaries (simple modulo)
	const idx_t wrapped_ix = ix % dimensions.x;
	const idx_t wrapped_iy = iy % dimensions.y;
	const idx_t wrapped_iz = iz % dimensions.z;

	const idx_t linear_idx =
		wrapped_iz + wrapped_iy * dimensions.z + wrapped_ix * dimensions.y * dimensions.z;
	return grid_values[linear_idx];
}

/**
 * @brief Compute gradient at a point using finite differences (device-safe)
 */
template<typename T>
HOST DEVICE Vector3_t<T> compute_gradient(const T* grid_values,
										  const Vector3_t<T>& world_pos,
										  const Vector3_t<T>& origin,
										  const Matrix3_t<T>& basis,
										  const Matrix3_t<T>& basis_inv,
										  const Vector3_t<idx_t>& dimensions) {
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	const idx_t nx = dimensions.x;
	const idx_t ny = dimensions.y;
	const idx_t nz = dimensions.z;

	// Check if we're in bounds for gradient calculation
	if (grid_pos.x < 1 || grid_pos.x >= nx - 1 || grid_pos.y < 1 || grid_pos.y >= ny - 1 ||
		grid_pos.z < 1 || grid_pos.z >= nz - 1) {
		return Vector3_t<T>{T{0}, T{0}, T{0}};
	}

	const idx_t i = static_cast<idx_t>(grid_pos.x);
	const idx_t j = static_cast<idx_t>(grid_pos.y);
	const idx_t k = static_cast<idx_t>(grid_pos.z);

	// Central differences
	const idx_t idx_xp = k + j * nz + (i + 1) * ny * nz;
	const idx_t idx_xm = k + j * nz + (i - 1) * ny * nz;
	const idx_t idx_yp = k + (j + 1) * nz + i * ny * nz;
	const idx_t idx_ym = k + (j - 1) * nz + i * ny * nz;
	const idx_t idx_zp = (k + 1) + j * nz + i * ny * nz;
	const idx_t idx_zm = (k - 1) + j * nz + i * ny * nz;

	const T dx_grid = (grid_values[idx_xp] - grid_values[idx_xm]) / T{2};
	const T dy_grid = (grid_values[idx_yp] - grid_values[idx_ym]) / T{2};
	const T dz_grid = (grid_values[idx_zp] - grid_values[idx_zm]) / T{2};

	// Transform gradient from grid space to world space
	const Vector3_t<T> grad_grid(dx_grid, dy_grid, dz_grid);
	return basis_inv.transpose().transform(grad_grid);
}
/**
 * @brief Device-safe index wrapping for periodic boundaries
 */
HOST DEVICE inline idx_t wrap_index(int index, idx_t size) {
	if (index < 0) {
		return static_cast<idx_t>(index + static_cast<int>(size) *
											  ((-index / static_cast<int>(size)) + 1)) %
			   size;
	}
	return static_cast<idx_t>(index) % size;
}
/**
 * @brief Device-safe neighbor list extraction (works with raw pointers)
 */
template<typename T>
HOST DEVICE auto get_neighbor_list_from_grid(const T* grid_values,
											 idx_t ix,
											 idx_t iy,
											 idx_t iz,
											 const Vector3_t<idx_t>& dimensions) {
	typename BaseGrid<T>::template NeighborList<T> neighbors;

	const idx_t nx_val = dimensions.x;
	const idx_t ny_val = dimensions.y;
	const idx_t nz_val = dimensions.z;

#if (!defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__))
	LOGINFO("get_neighbor_list_from_grid: ix={}, iy={}, iz={}, nx={}, ny={}, nz={}",
			ix,
			iy,
			iz,
			nx_val,
			ny_val,
			nz_val);
#endif

	// Fill 3x3x3 neighborhood
	for (int di = -1; di <= 1; ++di) {
		for (int dj = -1; dj <= 1; ++dj) {
			for (int dk = -1; dk <= 1; ++dk) {
				// Calculate neighbor indices with wrapping
				const idx_t ni = wrap_index(static_cast<int>(ix) + di, nx_val);
				const idx_t nj = wrap_index(static_cast<int>(iy) + dj, ny_val);
				const idx_t nk = wrap_index(static_cast<int>(iz) + dk, nz_val);

				const idx_t neighbor_idx = nk + nj * nz_val + ni * ny_val * nz_val;
				const T value = grid_values[neighbor_idx];
				neighbors.v[di + 1][dj + 1][dk + 1] = value;

#if (!defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__))
				if (di == -1 && dj == 0 && dk == 0) {
					LOGINFO(
						"left_neighbor: di={}, dj={}, dk={}, ni={}, nj={}, nk={}, idx={}, value={}",
						di,
						dj,
						dk,
						ni,
						nj,
						nk,
						neighbor_idx,
						value);
				}
#endif
			}
		}
	}

	return neighbors;
}

/**
 * @brief Device-safe single neighbor access (works with raw pointers)
 */
template<typename T>
HOST DEVICE T get_neighbor_from_grid(const T* grid_values,
									 idx_t ix,
									 idx_t iy,
									 idx_t iz,
									 int di,
									 int dj,
									 int dk,
									 const Vector3_t<idx_t>& dimensions) {
	const idx_t ni = wrap_index(static_cast<int>(ix) + di, dimensions.x);
	const idx_t nj = wrap_index(static_cast<int>(iy) + dj, dimensions.y);
	const idx_t nk = wrap_index(static_cast<int>(iz) + dk, dimensions.z);

	const idx_t neighbor_idx = nk + nj * dimensions.z + ni * dimensions.y * dimensions.z;
	return grid_values[neighbor_idx];
}

} // namespace ARBD
