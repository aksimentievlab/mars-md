/*********************************************************************
 * @file  BaseGrid.h
 *
 * @brief Declaration of templated BaseGrid class.
 *********************************************************************/
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

// ARBD2 includes
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Math/Matrix3.h"
#include "Math/Types.h"
#include "Math/Vector3.h"

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

namespace ARBD {

enum class BoundaryCondition { dirichlet, neumann, periodic };

enum class InterpolationOrder { linear = 1, cubic = 3 };

constexpr size_t STRLEN = 512;

class BaseGrid {
  private:
	// Grid parameters
	int nx_, ny_, nz_;
	size_t size_;
	Matrix3 basis_;
	Matrix3 basisInv_;
	Vector3 origin_;

	// Data storage with DeviceBuffer
	std::unique_ptr<DeviceBuffer<float>> val_buffer_;
	Resource resource_;

	void init();

  public:
	/*===============================*\
	| CONSTRUCTORS, DESTRUCTORS, I/O |
	\===============================*/

	// Default constructor
	BaseGrid(const Resource& resource = Resource::Local());

	// The most obvious of constructors
	BaseGrid(const Matrix3& basis0,
			 const Vector3& origin0,
			 int nx0,
			 int ny0,
			 int nz0,
			 const Resource& resource = Resource::Local());

	// Make an orthogonal grid given the box dimensions and resolution
	BaseGrid(const Vector3& box, float dx, const Resource& resource = Resource::Local());

	// The box gives the system geometry, grid point numbers define resolution
	BaseGrid(const Matrix3& box,
			 int nx0,
			 int ny0,
			 int nz0,
			 const Resource& resource = Resource::Local());

	// Box with origin and approximate resolution
	BaseGrid(const Matrix3& box,
			 const Vector3& origin0,
			 float dx,
			 const Resource& resource = Resource::Local());

	// Box with approximate resolution (auto-calculated grid spacing)
	BaseGrid(const Matrix3& box, float dx, const Resource& resource = Resource::Local());

	// Copy constructor
	BaseGrid(const BaseGrid& g);

	// Copy constructor with different resolution
	BaseGrid(const BaseGrid& g, int nx0, int ny0, int nz0);

	// Read from file constructor
	BaseGrid(const char* fileName, const Resource& resource = Resource::Local());

	// Move constructor
	BaseGrid(BaseGrid&& other) noexcept;

	// Destructor
	~BaseGrid() = default;

	/*===============================*\
	|         OPERATORS              |
	\===============================*/

	// Assignment operator
	BaseGrid& operator=(const BaseGrid& g);

	// Move assignment
	BaseGrid& operator=(BaseGrid&& other) noexcept;

	// Multiplication with another grid
	BaseGrid& mult(const BaseGrid& g);

	/*===============================*\
	|      GRID PROPERTIES           |
	\===============================*/

	// Accessors
	int nx() const {
		return nx_;
	}
	int ny() const {
		return ny_;
	}
	int nz() const {
		return nz_;
	}
	size_t size() const {
		return size_;
	}

	const Matrix3& basis() const {
		return basis_;
	}
	const Matrix3& basisInv() const {
		return basisInv_;
	}
	const Vector3& origin() const {
		return origin_;
	}

	const Resource& resource() const {
		return resource_;
	}

	// Get the system box
	Matrix3 getBox() const {
		return Matrix3(basis_.ex() * nx_, basis_.ey() * ny_, basis_.ez() * nz_);
	}

	/*===============================*\
	|      MEMORY MANAGEMENT         |
	\===============================*/

	// Get raw data pointer (device memory)
	float* data() {
		return val_buffer_->data();
	}
	const float* data() const {
		return val_buffer_->data();
	}

	// Zero the grid
	BACKEND::Event zero();

	// Copy data from another grid
	BACKEND::Event copy_from(const BaseGrid& other);

	/*===============================*\
	|      INDEX/POSITION CONVERSION |
	\===============================*/

	// Convert from grid indices to linear index
	HOST DEVICE size_t getIndex(int i, int j, int k) const {
		return size_t(i) * ny_ * nz_ + size_t(j) * nz_ + size_t(k);
	}

	// Convert from linear index to grid indices
	HOST DEVICE Vector3_t<int> getIndices(size_t idx) const {
		Vector3_t<int> result;
		result.z = int(idx % nz_);
		result.y = int((idx / nz_) % ny_);
		result.x = int(idx / (ny_ * nz_));
		return result;
	}

	// Get world position from grid indices
	HOST DEVICE Vector3 getPosition(int i, int j, int k) const {
		return origin_ + basis_.transform(Vector3(float(i), float(j), float(k)));
	}

	// Get world position from linear index
	HOST DEVICE Vector3 getPosition(size_t idx) const {
		auto ijk = getIndices(idx);
		return getPosition(ijk.x, ijk.y, ijk.z);
	}

	// Get grid indices from world position
	HOST DEVICE Vector3_t<int> getGridIndices(const Vector3& pos) const {
		Vector3 local = basisInv_.transform(pos - origin_);
		return Vector3_t<int>(int(std::round(local.x)),
							  int(std::round(local.y)),
							  int(std::round(local.z)));
	}

	/*===============================*\
	|         INTERPOLATION          |
	\===============================*/

	// Linear interpolation at world position
	float interpolatePotential(const Vector3& pos) const;

	// Trilinear interpolation with boundary conditions
	template<BoundaryCondition BC = BoundaryCondition::periodic>
	float interpolateTrilinear(const Vector3& pos) const {
		return dispatch_interpolate_trilinear<BC>(pos);
	}

	// Compute force (negative gradient) at position
	template<BoundaryCondition BC = BoundaryCondition::periodic>
	Vector3 interpolateForce(const Vector3& pos) const {
		return dispatch_interpolate_force<BC>(pos);
	}

	/*===============================*\
	|           FILE I/O             |
	\===============================*/

	// Write grid to file
	void write(const char* fileName) const;

	// Read grid from file
	void read(const char* fileName);

	/*===============================*\
	|        GRID OPERATIONS         |
	\===============================*/

	// Add scalar to all grid points
	BACKEND::Event add_scalar(float value);

	// Multiply all grid points by scalar
	BACKEND::Event multiply_scalar(float value);

	// Add another grid
	BACKEND::Event add_grid(const BaseGrid& other);

	// Element-wise multiplication with another grid
	BACKEND::Event multiply_grid(const BaseGrid& other);

	/*===============================*\
	|      BACKEND-SPECIFIC METHODS  |
	\===============================*/

#ifdef USE_SYCL
	// SYCL-specific operations
	BACKEND::Event sycl_zero_grid();
	BACKEND::Event sycl_copy_from(const BaseGrid& other);
	BACKEND::Event sycl_add_scalar(float value);
	BACKEND::Event sycl_multiply_scalar(float value);
	BACKEND::Event sycl_add_grid(const BaseGrid& other);
	BACKEND::Event sycl_multiply_grid(const BaseGrid& other);

	// SYCL interpolation
	template<BoundaryCondition BC = BoundaryCondition::periodic>
	float sycl_interpolate_trilinear(const Vector3& pos) const;

	template<BoundaryCondition BC = BoundaryCondition::periodic>
	Vector3 sycl_interpolate_force(const Vector3& pos) const;
#endif

#ifdef USE_CUDA
	// CUDA-specific operations
	BACKEND::Event cuda_zero_grid();
	BACKEND::Event cuda_copy_from(const BaseGrid& other);
	BACKEND::Event cuda_add_scalar(float value);
	BACKEND::Event cuda_multiply_scalar(float value);
	BACKEND::Event cuda_add_grid(const BaseGrid& other);
	BACKEND::Event cuda_multiply_grid(const BaseGrid& other);

	// CUDA interpolation
	template<BoundaryCondition BC = BoundaryCondition::periodic>
	float cuda_interpolate_trilinear(const Vector3& pos) const;

	template<BoundaryCondition BC = BoundaryCondition::periodic>
	Vector3 cuda_interpolate_force(const Vector3& pos) const;

	// CUDA texture memory operations (optional optimization)
	void cuda_bind_texture();
	void cuda_unbind_texture();
#endif

#ifdef USE_METAL
	// Metal-specific operations
	BACKEND::Event metal_zero_grid();
	BACKEND::Event metal_copy_from(const BaseGrid& other);
	BACKEND::Event metal_add_scalar(float value);
	BACKEND::Event metal_multiply_scalar(float value);
	BACKEND::Event metal_add_grid(const BaseGrid& other);
	BACKEND::Event metal_multiply_grid(const BaseGrid& other);

	// Metal interpolation
	template<BoundaryCondition BC = BoundaryCondition::periodic>
	float metal_interpolate_trilinear(const Vector3& pos) const;

	template<BoundaryCondition BC = BoundaryCondition::periodic>
	Vector3 metal_interpolate_force(const Vector3& pos) const;

	// Metal buffer management
	void metal_sync_buffers();
#endif

  private:
	/*===============================*\
	|        HELPER FUNCTIONS        |
	\===============================*/

	// Check if indices are within bounds
	HOST DEVICE bool isInBounds(int i, int j, int k) const {
		return i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_;
	}

	// Apply boundary conditions
	template<BoundaryCondition BC>
	HOST DEVICE Vector3_t<int> applyBoundaryConditions(int i, int j, int k) const;

	// Bilinear interpolation helpers
	HOST DEVICE float getValue(int i, int j, int k) const;
	HOST DEVICE float getValueSafe(int i, int j, int k, BoundaryCondition bc) const;

	/*===============================*\
	|      BACKEND DISPATCH          |
	\===============================*/

	// Internal dispatch methods that route to appropriate backend
	BACKEND::Event dispatch_zero();
	BACKEND::Event dispatch_copy_from(const BaseGrid& other);
	BACKEND::Event dispatch_add_scalar(float value);
	BACKEND::Event dispatch_multiply_scalar(float value);
	BACKEND::Event dispatch_add_grid(const BaseGrid& other);
	BACKEND::Event dispatch_multiply_grid(const BaseGrid& other);

	template<BoundaryCondition BC>
	float dispatch_interpolate_trilinear(const Vector3& pos) const;

	template<BoundaryCondition BC>
	Vector3 dispatch_interpolate_force(const Vector3& pos) const;

	/*===============================*\
	|      CPU IMPLEMENTATIONS       |
	\===============================*/

	// CPU fallback implementations
	template<BoundaryCondition BC>
	float cpu_interpolate_trilinear(const Vector3& pos, const float* data) const;

	template<BoundaryCondition BC>
	Vector3 cpu_interpolate_force(const Vector3& pos, const float* data) const;
};

/*===============================*\
|     TEMPLATE IMPLEMENTATIONS   |
\===============================*/

template<BoundaryCondition BC>
HOST DEVICE Vector3_t<int> BaseGrid::applyBoundaryConditions(int i, int j, int k) const {
	Vector3_t<int> result{i, j, k};

	if constexpr (BC == BoundaryCondition::periodic) {
		// Periodic boundary conditions
		result.x = ((i % nx_) + nx_) % nx_;
		result.y = ((j % ny_) + ny_) % ny_;
		result.z = ((k % nz_) + nz_) % nz_;
	} else if constexpr (BC == BoundaryCondition::dirichlet) {
		// Clamp to boundaries (Dirichlet: values at boundary are fixed)
		result.x = std::max(0, std::min(nx_ - 1, i));
		result.y = std::max(0, std::min(ny_ - 1, j));
		result.z = std::max(0, std::min(nz_ - 1, k));
	} else if constexpr (BC == BoundaryCondition::neumann) {
		// Reflect at boundaries (Neumann: derivatives at boundary are fixed)
		if (i < 0)
			result.x = -i;
		else if (i >= nx_)
			result.x = 2 * nx_ - 1 - i;

		if (j < 0)
			result.y = -j;
		else if (j >= ny_)
			result.y = 2 * ny_ - 1 - j;

		if (k < 0)
			result.z = -k;
		else if (k >= nz_)
			result.z = 2 * nz_ - 1 - k;
	}

	return result;
}

} // namespace ARBD