#pragma once
/**
 * @file PeriodicBox.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Lightweight device-safe periodic box for minimum image convention
 * @version 1.0
 * @date 2025-10-14
 *
 * @copyright Copyright (c) 2025
 */

#include "Header.h"
#include "Types/Types.h"

namespace ARBD {
enum class Periodicity { AllPeriodic = 3, TwoDimensional = 2, OneDimensional = 1, Open = 0 };

/**
 * @brief Lightweight device-safe periodic box for boundary calculations
 *
 * This class handles periodic boundary conditions without the overhead
 * of grid data storage. Designed for use with z-sort neighbor searching.
 */
class PeriodicBox {
  public:
	/**
	 * @brief Default constructor - creates non-periodic box
	 */
	HOST DEVICE PeriodicBox() : box_size_(Vector3(0, 0, 0)), periodic_{false, false, false} {}

	/**
	 * @brief Construct from box dimensions and periodicity
	 * @param box_size Box dimensions in x, y, z
	 * @param periodic_x Whether x dimension is periodic
	 * @param periodic_y Whether y dimension is periodic
	 * @param periodic_z Whether z dimension is periodic
	 */
	HOST DEVICE PeriodicBox(const Vector3& box_size,
							bool periodic_x = true,
							bool periodic_y = true,
							bool periodic_z = true)
		: box_size_(box_size), periodic_{periodic_x, periodic_y, periodic_z} {}

	/**
	 * @brief Construct from complex boundary conditions with basis vectors
	 * @param origin Origin point of the simulation box
	 * @param basis1 First basis vector
	 * @param basis2 Second basis vector
	 * @param basis3 Third basis vector
	 * @param periodic1 Whether first dimension is periodic
	 * @param periodic2 Whether second dimension is periodic
	 * @param periodic3 Whether third dimension is periodic
	 */
	HOST PeriodicBox(const Vector3& origin,
					 const Vector3& basis1,
					 const Vector3& basis2,
					 const Vector3& basis3,
					 bool periodic1 = true,
					 bool periodic2 = true,
					 bool periodic3 = true)
		: origin_(origin), basis_{basis1, basis2, basis3},
		  periodic_{periodic1, periodic2, periodic3} {
		// Calculate box size from basis vectors
		box_size_ = Vector3(basis1.length(), basis2.length(), basis3.length());
	}

	/**
	 * @brief Apply minimum image convention to vector difference
	 * @param dr Vector difference between two points
	 * @return Minimum image vector difference
	 */
	HOST DEVICE Vector3 wrap_diff(const Vector3& dr) const {
		Vector3 wrapped_dr = dr;

		// Apply wrapping for each periodic dimension
		if (periodic_[0]) {
			wrapped_dr.x = wrap_diff_scalar(dr.x, box_size_.x);
		}
		if (periodic_[1]) {
			wrapped_dr.y = wrap_diff_scalar(dr.y, box_size_.y);
		}
		if (periodic_[2]) {
			wrapped_dr.z = wrap_diff_scalar(dr.z, box_size_.z);
		}

		return wrapped_dr;
	}

	/**
	 * @brief Wrap an absolute position into the primary image
	 *
	 * Matches the legacy BaseGrid::wrap() semantics: the wrap is applied to
	 * the origin-relative coordinate, so the result lies in
	 * [origin, origin + box_size) rather than [0, box_size). A box centered
	 * on zero has origin = -box_size/2, so assuming an origin of zero would
	 * shift every particle by half a box.
	 *
	 * Non-periodic axes (and degenerate/zero extents) are left untouched.
	 *
	 * @param r Absolute position
	 * @return Position wrapped into the primary image
	 */
	HOST DEVICE Vector3 wrap(const Vector3& r) const {
		Vector3 wrapped = r;

		if (periodic_[0]) {
			wrapped.x = wrap_scalar(r.x, origin_.x, box_size_.x);
		}
		if (periodic_[1]) {
			wrapped.y = wrap_scalar(r.y, origin_.y, box_size_.y);
		}
		if (periodic_[2]) {
			wrapped.z = wrap_scalar(r.z, origin_.z, box_size_.z);
		}

		return wrapped;
	}

	/**
	 * @brief Get box size
	 */
	HOST DEVICE const Vector3& get_box_size() const {
		return box_size_;
	}

	/**
	 * @brief Get periodicity flags
	 */
	HOST DEVICE const bool* get_periodicity() const {
		return periodic_;
	}

	/**
	 * @brief Get origin point (for complex boundary conditions)
	 */
	HOST const Vector3& get_origin() const {
		return origin_;
	}

	/**
	 * @brief Get basis vectors (for complex boundary conditions)
	 */
	HOST const std::array<Vector3, 3>& get_basis() const {
		return basis_;
	}

	/**
	 * @brief Check if dimension is periodic
	 * @param dim Dimension index (0=x, 1=y, 2=z)
	 */
	HOST DEVICE bool is_periodic(int dim) const {
		return (dim >= 0 && dim < 3) ? periodic_[dim] : false;
	}

	/**
	 * @brief Get box volume
	 */
	HOST DEVICE float get_volume() const {
		return box_size_.x * box_size_.y * box_size_.z;
	}

	/**
	 * @brief Set box size (host-only)
	 *
	 * Keeps basis_ in sync: consumers such as the spatial decomposer derive
	 * the system bounds from the basis vectors, so leaving them at their
	 * default here would make the decomposition disagree with the box the
	 * integrator actually wraps into.
	 *
	 * Host-only: the box is configured once during setup (config parsing,
	 * Python bindings, SimSystem) and is thereafter read-only on the device,
	 * which receives it as a `const PeriodicBox*`. Mutating this shared,
	 * simulation-global state from a kernel would be a data race across
	 * threads. This also matches set_basis()/set_origin(), which are already
	 * host-only, and avoids touching basis_ (a std::array, whose operator[]
	 * is a host-only constexpr function) from device code.
	 */
	HOST void set_box_size(const Vector3& box_size) {
		box_size_ = box_size;
		basis_[0] = Vector3(box_size.x, 0, 0);
		basis_[1] = Vector3(0, box_size.y, 0);
		basis_[2] = Vector3(0, 0, box_size.z);
	}

	/**
	 * @brief Set periodicity for all dimensions
	 *
	 * Host-only for the same reason as set_box_size().
	 */
	HOST void set_periodicity(bool px, bool py, bool pz) {
		periodic_[0] = px;
		periodic_[1] = py;
		periodic_[2] = pz;
		if (px && py && pz) {
			periodicity_ = Periodicity::AllPeriodic;
		} else if (px + py + pz == 2) {
			periodicity_ = Periodicity::TwoDimensional;
		} else if (px || py || pz) {
			periodicity_ = Periodicity::OneDimensional;
		} else {
			periodicity_ = Periodicity::Open;
		}
	}

	/**
	 * @brief Set origin point (for complex boundary conditions)
	 */
	HOST void set_origin(const Vector3& origin) {
		origin_ = origin;
	}

	/**
	 * @brief Set basis vectors (for complex boundary conditions)
	 */
	HOST void set_basis(const Vector3& b1, const Vector3& b2, const Vector3& b3) {
		basis_ = {b1, b2, b3};
		// Update box size from new basis vectors
		box_size_ = Vector3(b1.length(), b2.length(), b3.length());
	}
	/**
	 * @brief Get periodicity
	 */
	HOST DEVICE Periodicity get_periodicity_type() const {
		return periodicity_;
	}

	HOST DEVICE int get_periodic_num() const {
		return static_cast<int>(periodicity_);
	}

  private:
	/**
	 * @brief Wrap an absolute scalar coordinate into [o, o + l)
	 * @param x Coordinate component
	 * @param o Origin component for this axis
	 * @param l Box length in that dimension
	 */
	HOST DEVICE static inline float wrap_scalar(float x, float o, float l) {
		if (l <= 0.0f)
			return x; // Non-periodic or invalid box size

		const float rel = x - o;
#ifdef USE_CUDA
		int image = int(floorf(rel / l));
#else
		int image = int(floor(rel / l));
#endif
		return x - image * l;
	}

	/**
	 * @brief Wrap scalar distance for minimum image convention
	 * @param x Distance component
	 * @param l Box length in that dimension
	 * @return Wrapped distance: -0.5*l <= x < 0.5*l
	 */
	HOST DEVICE static inline float wrap_diff_scalar(float x, float l) {
		if (l <= 0.0f)
			return x; // Non-periodic or invalid box size

#ifdef USE_CUDA
		int image = int(floorf(x / l));
#else
		int image = int(floor(x / l));
#endif
		x -= image * l;
		if (x >= 0.5f * l)
			x -= l;
		return x;
	}

	Vector3 box_size_; ///< Box dimensions (x, y, z)
	bool periodic_[3]; ///< Periodicity flags for each dimension
	Periodicity periodicity_{Periodicity::Open};

	// Complex boundary condition data (optional)
	Vector3 origin_{0, 0, 0}; ///< Origin point for complex boundary conditions
	std::array<Vector3, 3> basis_{Vector3{5000, 0, 0},
								  Vector3{0, 5000, 0},
								  Vector3{0, 0, 5000}}; ///< Basis vectors
};

} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::PeriodicBox> : std::true_type {};
#endif
