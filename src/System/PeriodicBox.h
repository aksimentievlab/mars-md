#pragma once
/**
 * @file PeriodicBox.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Lightweight device-safe periodic box for minimum image convention
 */

#include "Header.h"
#include "Types/Math.h"
#include "Types/Types.h"

namespace ARBD {

enum class Periodicity { AllPeriodic = 3, TwoDimensional = 2, OneDimensional = 1, Open = 0 };

class PeriodicBox {
  public:
	HOST DEVICE PeriodicBox()
		: box_size_(Vector3(arbd_real(0.0), arbd_real(0.0), arbd_real(0.0))),
		  periodic_{false, false, false}, triclinic_(false) {}

	HOST DEVICE PeriodicBox(const Vector3& box_size,
							bool periodic_x = true,
							bool periodic_y = true,
							bool periodic_z = true)
		: box_size_(box_size), periodic_{periodic_x, periodic_y, periodic_z}, triclinic_(false) {

		// Orthogonal basis
		v1_ = Vector3(box_size.x, arbd_real(0.0), arbd_real(0.0));
		v2_ = Vector3(arbd_real(0.0), box_size.y, arbd_real(0.0));
		v3_ = Vector3(arbd_real(0.0), arbd_real(0.0), box_size.z);
		basis_ = Matrix3(v1_, v2_, v3_);
		inv_v1x_ = (v1_.x > arbd_real(0.0)) ? arbd_real(1.0) / v1_.x : arbd_real(0.0);
		inv_v2y_ = (v2_.y > arbd_real(0.0)) ? arbd_real(1.0) / v2_.y : arbd_real(0.0);
		inv_v3z_ = (v3_.z > arbd_real(0.0)) ? arbd_real(1.0) / v3_.z : arbd_real(0.0);
	}

	HOST DEVICE PeriodicBox(const Vector3& box_size, Periodicity periodicity)
		: box_size_(box_size), triclinic_(false) {
		switch (periodicity) {
		case Periodicity::AllPeriodic:
			periodic_[0] = true;
			periodic_[1] = true;
			periodic_[2] = true;
			break;
		case Periodicity::TwoDimensional:
			periodic_[0] = true;
			periodic_[1] = true;
			periodic_[2] = false;
			break;
		case Periodicity::OneDimensional:
			periodic_[0] = true;
			periodic_[1] = false;
			periodic_[2] = false;
			break;
		case Periodicity::Open:
		default:
			periodic_[0] = false;
			periodic_[1] = false;
			periodic_[2] = false;
			break;
		}
		// Orthogonal basis
		v1_ = Vector3(box_size.x, arbd_real(0.0), arbd_real(0.0));
		v2_ = Vector3(arbd_real(0.0), box_size.y, arbd_real(0.0));
		v3_ = Vector3(arbd_real(0.0), arbd_real(0.0), box_size.z);
		basis_ = Matrix3(v1_, v2_, v3_);

		inv_v1x_ = (v1_.x > arbd_real(0.0)) ? arbd_real(1.0) / v1_.x : arbd_real(0.0);
		inv_v2y_ = (v2_.y > arbd_real(0.0)) ? arbd_real(1.0) / v2_.y : arbd_real(0.0);
		inv_v3z_ = (v3_.z > arbd_real(0.0)) ? arbd_real(1.0) / v3_.z : arbd_real(0.0);
	}

#ifdef HOST_GUARD

	HOST void set_basis(const Vector3& basis1, const Vector3& basis2, const Vector3& basis3) {
		// 1. Force arbitrary basis vectors into a standard lower-triangular matrix.
		// This rotates the system so basis1 aligns with X, and basis2 lies in the XY plane.
		arbd_real v1x = basis1.length();
		arbd_real v1y = arbd_real(0.0);
		arbd_real v1z = arbd_real(0.0);

		arbd_real v2x = (v1x > arbd_real(0.0)) ? basis1.dot(basis2) / v1x : arbd_real(0.0);
		arbd_real v2y_sq = basis2.dot(basis2) - v2x * v2x;

		// Max(0) guards against floating point errors giving tiny negative numbers
		arbd_real v2y = (v2y_sq > arbd_real(0.0)) ? ARBD::math::sqrt(v2y_sq) : arbd_real(0.0);
		arbd_real v2z = arbd_real(0.0);

		arbd_real v3x = (v1x > arbd_real(0.0)) ? basis1.dot(basis3) / v1x : arbd_real(0.0);
		arbd_real v3y =
			(v2y > arbd_real(0.0)) ? (basis2.dot(basis3) - v2x * v3x) / v2y : arbd_real(0.0);
		arbd_real v3z_sq = basis3.dot(basis3) - v3x * v3x - v3y * v3y;
		arbd_real v3z = (v3z_sq > arbd_real(0.0)) ? ARBD::math::sqrt(v3z_sq) : arbd_real(0.0);

		v1_ = Vector3(v1x, v1y, v1z);
		v2_ = Vector3(v2x, v2y, v2z);
		v3_ = Vector3(v3x, v3y, v3z);
		basis_ = Matrix3(v1_, v2_, v3_);

		// 2. The box size for grids/spatial hashing must be the perpendicular extents
		// (the diagonal of the lower triangular matrix), NOT the vector lengths.
		box_size_ = Vector3(v1x, v2y, v3z);
		inv_v1x_ = (v1_.x > arbd_real(0.0)) ? arbd_real(1.0) / v1_.x : arbd_real(0.0);
		inv_v2y_ = (v2_.y > arbd_real(0.0)) ? arbd_real(1.0) / v2_.y : arbd_real(0.0);
		inv_v3z_ = (v3_.z > arbd_real(0.0)) ? arbd_real(1.0) / v3_.z : arbd_real(0.0);

		// 3. Robustly check for orthogonality using a small tolerance
		constexpr arbd_real TOL = arbd_real(1e-5);
		if (std::abs(v2x) < TOL && std::abs(v3x) < TOL && std::abs(v3y) < TOL) {
			triclinic_ = false;
			// Snap to zero to prevent floating point drift in the fast-path
			v1_ = Vector3(v1x, arbd_real(0.0), arbd_real(0.0));
			v2_ = Vector3(arbd_real(0.0), v2y, arbd_real(0.0));
			v3_ = Vector3(arbd_real(0.0), arbd_real(0.0), v3z);
			basis_ = Matrix3(v1_, v2_, v3_);
		} else {
			triclinic_ = true;
		}
	}

	HOST PeriodicBox(const Vector3& origin,
					 const Vector3& basis1,
					 const Vector3& basis2,
					 const Vector3& basis3,
					 bool periodic1 = true,
					 bool periodic2 = true,
					 bool periodic3 = true)
		: origin_(origin), periodic_{periodic1, periodic2, periodic3} {
		set_basis(basis1, basis2, basis3);
	}
#endif

	/**
	 * @brief Apply minimum image convention to vector difference
	 */
	HOST DEVICE Vector3 wrap_diff(const Vector3& dr) const {
		if (!triclinic_) {
			// Fast-path for orthogonal boxes
			Vector3 wrapped_dr = dr;
			if (periodic_[0])
				wrapped_dr.x = wrap_diff_scalar(dr.x, box_size_.x);
			if (periodic_[1])
				wrapped_dr.y = wrap_diff_scalar(dr.y, box_size_.y);
			if (periodic_[2])
				wrapped_dr.z = wrap_diff_scalar(dr.z, box_size_.z);
			return wrapped_dr;
		}

		// Cascading shifts for lower-triangular triclinic boxes
		Vector3 w = dr;

		if (periodic_[2] && v3_.z > arbd_real(0.0)) {
			arbd_real sz = math::round(w.z * inv_v3z_);
			w.x -= sz * v3_.x;
			w.y -= sz * v3_.y;
			w.z -= sz * v3_.z;
		}
		if (periodic_[1] && v2_.y > arbd_real(0.0)) {
			arbd_real sy = math::round(w.y * inv_v2y_);
			w.x -= sy * v2_.x;
			w.y -= sy * v2_.y;
		}
		if (periodic_[0] && v1_.x > arbd_real(0.0)) {
			arbd_real sx = math::round(w.x * inv_v1x_);
			w.x -= sx * v1_.x;
		}

		return w;
	}

	/**
	 * @brief Wrap an absolute position into the primary image
	 */
	HOST DEVICE Vector3 wrap(const Vector3& r) const {
		if (!triclinic_) {
			// Fast-path for orthogonal boxes
			Vector3 wrapped = r;
			if (periodic_[0])
				wrapped.x = wrap_scalar(r.x, origin_.x, box_size_.x);
			if (periodic_[1])
				wrapped.y = wrap_scalar(r.y, origin_.y, box_size_.y);
			if (periodic_[2])
				wrapped.z = wrap_scalar(r.z, origin_.z, box_size_.z);
			return wrapped;
		}

		// 1. Shift by origin
		Vector3 p(r.x - origin_.x, r.y - origin_.y, r.z - origin_.z);

		// 2. Transform to fractional (scaled) coordinates (h^-1 * p)
		arbd_real sz = (v3_.z > arbd_real(0.0)) ? p.z * inv_v3z_ : arbd_real(0.0);
		arbd_real sy = (v2_.y > arbd_real(0.0)) ? (p.y - sz * v3_.y) * inv_v2y_ : arbd_real(0.0);
		arbd_real sx =
			(v1_.x > arbd_real(0.0)) ? (p.x - sy * v2_.x - sz * v3_.x) * inv_v1x_ : arbd_real(0.0);

		// 3. Apply periodicity in scaled space [0, 1)
		if (periodic_[2])
			sz = sz - math::floor(sz);
		if (periodic_[1])
			sy = sy - math::floor(sy);
		if (periodic_[0])
			sx = sx - math::floor(sx);

		// 4. Transform back to Cartesian (h * s) + origin
		Vector3 wrapped;
		wrapped.x = sx * v1_.x + sy * v2_.x + sz * v3_.x + origin_.x;
		wrapped.y = sy * v2_.y + sz * v3_.y + origin_.y;
		wrapped.z = sz * v3_.z + origin_.z;

		return wrapped;
	}

	HOST DEVICE const Vector3& get_box_size() const {
		return box_size_;
	}
	HOST DEVICE const bool* get_periodicity() const {
		return periodic_;
	}
	HOST const Vector3& get_origin() const {
		return origin_;
	}
	HOST const Matrix3& get_basis() const {
		return basis_;
	}
	HOST DEVICE bool is_periodic(int dim) const {
		return (dim >= 0 && dim < 3) ? periodic_[dim] : false;
	}
	HOST DEVICE bool is_triclinic() const {
		return triclinic_;
	}
	HOST DEVICE arbd_real get_volume() const {
		return box_size_.x * box_size_.y * box_size_.z;
	}

	HOST void set_box_size(const Vector3& box_size) {
		box_size_ = box_size;
		v1_ = Vector3(box_size.x, arbd_real(0.0), arbd_real(0.0));
		v2_ = Vector3(arbd_real(0.0), box_size.y, arbd_real(0.0));
		v3_ = Vector3(arbd_real(0.0), arbd_real(0.0), box_size.z);
		basis_ = Matrix3(v1_, v2_, v3_);
		triclinic_ = false;

		inv_v1x_ = (v1_.x > arbd_real(0.0)) ? arbd_real(1.0) / v1_.x : arbd_real(0.0);
		inv_v2y_ = (v2_.y > arbd_real(0.0)) ? arbd_real(1.0) / v2_.y : arbd_real(0.0);
		inv_v3z_ = (v3_.z > arbd_real(0.0)) ? arbd_real(1.0) / v3_.z : arbd_real(0.0);
	}

	HOST void set_periodicity(bool px, bool py, bool pz) {
		periodic_[0] = px;
		periodic_[1] = py;
		periodic_[2] = pz;
		if (px && py && pz)
			periodicity_ = Periodicity::AllPeriodic;
		else if (px + py + pz == 2)
			periodicity_ = Periodicity::TwoDimensional;
		else if (px || py || pz)
			periodicity_ = Periodicity::OneDimensional;
		else
			periodicity_ = Periodicity::Open;
	}
	/**
	 * @brief Check if a point lies within the primary simulation box [origin, origin + box)
	 * @param r Absolute position vector
	 * @return True if the point is inside the box, false otherwise
	 */
	HOST DEVICE bool is_inside(const Vector3& r) const {
		Vector3 p = r - origin_;

		if (!triclinic_) {
			if (p.x < arbd_real(0.0) || p.y < arbd_real(0.0) || p.z < arbd_real(0.0))
				return false;
			if (p.x >= box_size_.x || p.y >= box_size_.y || p.z >= box_size_.z)
				return false;
			return true;
		}

		arbd_real sz = (v3_.z > arbd_real(0.0)) ? p.z * inv_v3z_ : arbd_real(0.0);
		if (sz < arbd_real(0.0) || sz >= arbd_real(1.0))
			return false;

		arbd_real sy = (v2_.y > arbd_real(0.0)) ? (p.y - sz * v3_.y) * inv_v2y_ : arbd_real(0.0);
		if (sy < arbd_real(0.0) || sy >= arbd_real(1.0))
			return false;

		arbd_real sx =
			(v1_.x > arbd_real(0.0)) ? (p.x - sy * v2_.x - sz * v3_.x) * inv_v1x_ : arbd_real(0.0);
		if (sx < arbd_real(0.0) || sx >= arbd_real(1.0))
			return false;

		return true;
	}

	HOST DEVICE bool is_in_physical_box(const Vector3& r) const {
		Vector3 p = get_fractional_coordinates(r);
		if (p.x < halo_margins_.x || p.y < halo_margins_.y || p.z < halo_margins_.z)
			return false;
		if (p.x >= arbd_real(1.0) - halo_margins_.x || p.y >= arbd_real(1.0) - halo_margins_.y ||
			p.z >= arbd_real(1.0) - halo_margins_.z)
			return false;
		return true;
	}

	HOST DEVICE bool is_ghost(const Vector3& r) const {
		short g = check_halo_regions(r);
		return g != 0;
	}

	HOST void set_origin(const Vector3& origin) {
		origin_ = origin;
	}

	HOST Periodicity get_periodicity_type() const {
		return periodicity_;
	}
	HOST DEVICE int get_periodic_num() const {
		return static_cast<int>(periodicity_);
	}

	/**
	 * @brief Convert an absolute Cartesian position to fractional coordinates [0, 1).
	 */
	HOST DEVICE Vector3 get_fractional_coordinates(const Vector3& r) const {
		Vector3 p = r - origin_;
		if (!triclinic_) {
			return Vector3((box_size_.x > arbd_real(0.0)) ? p.x / box_size_.x : arbd_real(0.0),
						   (box_size_.y > arbd_real(0.0)) ? p.y / box_size_.y : arbd_real(0.0),
						   (box_size_.z > arbd_real(0.0)) ? p.z / box_size_.z : arbd_real(0.0));
		}
		arbd_real sz = (v3_.z > arbd_real(0.0)) ? p.z * inv_v3z_ : arbd_real(0.0);
		arbd_real sy = (v2_.y > arbd_real(0.0)) ? (p.y - sz * v3_.y) * inv_v2y_ : arbd_real(0.0);
		arbd_real sx =
			(v1_.x > arbd_real(0.0)) ? (p.x - sy * v2_.x - sz * v3_.x) * inv_v1x_ : arbd_real(0.0);
		return Vector3(sx, sy, sz);
	}

	/**
	 * @brief Convert fractional coordinates [0, 1) back to absolute Cartesian space.
	 */
	HOST DEVICE Vector3 get_cartesian_coordinates(const Vector3& frac) const {
		if (!triclinic_) {
			return Vector3(frac.x * box_size_.x + origin_.x,
						   frac.y * box_size_.y + origin_.y,
						   frac.z * box_size_.z + origin_.z);
		}
		return Vector3(frac.x * v1_.x + frac.y * v2_.x + frac.z * v3_.x + origin_.x,
					   frac.y * v2_.y + frac.z * v3_.y + origin_.y,
					   frac.z * v3_.z + origin_.z);
	}

	/**
	 * @brief Precompute and cache the fractional halo margins based on the interaction cutoff.
	 * @param rc The maximum Cartesian cutoff radius for pair interactions
	 */
	HOST void set_halo_margin(arbd_real rc) {
		halo_margins_ = Vector3();
		if (!triclinic_) {
			// Orthogonal fast-path: rc / L
			if (!periodic_[0]) {
				halo_margins_.x = rc * inv_v1x_;
			}
			if (!periodic_[1]) {
				halo_margins_.y = rc * inv_v2y_;
			}
			if (!periodic_[2]) {
				halo_margins_.z = rc * inv_v3z_;
			}
			return;
		}

		// Triclinic path: Calculate the row norms of the inverse basis matrix
		arbd_real Gxx = inv_v1x_;

		arbd_real Gyx = -v2_.x * inv_v1x_ * inv_v2y_;
		arbd_real Gyy = inv_v2y_;

		arbd_real Gzx = (v2_.x * v3_.y - v2_.y * v3_.x) * inv_v1x_ * inv_v2y_ * inv_v3z_;
		arbd_real Gzy = -v3_.y * inv_v2y_ * inv_v3z_;
		arbd_real Gzz = inv_v3z_;

		// Max fractional extents of a sphere of radius rc
		if (!periodic_[0]) {
			halo_margins_.x = rc * Gxx;
		}
		if (!periodic_[1]) {
			halo_margins_.y = rc * math::sqrt(Gyx * Gyx + Gyy * Gyy);
		}
		if (!periodic_[2]) {
			halo_margins_.z = rc * math::sqrt(Gzx * Gzx + Gzy * Gzy + Gzz * Gzz);
		}
	}

	/**
	 * @brief Retrieve the precomputed fractional halo margins.
	 */
	HOST DEVICE const Vector3& get_halo_margins() const {
		return halo_margins_;
	}

	HOST DEVICE short check_migration_direction(const Vector3& r) const {
		Vector3 frac = get_fractional_coordinates(r);

		if (frac.x < arbd_real(0.0))
			return 0;
		if (frac.x >= arbd_real(1.0))
			return 1;

		if (frac.y < arbd_real(0.0))
			return 2;
		if (frac.y >= arbd_real(1.0))
			return 3;

		if (frac.z < arbd_real(0.0))
			return 4;
		if (frac.z >= arbd_real(1.0))
			return 5;

		return -1; // Safely inside
	}
	/**
	 * @brief Determine which halo buffers a particle should be copied to.
	 * @param r Absolute Cartesian position of the particle
	 * @return A bitmask indicating the required halo transfers:
	 *         bit 0: X-   bit 1: X+
	 *         bit 2: Y-   bit 3: Y+
	 *         bit 4: Z-   bit 5: Z+
	 */
	HOST DEVICE uint8_t check_halo_regions(const Vector3& r) const {
		Vector3 frac = get_fractional_coordinates(r);
		uint8_t flags = 0;

		// We use `else if` because a particle cannot simultaneously be on
		// both the left and right walls (assuming rc < 0.5 * box_length)
		if (frac.x < halo_margins_.x) {
			flags |= (1 << 0);
		} else if (frac.x >= arbd_real(1.0) - halo_margins_.x) {
			flags |= (1 << 1);
		}

		if (frac.y < halo_margins_.y) {
			flags |= (1 << 2);
		} else if (frac.y >= arbd_real(1.0) - halo_margins_.y) {
			flags |= (1 << 3);
		}

		if (frac.z < halo_margins_.z) {
			flags |= (1 << 4);
		} else if (frac.z >= arbd_real(1.0) - halo_margins_.z) {
			flags |= (1 << 5);
		}

		return flags;
	}

  private:
	HOST DEVICE static inline arbd_real wrap_scalar(arbd_real x, arbd_real o, arbd_real l) {
		if (l <= arbd_real(0.0))
			return x;
		const arbd_real rel = x - o;
		int image = int(math::floor(rel * (arbd_real(1.0) / l)));
		return x - image * l;
	}

	HOST DEVICE static inline arbd_real wrap_diff_scalar(arbd_real x, arbd_real l) {
		if (l <= arbd_real(0.0))
			return x;
		int image = int(math::floor(x * (arbd_real(1.0) / l)));
		x -= image * l;
		if (x >= arbd_real(0.5) * l)
			x -= l;
		return x;
	}

	Vector3 box_size_;
	bool periodic_[3];
	Periodicity periodicity_{Periodicity::Open};
	bool triclinic_{false};

	Vector3 origin_{0, 0, 0};

	// Explicitly storing the lower-triangular basis vectors ensures
	// fast, memory-coalesced access on the GPU without Matrix3 overhead.
	Vector3 v1_{5000.0, 0, 0};
	Vector3 v2_{0, 5000.0, 0};
	Vector3 v3_{0, 0, 5000.0};

	Vector3 halo_margins_{0, 0, 0};

	arbd_real inv_v1x_{0.0002};
	arbd_real inv_v2y_{0.0002};
	arbd_real inv_v3z_{0.0002};

	Matrix3 basis_;
};
} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::PeriodicBox> : std::true_type {};
#endif
