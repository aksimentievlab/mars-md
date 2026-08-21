#pragma once
#include "Header.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

/**
 * @brief This file is the basic definitions using for bonded and nonbonded interactions used in
 * kernels.
 */

// Include STL headers for host-only class (before namespace to avoid conflicts)
#if (!defined(__CUDACC__) || \
	 (!defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)))
#include <string>
#include <unordered_map>
#include <vector>
#endif

namespace ARBD {

/**
 * @brief Scalar force magnitude and energy from a potential evaluation
 *
 * The named fields must stay the same type as `float2`'s components: they
 * alias the same storage, and `float2` is `Vec2<arbd_real>`.
 */
struct ScalarForceEnergy {
	union {
		float2 fe{arbd_real(0), arbd_real(0)};
		struct {
			arbd_real force_magnitude;
			arbd_real energy;
		};
	};
};

struct CalcDistance {
	Vector3 r_ij;		 // vector between two particles
	arbd_real distance;	 // distance between two particles
	Vector3 unit_vector; // unit vector in the direction of the vector

	DEVICE static CalcDistance compute(const Vector3& from, const Vector3& to, const PeriodicBox* pbox) {
		CalcDistance dist;
		dist.r_ij = pbox->wrap_diff(to - from);
		dist.distance = dist.r_ij.length();
		if (dist.distance > arbd_real(1e-6)) {
			dist.unit_vector = dist.r_ij / dist.distance;
		} else {
			dist.unit_vector = Vector3(arbd_real(0));
		}
		return dist;
	}

	DEVICE static CalcDistance
	compute(const Vector3* positions, const int2& particle_indices, const PeriodicBox* pbox) {
		return compute(positions[particle_indices.x], positions[particle_indices.y], pbox);
	}
};
/** Future pybind class for new analytical pairwise potential registration */
// ============================================================================
// POTENTIAL REGISTRATION
// ============================================================================
// Host-only class - only define in host compilation
#if (!defined(__CUDACC__) || \
	 (!defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)))
class Register_Potential {
  public:
	int register_potential(const std::string& name) {
		if (m_name_to_id.find(name) == m_name_to_id.end()) {
			int new_id = m_name_to_id.size();
			m_name_to_id[name] = new_id;
		}
		return m_name_to_id[name];
	}

	int get_id(const std::string& name) const {
		return m_name_to_id.at(name);
	}

  private:
	std::unordered_map<std::string, int> m_name_to_id;
	int bodies_{2};				// 2 for pairwise, 3 for triplet, 4 for quadruplet
	std::vector<float> params_; // Parameters for the potential
};
#endif

} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ScalarForceEnergy> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::CalcDistance> : std::true_type {};
#endif
