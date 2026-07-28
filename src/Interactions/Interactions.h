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

struct ScalarForceEnergy {
	union {
		float2 fe{0.0f, 0.0f};
		struct {
			float force_magnitude;
			float energy;
		};
	};
};

struct CalcDistance {
	Vector3 r_ij;		 // vector between two particles
	float distance;		 // distance between two particles
	Vector3 unit_vector; // unit vector in the direction of the vector

	DEVICE static CalcDistance
	compute(const Vector3* positions, const int2& particle_indices, const PeriodicBox* pbox) {
		CalcDistance dist;
		dist.r_ij = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		dist.distance = dist.r_ij.length();
		if (dist.distance > 1e-6f) {
			dist.unit_vector = dist.r_ij / dist.distance;
		} else {
			dist.unit_vector = Vector3(0.0f);
		}
		return dist;
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
