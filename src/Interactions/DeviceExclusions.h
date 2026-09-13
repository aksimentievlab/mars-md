#pragma once
/*********************************************************************
 * @file  DeviceExclusions.h
 *
 * @brief Device view of the exclusion CSR, shared by pairlist builders.
 *
 * Builders consult this at emit time so excluded pairs never enter the
 * neighbor list and downstream nonbonded kernels need no exclusion test.
 *********************************************************************/

#include "Header.h"
#include "Types/Types.h"

namespace MARS {

/**
 * @brief Borrowed device view of the per-particle exclusion CSR.
 *
 * Storage is owned by DeviceBondedInteractions; the pointers must stay valid
 * until the next pairlist build. A default-constructed view excludes nothing.
 */
struct ExclusionView {
	DEVICE_PTR(const int) __restrict__ offsets{nullptr};  ///< size num_particles + 1
	DEVICE_PTR(const int) __restrict__ excluded{nullptr}; ///< excluded partners, concatenated
	idx_t num_particles{0};								  ///< particles covered by @c offsets
	/// @brief Per-particle rigid-body id, -1 when unattached. Null disables the test.
	DEVICE_PTR(const int) __restrict__ rigid_body_id{nullptr};

	/**
	 * @brief Start of @p particle's exclusion row, or 0 if it has none.
	 * @note Each exclusion is stored under both partners, so one endpoint's row
	 *       is sufficient. Hoist this out of a neighbour loop: it depends only
	 *       on the thread's own particle.
	 */
	DEVICE int row_begin(int particle) const {
		return (particle >= 0 && static_cast<idx_t>(particle) < num_particles) ? offsets[particle]
																			   : 0;
	}

	/// @brief One past the end of @p particle's exclusion row.
	DEVICE int row_end(int particle) const {
		return (particle >= 0 && static_cast<idx_t>(particle) < num_particles)
				   ? offsets[particle + 1]
				   : 0;
	}

	/// @brief True when @p partner appears in the row [@p begin, @p end).
	DEVICE bool row_contains(int begin, int end, int partner) const {
		for (int e = begin; e < end; ++e) {
			if (excluded[e] == partner)
				return true;
		}
		return false;
	}

	/**
	 * @brief Rigid body @p particle is attached to, or -1 if none.
	 * @note Hoist this out of a neighbour loop; it depends only on the thread's
	 *       own particle. Returns -1 when the test is disabled.
	 */
	DEVICE int body_of(int particle) const {
		return rigid_body_id ? rigid_body_id[particle] : -1;
	}

	/// @brief True when @p partner is attached to the same body as @p body, which must come
	///        from @c body_of. A negative @p body never matches.
	DEVICE bool same_body(int body, int partner) const {
		return body >= 0 && rigid_body_id[partner] == body;
	}

	/// @brief True when @p a and @p b must not interact nonbonded.
	DEVICE bool is_excluded(int a, int b) const {
		return same_body(body_of(a), b) || row_contains(row_begin(a), row_end(a), b);
	}
};

} // namespace MARS

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::ExclusionView> : std::true_type {};
#endif
