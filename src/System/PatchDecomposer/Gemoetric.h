#pragma once

#include "../PatchDecomposer.h"
#include "Backend/Buffer.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Geometric patch decomposition for systems with specific shapes
 *
 * Uses geometric information (e.g., membrane boundaries) to
 * create domain partitions that respect system geometry.
 */
/**
 * @brief Geometric patch decomposition for systems with specific shapes
 *
 * Uses geometric information (e.g., membrane boundaries) to
 * create domain partitions that respect system geometry.
 */
class GeometricPatchDecomposer : public PatchDecomposer {
  public:
	GeometricPatchDecomposer() {
		type_ = DecomposerType::Geometric;
	}

	/**
	 * @brief Geometry-aware decomposition for complex systems
	 *
	 * Creates patches that respect:
	 * - Membrane and interface boundaries
	 * - Geometric constraints from grids
	 * - Material property discontinuities
	 * - System-specific spatial features
	 */
	DecompositionPlan decompose(SimSystem& system, SystemState& state) override;

	std::string get_name() const override {
		return "Geometric Decomposer (Boundary-Aware)";
	}

  private:
	/**
	 * @brief Detect geometric features in particle data
	 */
	struct GeometricFeature {
		Vector3 normal;
		Vector3 center;
		float thickness;
		int feature_type; // membrane, interface, etc.
	};

	std::vector<GeometricFeature>
	detect_geometric_features(const HostParticleData& particles,
							  const std::vector<ParticleType>& types) const;

	/**
	 * @brief Create patches that respect geometric boundaries
	 */
	std::vector<Vector3>
	create_geometry_aware_boundaries(const std::vector<GeometricFeature>& features,
									 const Vector3& system_min,
									 const Vector3& system_max) const;
};
} // namespace ARBD
