#pragma once

#include "ARBDLogger.h"
#include "Header.h"
#include "Types/Types.h"

namespace ARBD {
HOST DEVICE void __inline__ BDIntegrate() {
	// std::cout << "Computes::BDIntegrate_inline" << std::endl;
	LOGINFO("Integrator::BDIntegrate\n");
};

HOST DEVICE void __inline__ BDIntegrate(Vector3* RESTRICT pos,
										const Vector3* const RESTRICT force,
										const int& idx,
										float& root_Dt,
										Vector3& normal_sample_3D) {
	LOGINFO("Integrator::BDIntegrate\n");
	pos[idx] = pos[idx] + force[idx] * root_Dt + normal_sample_3D;
};
} // namespace ARBD
