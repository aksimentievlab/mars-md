/**
 * @file ParticlePatchAssign.h
 * @brief Kernels for converting each patch between AoS and SoA
 */

#pragma once

#include "Backend/KernelConfig.h"
#include "Backend/Resource.h"
#include "ParticlePatch.h"

namespace ARBD {
Event launch_particle_patch_assign(const Resource& resource, const ParticlePatch& patch) {

	// TODO: Implement
	return Event();
};

} // namespace ARBD
