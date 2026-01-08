#pragma once
#include "Objects/ParticleProperties.h"
#include "Types/Types.h"
#include <string>
#include <vector>

ARBD::HostParticleData
create_test_particles(int count, const std::string& pattern = "linear", float box_size = 100.0f);
// =============================================================================
std::vector<ARBD::ParticleType> create_test_particle_types(int type_count = 2,
														   int particles_per_type = 50);
