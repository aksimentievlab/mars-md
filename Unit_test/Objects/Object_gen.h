#pragma once
#include "Objects/ParticleProperties.h"
#include "Types/Types.h"
#include <string>
#include <vector>

MARS::HostParticleData
create_test_particles(int count, const std::string& pattern = "linear", float box_size = 100.0f);
// =============================================================================
std::vector<MARS::ParticleType> create_test_particle_types(int type_count = 2,
														   int particles_per_type = 50);
