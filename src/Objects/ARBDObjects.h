#pragma once
#include "Interactions/BondedInteraction.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include "SimParam.h"
#include "Types/BaseGrid.h"
#include <vector>

namespace ARBD {
/** 
* @brief ARBDObjects is a struct that contains all the objects in the simulation.
* HOST side only.
*/
struct ARBDObjects {
	std::vector<RigidBodyType> rigid_body_types{};
	std::vector<ParticleType> particle_types{};
	std::vector<RigidBody> rigid_bodies{};
	std::vector<Particle> particles{};
	std::vector<Bond> bonds{};
	std::vector<Angle> angles{};
	std::vector<Dihedral> dihedrals{};
	std::vector<Exclude> exclusions{};
	std::vector<Restraint> restraints{};
	std::vector<NonbondedInteraction> nb_interactions{};
	std::vector<BondedInteraction> bonded_interactions{};
	std::vector<std::string> tabulated_file_names{};
	std::vector<std::string> grid_file_names{};
	std::vector<BaseGrid<float>> part_grid_dictionary{};
};

} // namespace ARBD
