/**
 * @file LocalInteraction.h
 * @brief Defines the LocalInteraction class and its related structures
 */

#pragma once
#include "BondedInteraction.h"
#include "Header.h"
#include "IO/Reader.h"
#include "Objects/ParticleProperties.h"

namespace ARBD {

class NonbondedInteraction {
	InteractionForm form;
	std::vector<Particle> objects;

  public:
	std::vector<Particle> get_objects() {
		return objects;
	}
	void assign_forces(Vector3* forces) {
		for (auto& object : objects) {
			object.force = forces[object.id];
		}
	}
	void assign_energies(float* energies) {
		for (auto& object : objects) {
			object.energy = energies[object.id];
		}
	}
};

} // namespace ARBD
