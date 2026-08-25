#include "Patch.h"

// Skleton only for future reaction-diffusion wire up.
namespace ARBD {

class Reactions;

class ReactionManager {
  public:
	ReactionManager(SimSystem& sys,
					Patch& patch,
					std::vector<std::unique_ptr<Reactions>>& Reactions,
					DeviceBuffer<ParticleView>& birth_buffer,
					NeighborList& neighbor_list)
		: sys_(sys), patch_(patch), Reactions_(Reactions), birth_buffer_(birth_buffer),
		  neighbor_list_(neighbor_list) {}
	~ReactionManager() = default;

	/**
	 * @brief Execute the reaction
	 * @todo:
	 * Step1: Run Reaction Kernel, Sets FLAG_DEAD on some particles, Creates new particles in a
	 * temporary "Birth Buffer".
   * Step2: Remove Dead Particles, Compacts DeviceParticle array, Generates 'permutation_map'.
   * Step3: Update Interactions (Fix Indices), If patch has topology changes, update topology of
  interactions.
	* Step4: Add New Particles, Appends from "Birth Buffer" to end
  of DeviceParticle. (No index shifting for existing particles, so safe).
   * Step5: Rebuild Neighbor Lists, Mandatory after moving particles.

	 */
	void reaction() {
		auto perm_map = patch.compact_particles(); // step 1 and 2
		// 3. Update Interactions (Fix Indices)
		if (patch.has_topology_changes()) {
			for (auto& interaction : Reactions_) {
				interaction->update_topology(perm_map);
			}
		}
		// 4. Add New Particles
		//    Appends from "Birth Buffer" to end of DeviceParticle.
		//    (No index shifting for existing particles, so safe).
		patch.append_from_buffer(birth_buffer);

		// 5. Rebuild Neighbor Lists
		//    Mandatory after moving particles.
		neighbor_list_.force_rebuild();
	}

	void update_topology() {
		for (auto& interaction : Reactions_) {
			interaction->update_topology();
		}
	}

  private:
	SimSystem& sys_;
	Patch& patch_;
	std::vector<std::unique_ptr<Reactions>>& Reactions_;
	DeviceBuffer<ParticleView>& birth_buffer_;
	NeighborList& neighbor_list_;
};

} // namespace ARBD
