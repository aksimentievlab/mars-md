/**
  @brief Python bindings for SimManager - skeleton (construction, init/run,
  and the pending-data setters used by the pure-Python configuration path).
  SystemState/PatchManager/RigidBodyManager stay unexposed, matching
  pysystem.cpp/pyobjects.cpp's existing "runtime state is internal" stance.

  Nonbonded interactions are deliberately NOT staged here: unlike particles/
  bonded-topology/rigid-bodies (per-run data SimManager caches until init()),
  NonBondedInteractions lives on SimSystem itself (see SimSystem::
  get_nonbonded_interactions() in pysystem.cpp) since it's tied to particle
  type definitions, which are also SimSystem-owned.

  A manager is bound to exactly one SimSystem for its lifetime. Configure that
  system directly, then stage the per-run data; init() consumes it.

  @example (in Python):
  ```python
  >>> from marsmd import SimSystem, SimManager, Resource
  >>> sys = SimSystem([Resource(0)])
  >>> mgr = SimManager(sys)
  >>> mgr.stage_particles(my_particle_list)
  >>> mgr.stage_bonded_interactions(my_bonded_interactions)
  >>> mgr.stage_rigid_bodies(my_rigid_body_list)
  >>> mgr.init()
  >>> mgr.run()
  ```
*/
#include "SimManager.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace MARS;

void init_pysim(nb::module_& m) {
	nb::class_<SimManager>(m, "SimManager")
		.def(nb::init<SimSystem&>(),
			 nb::arg("sys"),
			 nb::keep_alive<1, 2>(),
			 "Bind to one SimSystem for this manager's lifetime. Configure that "
			 "system directly, stage_*() the initial data, then init().")
		.def("init",
			 &SimManager::init,
			 "Set up domain decomposition, output writers, IMD, and initial conditions")
		.def("stage_particles",
			 &SimManager::set_initial_particles,
			 nb::arg("particles"),
			 "Stage initial particles (ParticleIO list), consumed by init()")
		.def("stage_bonded_interactions",
			 &SimManager::set_bonded_interactions,
			 nb::arg("bonded_interactions"),
			 "Stage parsed bonds/angles/dihedrals/exclusions, consumed by init()")
		.def("stage_rigid_bodies",
			 &SimManager::set_initial_rigid_bodies,
			 nb::arg("bodies"),
			 "Stage initial rigid bodies (RigidBody list), consumed by init()")
		.def("run",
			 &SimManager::run,
			 nb::call_guard<nb::gil_scoped_release>(),
			 "Run the main simulation loop (releases the GIL - this blocks for the "
			 "full simulation)")
		.def("write_psf",
			 &SimManager::write_psf,
			 nb::arg("path") = "",
			 "Write a PSF matching the DCD's atom order "
			 "([regular][attached][cosmetic]). Call after init(). "
			 "Defaults to '<outputName>.psf'.")
		.def("write_pdb",
			 &SimManager::write_pdb,
			 nb::arg("path") = "",
			 "Write a PDB snapshot of the current positions, same atom order and "
			 "topology as write_psf(). Defaults to '<outputName>.pdb'.")
		.def("__repr__", [](const SimManager&) { return std::string("SimManager()"); });
}
