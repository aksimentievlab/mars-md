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

  @note Example usage (in Python):
  ```python
  >>> from arbd2v import SimSystem, ConfigParser, SimManager, Resource, ResourceType
  >>> sys = SimSystem([Resource(ResourceType.CUDA, 0)])
  >>> parser = ConfigParser(sys, "circovirus.bd")
  >>> mgr = SimManager(sys, parser)
  >>> mgr.init()
  >>> mgr.run()
  >>> mgr.get_total_time()

  # Pure-Python configuration, no ConfigParser:
  >>> mgr = SimManager(sys)
  >>> mgr.send_particles(my_particle_list)
  >>> mgr.send_bonded_interactions(my_bonded_interactions)
  >>> mgr.send_rigid_bodies(my_rigid_body_list)
  >>> mgr.init()
  >>> mgr.run()
  ```
*/
#include "IO/ConfigParser.h"
#include "SimManager.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

void init_pysim(py::module_& m) {
	py::class_<SimManager>(m, "SimManager")
		.def(py::init<SimSystem&>(),
			 py::arg("sys"),
			 py::keep_alive<1, 2>(),
			 "Construct without a ConfigParser - configure via send_particles/"
			 "send_bonded_interactions/send_rigid_bodies before init()")
		.def(py::init<SimSystem&, const ConfigParser&>(),
			 py::arg("sys"),
			 py::arg("parser"),
			 py::keep_alive<1, 2>(),
			 "Construct from a loaded ConfigParser")
		.def("init",
			 &SimManager::init,
			 "Set up domain decomposition, output writers, IMD, and initial conditions")
		.def("load_config",
			 &SimManager::load_config,
			 py::arg("parser"),
			 "Load particles/bonded interactions/rigid bodies from a ConfigParser")
		.def("stage_particles",
			 &SimManager::set_initial_particles,
			 py::arg("particles"),
			 "Stage initial particles (ParticleIO list), consumed by init()")
		.def("stage_bonded_interactions",
			 &SimManager::set_bonded_interactions,
			 py::arg("bonded_interactions"),
			 "Stage parsed bonds/angles/dihedrals/exclusions, consumed by init()")
		.def("stage_rigid_bodies",
			 &SimManager::set_initial_rigid_bodies,
			 py::arg("bodies"),
			 "Stage initial rigid bodies (RigidBody list), consumed by init()")
		.def("run",
			 &SimManager::run,
			 py::call_guard<py::gil_scoped_release>(),
			 "Run the main simulation loop (releases the GIL - this blocks for the "
			 "full simulation)")
		.def("write_psf",
			 &SimManager::write_psf,
			 py::arg("path") = "",
			 "Write a PSF matching the DCD's atom order "
			 "([regular][attached][cosmetic]). Call after init(). "
			 "Defaults to '<outputName>.psf'.")
		.def("write_pdb",
			 &SimManager::write_pdb,
			 py::arg("path") = "",
			 "Write a PDB snapshot of the current positions, same atom order and "
			 "topology as write_psf(). Defaults to '<outputName>.pdb'.")
		.def("get_total_time", &SimManager::get_total_time)
		.def("get_io_time", &SimManager::get_io_time)
		.def("get_energy_time", &SimManager::get_energy_time)
		.def("__repr__", [](const SimManager& mgr) {
			return "SimManager(total_time=" + std::to_string(mgr.get_total_time()) + "s)";
		});
}
