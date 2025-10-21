#include "Objects/ARBDObjects.h"
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

// ============================================================================
// PARTICLE AND PARTICLE TYPE BINDINGS
// ============================================================================
/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import Particle
 * >>> p = Particle()
 * >>> p.id = 0
 * >>> p.type_id = 1
 * >>> p.position = [0.0, 0.0, 0.0]
 * >>> p.momentum = [1.0, 0.0, 0.0]
 * >>> p.force = [0.0, 0.0, 0.0]
 * >>> print(p)
 * Particle(id=0, type_id=1, position=[0.0, 0.0, 0.0])
 * ```
 */

void declare_particle(py::module& m) {
	py::class_<Particle>(m, "Particle")
		.def(py::init<>())
		.def_readwrite("id", &Particle::id)
		.def_readwrite("type_id", &Particle::type_id)
		.def_readwrite("position", &Particle::position)
		.def_readwrite("momentum", &Particle::momentum)
		.def_readwrite("force", &Particle::force)
		.def_readwrite("orientation", &Particle::orientation)
		.def_readwrite("energy", &Particle::energy)
		.def_readwrite("is_dummy", &Particle::is_dummy)
		.def_readwrite("has_orientation", &Particle::has_orientation)
		.def_readwrite("colvars_groups", &Particle::colvars_groups)
		.def("__repr__", [](const Particle& p) {
			return "Particle(id=" + std::to_string(p.id) +
				   ", type_id=" + std::to_string(p.type_id) +
				   ", position=" + p.position.to_string() + ")";
		});
}

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import ParticleType
 * >>> pt = ParticleType("A")
 * >>> print(pt)
 * ParticleType(name='A', id=0, num=0)
 * ```
 */
void declare_particle_type(py::module& m) {
	py::class_<ParticleType>(m, "ParticleType")
		.def(py::init<const std::string&>(), py::arg("name"))
		.def_readwrite("name", &ParticleType::name)
		.def_readwrite("id", &ParticleType::id)
		.def_readwrite("num", &ParticleType::num)
		.def_readwrite("mass", &ParticleType::mass)
		.def_readwrite("charge", &ParticleType::charge)
		.def_readwrite("radius", &ParticleType::radius)
		.def_readwrite("eps", &ParticleType::eps)
		.def_readwrite("diffusion", &ParticleType::diffusion)
		.def_readwrite("transDamping", &ParticleType::transDamping)
		.def_readwrite("mu", &ParticleType::mu)
		.def_readwrite("numPartGridFiles", &ParticleType::numPartGridFiles)
		.def_readwrite("grid_ids", &ParticleType::grid_ids)
		.def("__repr__", [](const ParticleType& pt) {
			return "ParticleType(name='" + pt.name + "', id=" + std::to_string(pt.id) +
				   ", num=" + std::to_string(pt.num) + ")";
		});
}

// ============================================================================
// RIGID BODY BINDINGS
// ============================================================================

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import RigidBody
 * >>> rb = RigidBody()
 * >>> print(rb)
 * RigidBody(id=0, type_id=0, position=[0.0, 0.0, 0.0])
 * ```
 */
void declare_rigid_body(py::module& m) {
	py::class_<RigidBody>(m, "RigidBody")
		.def(py::init<>())
		.def_readwrite("id", &RigidBody::id)
		.def_readwrite("type_id", &RigidBody::type_id)
		.def_readwrite("position", &RigidBody::position)
		.def_readwrite("orientation", &RigidBody::orientation)
		.def_readwrite("momentum", &RigidBody::momentum)
		.def_readwrite("angularMomentum", &RigidBody::angularMomentum)
		.def_readwrite("force", &RigidBody::force)
		.def_readwrite("torque", &RigidBody::torque)
		.def_readwrite("is_dummy", &RigidBody::is_dummy)
		.def_readwrite("has_orientation", &RigidBody::has_orientation)
		.def("__repr__", [](const RigidBody& rb) {
			return "RigidBody(id=" + std::to_string(rb.id) +
				   ", type_id=" + std::to_string(rb.type_id) +
				   ", position=" + rb.position.to_string() + ")";
		});
}

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import RigidBodyType
 * >>> rbt = RigidBodyType("A")
 * >>> print(rbt)
 * RigidBodyType(name='A', id=0, mass=0.0)
 * ```
 */
void declare_rigid_body_type(py::module& m) {
	py::class_<RigidBodyType>(m, "RigidBodyType")
		.def(py::init<>())
		.def_readwrite("name", &RigidBodyType::name)
		.def_readwrite("id", &RigidBodyType::id)
		.def_readwrite("mass", &RigidBodyType::mass)
		.def_readwrite("inertia", &RigidBodyType::inertia)
		.def_readwrite("transDamping", &RigidBodyType::transDamping)
		.def_readwrite("rotDamping", &RigidBodyType::rotDamping)
		.def_readwrite("transForceCoeff", &RigidBodyType::transForceCoeff)
		.def_readwrite("rotTorqueCoeff", &RigidBodyType::rotTorqueCoeff)
		.def_readwrite("rotational_diffusivity", &RigidBodyType::rotational_diffusivity)
		.def_readwrite("rotational_damping_coefficient",
					   &RigidBodyType::rotational_damping_coefficient)
		.def_readwrite("charge", &RigidBodyType::charge)
		.def_readwrite("radius", &RigidBodyType::radius)
		.def_readwrite("eps", &RigidBodyType::eps)
		.def_readwrite("diffusion", &RigidBodyType::diffusion)
		.def_readwrite("mu", &RigidBodyType::mu)
		.def_readwrite("numPartGridFiles", &RigidBodyType::numPartGridFiles)
		.def_readwrite("attached_particle", &RigidBodyType::attached_particle)
		.def("__repr__", [](const RigidBodyType& rbt) {
			return "RigidBodyType(name='" + rbt.name + "', id=" + std::to_string(rbt.id) +
				   ", mass=" + std::to_string(rbt.mass) + ")";
		});
}

// ============================================================================
// ARBD OBJECTS CONTAINER BINDINGS
// ============================================================================

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import ARBDObjects
 * >>> obj = ARBDObjects()
 * >>> print(obj)
 * ARBDObjects(particles=0, rigid_bodies=0, bonds=0, angles=0, dihedrals=0)
 * ```
 */
void declare_arbd_objects(py::module& m) {
	py::class_<ARBDObjects>(m, "ARBDObjects")
		.def(py::init<>())
		.def_readwrite("rigid_body_types", &ARBDObjects::rigid_body_types)
		.def_readwrite("particle_types", &ARBDObjects::particle_types)
		.def_readwrite("rigid_bodies", &ARBDObjects::rigid_bodies)
		.def_readwrite("particles", &ARBDObjects::particles)
		.def_readwrite("bonds", &ARBDObjects::bonds)
		.def_readwrite("angles", &ARBDObjects::angles)
		.def_readwrite("dihedrals", &ARBDObjects::dihedrals)
		.def_readwrite("exclusions", &ARBDObjects::exclusions)
		.def_readwrite("restraints", &ARBDObjects::restraints)
		.def_readwrite("nb_interactions", &ARBDObjects::nb_interactions)
		.def_readwrite("bonded_interactions", &ARBDObjects::bonded_interactions)
		.def_readwrite("tabulated_file_names", &ARBDObjects::tabulated_file_names)
		.def_readwrite("grid_file_names", &ARBDObjects::grid_file_names)
		// Helper methods
		.def("add_particle_type",
			 [](ARBDObjects& obj, const ParticleType& pt) { obj.particle_types.push_back(pt); })
		.def("add_particle",
			 [](ARBDObjects& obj, const Particle& p) { obj.particles.push_back(p); })
		.def(
			"add_rigid_body_type",
			[](ARBDObjects& obj, const RigidBodyType& rbt) { obj.rigid_body_types.push_back(rbt); })
		.def("add_rigid_body",
			 [](ARBDObjects& obj, const RigidBody& rb) { obj.rigid_bodies.push_back(rb); })
		.def("clear",
			 [](ARBDObjects& obj) {
				 obj.rigid_body_types.clear();
				 obj.particle_types.clear();
				 obj.rigid_bodies.clear();
				 obj.particles.clear();
				 obj.bonds.clear();
				 obj.angles.clear();
				 obj.dihedrals.clear();
				 obj.exclusions.clear();
				 obj.restraints.clear();
				 obj.nb_interactions.clear();
				 obj.bonded_interactions.clear();
				 obj.tabulated_file_names.clear();
				 obj.grid_file_names.clear();
			 })
		.def("__repr__", [](const ARBDObjects& obj) {
			return "ARBDObjects(particles=" + std::to_string(obj.particles.size()) +
				   ", rigid_bodies=" + std::to_string(obj.rigid_bodies.size()) +
				   ", bonds=" + std::to_string(obj.bonds.size()) +
				   ", angles=" + std::to_string(obj.angles.size()) +
				   ", dihedrals=" + std::to_string(obj.dihedrals.size()) + ")";
		});
}

// ============================================================================
// MAIN INITIALIZATION FUNCTION
// ============================================================================

void init_pyobjects(py::module_& m) {
	// Particle and particle type
	declare_particle(m);
	declare_particle_type(m);

	// Rigid body
	declare_rigid_body(m);
	declare_rigid_body_type(m);

	// ARBD objects container
	declare_arbd_objects(m);
}
