#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

namespace {

template<typename Class, Vector3 Class::*Member>
void def_vector3_property(py::class_<Class>& cls, const char* py_name) {
	cls.def_property(
		py_name,
		[](const Class& obj) -> const Vector3& { return obj.*Member; },
		[](Class& obj, const Vector3& value) { obj.*Member = value; });
}

template<typename Class, float Class::*Member>
void def_float_property(py::class_<Class>& cls, const char* py_name) {
	cls.def_property(
		py_name,
		[](const Class& obj) { return obj.*Member; },
		[](Class& obj, float value) { obj.*Member = value; });
}

} // namespace

// ============================================================================
// HOST PARTICLE IO (ConfigParser initial topology)
// ============================================================================
/**
 * @note Example usage (in Python):
 * ```python
 * >>> from pyarbd import ConfigParser, SimSystem, Resource, ResourceType, ParticleIO
 * >>> sys = SimSystem([Resource(ResourceType.CUDA, 0)])
 * >>> parser = ConfigParser(sys, "circovirus.bd")
 * >>> p = parser.get_init_particles()[0]
 * >>> p.idx
 * 0
 * >>> p.type_name
 * 'A'
 * >>> p.position
 * Vector3(0.0, 0.0, 0.0)
 * ```
 *
 * ParticleIO is the host-side particle record loaded by ConfigParser.
 * Names follow arbdmodel.core_objects.PointParticle where applicable:
 * - idx (particle index; stored as id in C++)
 * - type_name (string type label from input files)
 */
void declare_particle_io(py::module& m) {
	py::class_<ParticleIO>(m, "ParticleIO")
		.def(py::init<>())
		.def_property(
			"idx",
			[](const ParticleIO& p) { return p.id; },
			[](ParticleIO& p, int value) { p.id = value; },
			"Particle index (arbdmodel PointParticle.idx)")
		.def_readwrite("type_name", &ParticleIO::type_name)
		.def_readwrite("position", &ParticleIO::position)
		.def_readwrite("momentum", &ParticleIO::momentum)
		.def_readwrite("orientation", &ParticleIO::orientation)
		.def_readwrite("force", &ParticleIO::force)
		.def_readwrite("energy", &ParticleIO::energy)
		.def_readwrite("group_id", &ParticleIO::group_id)
		.def_readwrite("colvars_group_id", &ParticleIO::colvars_group_id)
		.def_readwrite("attached_rigid_body_id", &ParticleIO::attached_rigid_body_id)
		.def("__repr__", [](const ParticleIO& p) {
			return "ParticleIO(idx=" + std::to_string(p.id) + ", type_name='" + p.type_name +
				   "', position=" + p.position.to_string() + ")";
		});
}

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from pyarbd import ParticleType
 * >>> pt = ParticleType("A")
 * >>> pt.diffusivity = [43.5, 43.5, 43.5]
 * >>> pt.damping_coefficient = [10.0, 10.0, 10.0]
 * ```
 */
void declare_particle_type(py::module& m) {
	auto cls = py::class_<ParticleType>(m, "ParticleType")
				   .def(py::init<const std::string&>(), py::arg("name"))
				   .def_readwrite("name", &ParticleType::name)
				   .def_readwrite("id", &ParticleType::id)
				   .def_readwrite("num", &ParticleType::num)
				   .def_readwrite("mass", &ParticleType::mass)
				   .def_readwrite("charge", &ParticleType::charge)
				   .def_readwrite("radius", &ParticleType::radius)
				   .def_readwrite("eps", &ParticleType::eps)
				   .def_readwrite("mu", &ParticleType::mu)
				   .def_readwrite("pmf_scale", &ParticleType::pmf_scale)
				   .def_readwrite("pmf_scale_slope", &ParticleType::pmf_scale_slope)
				   .def_readwrite("pmf_smd_freq", &ParticleType::pmf_smd_freq)
				   .def_readwrite("pmf_grid_id", &ParticleType::pmf_grid_id)
				   .def_readwrite("diffusion_grid_id", &ParticleType::diffusion_grid_id)
				   .def_readwrite("force_grid_id", &ParticleType::force_grid_id)
				   .def("__repr__", [](const ParticleType& pt) {
					   return "ParticleType(name='" + pt.name + "', id=" + std::to_string(pt.id) +
							  ", num=" + std::to_string(pt.num) + ")";
				   });

	def_vector3_property<ParticleType, &ParticleType::diffusion>(cls, "diffusivity");
	def_vector3_property<ParticleType, &ParticleType::trans_damping>(cls, "damping_coefficient");
}

// ============================================================================
// RIGID BODY BINDINGS
// ============================================================================

void declare_rigid_body(py::module& m) {
	py::class_<RigidBody>(m, "RigidBody")
		.def(py::init<>())
		.def_property(
			"idx",
			[](const RigidBody& rb) { return rb.id; },
			[](RigidBody& rb, int value) { rb.id = value; })
		.def_readwrite("type_id", &RigidBody::type_id)
		.def_readwrite("position", &RigidBody::position)
		.def_readwrite("orientation", &RigidBody::orientation)
		.def_readwrite("momentum", &RigidBody::momentum)
		.def_property(
			"angular_momentum",
			[](const RigidBody& rb) -> const Vector3& { return rb.angularMomentum; },
			[](RigidBody& rb, const Vector3& value) { rb.angularMomentum = value; })
		.def_readwrite("force", &RigidBody::force)
		.def_readwrite("torque", &RigidBody::torque)
		.def_readwrite("is_dummy", &RigidBody::is_dummy)
		.def_readwrite("has_orientation", &RigidBody::has_orientation)
		.def("__repr__", [](const RigidBody& rb) {
			return "RigidBody(idx=" + std::to_string(rb.id) + ", type_id=" +
				   std::to_string(rb.type_id) + ", position=" + rb.position.to_string() + ")";
		});
}

void declare_rigid_body_type(py::module& m) {
	auto cls = py::class_<RigidBodyType>(m, "RigidBodyType")
				   .def(
					   py::init([](const std::string& name) {
						   RigidBodyType rbt{};
						   rbt.name = name;
						   return rbt;
					   }),
					   py::arg("name"))
				   .def_readwrite("name", &RigidBodyType::name)
				   .def_readwrite("id", &RigidBodyType::id)
				   .def_readwrite("mass", &RigidBodyType::mass)
				   .def_readwrite("charge", &RigidBodyType::charge)
				   .def_readwrite("radius", &RigidBodyType::radius)
				   .def_readwrite("eps", &RigidBodyType::eps)
				   .def_readwrite("mu", &RigidBodyType::mu)
				   .def_readwrite("num_grid_files", &RigidBodyType::num_grid_files)
				   .def_property(
					   "attached_particles",
					   [](RigidBodyType& rbt) -> std::vector<ParticleIO>& {
						   return rbt.attached_particle;
					   },
					   [](RigidBodyType& rbt, const std::vector<ParticleIO>& particles) {
						   rbt.attached_particle = particles;
					   })
				   .def("__repr__", [](const RigidBodyType& rbt) {
					   return "RigidBodyType(name='" + rbt.name + "', id=" + std::to_string(rbt.id) +
							  ", mass=" + std::to_string(rbt.mass) + ")";
				   });

	def_vector3_property<RigidBodyType, &RigidBodyType::inertia>(cls, "moment_of_inertia");
	def_vector3_property<RigidBodyType, &RigidBodyType::trans_damping>(cls, "damping_coefficient");
	def_vector3_property<RigidBodyType, &RigidBodyType::rot_damping>(cls, "rotational_damping");
	def_vector3_property<RigidBodyType, &RigidBodyType::trans_force_coeff>(cls,
																			 "trans_force_coeff");
	def_vector3_property<RigidBodyType, &RigidBodyType::rot_torque_coeff>(cls, "rot_torque_coeff");
	def_float_property<RigidBodyType, &RigidBodyType::diffusion>(cls, "diffusivity");
	def_float_property<RigidBodyType, &RigidBodyType::rot_diffusivity>(cls, "rotational_diffusivity");
	def_float_property<RigidBodyType, &RigidBodyType::rot_damping_coefficient>(
		cls, "rotational_damping_coefficient");
}

// ============================================================================
// NOTE: SystemState is NOT exposed to Python
// ============================================================================
// SystemState is an internal runtime container managed by SimManager.
// Python users should:
// 1. Configure the system via SimSystem
// 2. Provide initial data via ConfigParser or direct ParticleIO creation
// 3. Run simulation via SimManager (when exposed)
// 4. Read results from output files (DCD, etc.)

void init_pyobjects(py::module_& m) {
	declare_particle_io(m);
	declare_particle_type(m);
	declare_rigid_body(m);
	declare_rigid_body_type(m);
}
