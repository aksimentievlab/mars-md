#include "Objects/Grid.h"
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include "PyTypeCasters.h"
#include <atomic>
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
 * >>> p.type_name
 * 'A'
 * >>> p.position
 * array([0., 0., 0.], dtype=float32)
 * ```
 *
 * ParticleIO is the host-side particle record loaded by ConfigParser.
 * Names follow arbdmodel.core_objects.PointParticle where applicable.
 * `id` (C++) is not exposed - it's an insertion-order index the engine
 * assigns itself (ConfigParser.cpp: `p.id = init_particles_.size()`), not
 * something a Python caller supplies or should rely on.
 */
void declare_particle_io(py::module& m) {
	py::class_<ParticleIO>(m, "Particle")
		// Every Python-constructed particle gets a unique handle so bonded
		// terms can name it before SimManager.stage_particles() fixes the
		// ordering (see ParticleUids in Interactions/BondedInteraction.h).
		// Note a copy shares its source's handle and so refers to the same
		// logical particle for bonding purposes.
		.def(py::init([]() {
			static std::atomic<int> next_uid{0};
			ParticleIO p{};
			p.uid = next_uid++;
			return p;
		}))
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
			return "ParticleIO(type_name='" + p.type_name +
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
	// Bound so pybind11/stl.h can convert ParticleType::pmf_grids
	// (std::vector<GridTerm>) instead of failing at attribute access.
	// `grid_id` (C++) is not settable directly - like the other insertion-
	// order-assigned ids in this file, it's an index into GridManager's grid
	// views, only meaningful once GridManager has actually loaded the grid.
	// Build from the GridKey that GridManager.add_dense_grid()/add_grid()
	// returns instead; `grid_id` stays readable for introspection.
	py::class_<GridTerm>(m, "GridTerm")
		.def(py::init<>())
		.def(py::init(
				 [](const GridKey& grid, float scale, float scale_slope, int boundary_condition) {
					 GridTerm t{};
					 t.grid_id = grid.grid_id;
					 t.scale = scale;
					 t.scale_slope = scale_slope;
					 t.boundary_condition = boundary_condition;
					 return t;
				 }),
			 py::arg("grid"),
			 py::arg("scale") = 1.0f,
			 py::arg("scale_slope") = 0.0f,
			 py::arg("boundary_condition") = -1)
		.def_property_readonly("grid_id", [](const GridTerm& t) { return t.grid_id; })
		.def_readwrite("scale", &GridTerm::scale)
		.def_readwrite("scale_slope", &GridTerm::scale_slope)
		.def_readwrite("boundary_condition", &GridTerm::boundary_condition)
		.def("__repr__", [](const GridTerm& t) {
			return "GridTerm(grid_id=" + std::to_string(t.grid_id) +
				   ", scale=" + std::to_string(t.scale) + ")";
		});

	// `id` (C++) is not exposed - it's only meaningful after
	// SimSystem::assign_particle_type_ids() runs (insertion-order assigned),
	// so a value read here before that point would just be stale/wrong. Look
	// it up by name via SimSystem.get_particle_type_id() instead once types
	// are registered.
	auto cls =
		py::class_<ParticleType>(m, "ParticleType")
			.def(py::init<const std::string&>(), py::arg("name"))
			.def_readwrite("name", &ParticleType::name)
			.def_readwrite("num", &ParticleType::num)
			.def_readwrite("mass", &ParticleType::mass)
			.def_readwrite("charge", &ParticleType::charge)
			.def_readwrite("radius", &ParticleType::radius)
			.def_readwrite("eps", &ParticleType::eps)
			.def_readwrite("mu", &ParticleType::mu)
			.def_readwrite("pmf_smd_freq", &ParticleType::pmf_smd_freq)
			// One entry per gridFile, each with its own scale/slope/BC -
			// replaces the old scalar pmf_scale/pmf_grid_id pair.
			.def_readwrite("pmf_grids", &ParticleType::pmf_grids)
			.def_readwrite("pmf_grid_names", &ParticleType::pmf_grid_names)
			.def_readwrite("diffusion_grid_id", &ParticleType::diffusion_grid_id)
			.def_readwrite("force_grid_id", &ParticleType::force_grid_id)
			.def("__repr__", [](const ParticleType& pt) {
				return "ParticleType(name='" + pt.name + "', num=" + std::to_string(pt.num) + ")";
			});

	def_vector3_property<ParticleType, &ParticleType::diffusion>(cls, "diffusivity");
	def_vector3_property<ParticleType, &ParticleType::trans_damping>(cls, "damping_coefficient");
}

// ============================================================================
// RIGID BODY BINDINGS
// ============================================================================

// `id` (C++) is not exposed here either - same insertion-order-assigned
// bookkeeping field as ParticleIO::id (SimManager.cpp assigns it from
// init_rigid_bodies_.size() when staged), not user-supplied data. `type_id`
// (C++) is not exposed either, for the same reason ParticleType.id isn't:
// use type_name (like ParticleIO) instead, resolved by SimSystem at
// SystemState::set_init_rigid_body_data() time.
void declare_rigid_body(py::module& m) {
	py::class_<RigidBodyIO>(m, "RigidBody")
		.def(py::init<>())
		.def_readwrite("type_name", &RigidBodyIO::type_name)
		.def_readwrite("position", &RigidBodyIO::position)
		.def_readwrite("orientation", &RigidBodyIO::orientation)
		.def_readwrite("momentum", &RigidBodyIO::momentum)
		.def_property(
			"angular_momentum",
			[](const RigidBodyIO& rb) -> const Vector3& { return rb.angular_momentum; },
			[](RigidBodyIO& rb, const Vector3& value) { rb.angular_momentum = value; })
		.def_readwrite("force", &RigidBodyIO::force)
		.def_readwrite("torque", &RigidBodyIO::torque)
		.def_readwrite("is_dummy", &RigidBodyIO::is_dummy)
		.def_readwrite("has_orientation", &RigidBodyIO::has_orientation)
		.def("__repr__", [](const RigidBodyIO& rb) {
			return "RigidBody(type_name='" + rb.type_name +
				   "', position=" + rb.position.to_string() + ")";
		});
}

// `id` (C++) is not exposed - same insertion-order-assigned-by-SimSystem
// caveat as ParticleType::id above; look it up by name via
// SimSystem.get_rigid_body_type_id() instead.
void declare_rigid_body_type(py::module& m) {
	auto cls = py::class_<RigidBodyType>(m, "RigidBodyType")
				   .def(py::init([](const std::string& name) {
							RigidBodyType rbt{};
							rbt.name = name;
							return rbt;
						}),
						py::arg("name"))
				   .def_readwrite("name", &RigidBodyType::name)
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
					   return "RigidBodyType(name='" + rbt.name +
							  "', mass=" + std::to_string(rbt.mass) + ")";
				   });

	def_vector3_property<RigidBodyType, &RigidBodyType::inertia>(cls, "moment_of_inertia");
	def_vector3_property<RigidBodyType, &RigidBodyType::trans_damping>(cls, "damping_coefficient");
	def_vector3_property<RigidBodyType, &RigidBodyType::rot_damping>(cls, "rotational_damping");
	def_vector3_property<RigidBodyType, &RigidBodyType::trans_force_coeff>(cls,
																		   "trans_force_coeff");
	def_vector3_property<RigidBodyType, &RigidBodyType::rot_torque_coeff>(cls, "rot_torque_coeff");
	def_float_property<RigidBodyType, &RigidBodyType::diffusion>(cls, "diffusivity");
	def_float_property<RigidBodyType, &RigidBodyType::rot_diffusivity>(cls,
																	   "rotational_diffusivity");
	def_float_property<RigidBodyType, &RigidBodyType::rot_damping_coefficient>(
		cls,
		"rotational_damping_coefficient");
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
