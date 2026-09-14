#include "Backend/Resource.h"
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#elif defined(USE_SYCL)
#include "Backend/SYCL/SYCLManager.h"
#endif
#include "IO/ConfigParser.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/RigidBodyProperties.h"
#include "Objects/Tables.h"
#include "PyTypeCasters.h"
#include "SimParam.h"
#include "System/SimSystem.h"
// nanobind core
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

// Standard library
#include <string>

namespace nb = nanobind;
using namespace MARS;

/**
 * @brief Python bindings for SimSystem (time-invariant configuration) and ConfigParser
 *
 * ## Architecture:
 * - **SimSystem**: Time-invariant configuration (temperature, box, cutoff, particle types, etc.)
 * - **ConfigParser**: Loads config files, temporarily holds initial topology
 * - **SystemState**: Internal runtime state (NOT exposed to Python - managed by SimManager)
 * @usage: Option 1: Load from existing config file (use ConfigParser)
 * ```python
 * from marsmd import ConfigParser, SimSystem, Resource
 * # Create SimSystem with resources
 * sys = SimSystem(resources)
 *
 * # Load config file and configure the system
 * parser = ConfigParser(sys, "circovirus.bd")
 *
 * # Get initial topology as ParticleIO records (used once during initialization)
 * init_particles = parser.get_init_particles()
 *
 * # Runtime state is managed internally by SimManager (not exposed to Python)
 * # Results are written to output files (DCD, etc.)
 * ```
 *
 * @usage: Option 2: Pure Python configuration (no ConfigParser needed)
 * ```python
 * # Create and configure system directly
 * sys = SimSystem(resources)
 * sys.set_temperature(300.0)
 * sys.set_cutoff(10.0)
 * sys.set_box_size(100.0, 100.0, 100.0)
 * sys.set_timestep(0.01)
 * sys.set_num_steps(1000)
 *
 * # Define particle types (add_particle_type copies into the system;
 * # get_particle_types() returns a snapshot copy, so appending to it
 * # would be a no-op)
 * ptype = ParticleType("A")
 * ptype.mass = 1.0
 * ptype.charge = 0.0
 * sys.add_particle_type(ptype)
 *
 * # Initial particles can be provided via ConfigParser.get_init_particles()
 * # or generated programmatically and passed to SimManager
 * # Runtime state is managed internally - results written to output files
 * ```
 */
void init_pysystem(nb::module_& m) {
	//================================================================================
	// Enum Bindings
	//================================================================================
	nb::enum_<Periodicity>(m, "Periodicity")
		.value("AllPeriodic", Periodicity::AllPeriodic)
		.value("TwoDimensional", Periodicity::TwoDimensional)
		.value("OneDimensional", Periodicity::OneDimensional)
		.value("Open", Periodicity::Open);

	nb::enum_<DecomposerType>(m, "DecomposerType")
		.value("Spatial", DecomposerType::Spatial)
		.value("RecursiveBisection", DecomposerType::RecursiveBisection)
		.value("Geometric", DecomposerType::Geometric);

	nb::enum_<DecomposeDirection>(m, "DecomposeDirection")
		.value("X", DecomposeDirection::X)
		.value("Y", DecomposeDirection::Y)
		.value("Z", DecomposeDirection::Z);

	nb::enum_<LongRangeMethod>(m, "LongRangeMethod")
		.value("CutoffAMR", LongRangeMethod::CutoffAMR)
		.value("PPPM", LongRangeMethod::PPPM)
		.value("PME", LongRangeMethod::PME)
		.value("FMM", LongRangeMethod::FMM)
		.value("Direct", LongRangeMethod::Direct)
		.value("None", LongRangeMethod::None);

	nb::enum_<OutputFormat>(m, "OutputFormat")
		.value("DCD", OutputFormat::DCD)
		.value("PDB", OutputFormat::PDB)
		.value("HDF5", OutputFormat::HDF5);

	// Required by set_particle_integrator_type / set_rigid_body_integrator_type,
	// which are uncallable without it.
	nb::enum_<IntegratorType>(m, "IntegratorType")
		.value("Langevin", IntegratorType::Langevin)
		.value("Brownian", IntegratorType::Brownian)
		.value("VelocityVerlet", IntegratorType::VelocityVerlet);

	// Temperature format enum
	nb::enum_<Temperature::Format>(m, "TemperatureFormat")
		.value("Value", Temperature::Format::Value)
		.value("Grid", Temperature::Format::Grid);

	//================================================================================
	// Basic Structure Bindings
	//================================================================================
	// Length is a typedef for float, so we just expose it as float in Python
	// Users can use float directly for Length values

	nb::class_<Temperature>(m, "Temperature")
		.def(nb::init<>())
		.def_rw("format", &Temperature::format)
		.def_rw("value", &Temperature::value)
		.def("__repr__", [](const Temperature& t) {
			return std::string("Temperature(format=") +
				   (t.format == Temperature::Format::Value ? "Value" : "Grid") +
				   ", value=" + std::to_string(t.value) + ")";
		});

	nb::class_<SimSteps>(m, "SimSteps")
		.def(nb::init<float, float>(), nb::arg("timestep"), nb::arg("total_simulation_time"))
		.def(nb::init<float, int>(), nb::arg("timestep"), nb::arg("steps"))
		.def_rw("timestep", &SimSteps::timestep)
		.def_rw("steps", &SimSteps::steps)
		.def_rw("total_simulation_time", &SimSteps::total_simulation_time)
		.def("set_total_steps", &SimSteps::set_total_steps)
		.def("__repr__", [](const SimSteps& ss) {
			return "SimSteps(timestep=" + std::to_string(ss.timestep) +
				   ", steps=" + std::to_string(ss.steps) +
				   ", total_simulation_time=" + std::to_string(ss.total_simulation_time) + ")";
		});

	nb::class_<PeriodicBox>(m, "PeriodicBox")
		.def(nb::init<>(), "Create non-periodic box")
		.def(nb::init<Vector3, bool, bool, bool>(),
			 nb::arg("box_size"),
			 nb::arg("periodic_x") = true,
			 nb::arg("periodic_y") = true,
			 nb::arg("periodic_z") = true,
			 "Create periodic box from dimensions")
		.def(nb::init<Vector3, Vector3, Vector3, Vector3, bool, bool, bool>(),
			 nb::arg("origin"),
			 nb::arg("basis1"),
			 nb::arg("basis2"),
			 nb::arg("basis3"),
			 nb::arg("periodic1") = true,
			 nb::arg("periodic2") = true,
			 nb::arg("periodic3") = true,
			 "Create periodic box from basis vectors")
		.def("get_box_size", &PeriodicBox::get_box_size, "Get box dimensions")
		.def("get_periodicity", &PeriodicBox::get_periodicity, "Get periodicity flags")
		.def("get_origin", &PeriodicBox::get_origin, "Get origin point")
		.def("get_basis", &PeriodicBox::get_basis, "Get basis vectors")
		.def("set_box_size", &PeriodicBox::set_box_size, "Set box dimensions")
		.def("set_periodicity", &PeriodicBox::set_periodicity, "Set periodicity")
		.def("set_origin", &PeriodicBox::set_origin, "Set origin point")
		.def("set_basis", &PeriodicBox::set_basis, "Set basis vectors")
		.def("wrap_diff", &PeriodicBox::wrap_diff, "Apply minimum image convention")
		.def("is_periodic", &PeriodicBox::is_periodic, "Check if dimension is periodic")
		.def("get_volume", &PeriodicBox::get_volume, "Get box volume");

	//================================================================================
	// SimSystem Binding - Time-immutable system configuration
	//================================================================================

	nb::class_<SimSystem>(m, "SimSystem")
		/**
		 * @brief Create a simulation system on the given device indices.
		 * @param gpus device indices, defaulting to device 0. The backend is
		 *        fixed when the engine is built, so only the device is chosen.
		 */
		.def(
			"__init__",
			[](SimSystem* self, const std::vector<short>& gpus) {
				// Device discovery, once per process; what mars.cpp does before
				// building any Resource. See dev_notes.md.
				static bool backend_discovered = false;
				if (!backend_discovered) {
#ifdef USE_CUDA
					CUDA::Manager::init();
#elif defined(USE_SYCL)
					SYCL::Manager::init();
#endif
					backend_discovered = true;
				}
				std::vector<Resource> resources;
				resources.reserve(gpus.size());
				for (short device_id : gpus) {
					resources.emplace_back(device_id);
				}
				new (self) SimSystem(resources);
			},
			nb::arg("gpus") = std::vector<short>{0},
			"Create a simulation system on the given device indices")
		.def(
			"set_temperature_value",
			[](SimSystem& sys, float temp) { sys.set_temperature(temp); },
			nb::arg("temperature"),
			"Set system temperature (constant value)")
		.def(
			"set_temperature_grid",
			[](SimSystem& sys, const BaseGrid<mars_real>& grid) { sys.set_temperature(grid); },
			nb::arg("grid"),
			"Set system temperature (spatial grid)")
		.def(
			"get_temperature",
			[](const SimSystem& sys, Vector3 position) { return sys.get_temperature(position); },
			nb::arg("position") = Vector3{0, 0, 0},
			"Get temperature")
		.def(
			"set_cutoff",
			[](SimSystem& sys, float cutoff) { sys.set_cutoff(Length(cutoff)); },
			nb::arg("cutoff"),
			"Set interaction cutoff distance")
		.def(
			"get_cutoff",
			[](const SimSystem& sys) -> float { return static_cast<float>(sys.get_cutoff()); },
			"Get interaction cutoff distance")
		.def(
			"set_pairlist_cutoff",
			[](SimSystem& sys, float cutoff) { sys.set_pairlist_cutoff(Length(cutoff)); },
			nb::arg("pairlist_cutoff"),
			"Set pairlist cutoff distance")
		.def("set_base_seed", &SimSystem::set_base_seed, nb::arg("seed"), "Set the base RNG seed")
		.def("set_salt_concentration",
			 &SimSystem::set_salt_concentration,
			 nb::arg("salt_concentration"),
			 "Set salt concentration")
		.def("set_box_size",
			 &SimSystem::set_box_size,
			 nb::arg("x"),
			 nb::arg("y"),
			 nb::arg("z"),
			 "Set box dimensions")
		.def("get_box_size", &SimSystem::get_box_size, "Get box dimensions")
		.def("set_origin",
			 &SimSystem::set_origin,
			 nb::arg("x"),
			 nb::arg("y"),
			 nb::arg("z"),
			 "Set the simulation box origin")
		.def("set_periodicity",
			 &SimSystem::set_periodicity,
			 nb::arg("px"),
			 nb::arg("py"),
			 nb::arg("pz"),
			 "Set periodicity")
		.def("get_boundary_conditions",
			 &SimSystem::get_boundary_conditions,
			 nb::rv_policy::reference_internal,
			 "Get boundary conditions (PeriodicBox)")
		// Method selection
		.def("set_decomposer_type",
			 &SimSystem::set_decomposer_type,
			 nb::arg("type"),
			 nb::arg("direction") = DecomposeDirection::Z,
			 "Set domain decomposition method")
		.def("get_decomposer_type",
			 &SimSystem::get_decomposer_type,
			 "Get domain decomposition method")
		.def("set_long_range_method",
			 &SimSystem::set_long_range_method,
			 nb::arg("method"),
			 "Set long-range interaction method")
		.def("get_long_range_method",
			 &SimSystem::get_long_range_method,
			 "Get long-range interaction method")
		.def("set_particle_integrator_type",
			 &SimSystem::set_particle_integrator_type,
			 nb::arg("type"),
			 "Set particle dynamics algorithm")
		.def("get_particle_algorithm",
			 &SimSystem::get_particle_algorithm,
			 "Get particle dynamics algorithm")
		.def("set_rigid_body_integrator_type",
			 &SimSystem::set_rigid_body_integrator_type,
			 nb::arg("type"),
			 "Set rigid body dynamics algorithm")
		.def("get_rigid_body_algorithm",
			 &SimSystem::get_rigid_body_algorithm,
			 "Get rigid body dynamics algorithm")
		// Simulation control
		.def("set_timestep", &SimSystem::set_timestep, nb::arg("dt"), "Set timestep")
		.def("get_timestep", &SimSystem::get_timestep, "Get timestep")
		.def("set_num_steps", &SimSystem::set_num_steps, nb::arg("n"), "Set number of steps")
		.def("get_num_steps", &SimSystem::get_num_steps, "Get number of steps")
		.def("set_output_period",
			 &SimSystem::set_output_period,
			 nb::arg("period"),
			 "Set trajectory output period")
		.def("get_output_period", &SimSystem::get_output_period, "Get trajectory output period")
		.def("set_neighbor_list_rebuild_period",
			 &SimSystem::set_neighbor_list_rebuild_period,
			 nb::arg("period"),
			 "Set neighbor list rebuild period")
		.def("get_neighbor_list_rebuild_period",
			 &SimSystem::get_neighbor_list_rebuild_period,
			 "Get neighbor list rebuild period")
		.def("set_reorder_period",
			 &SimSystem::set_reorder_period,
			 nb::arg("period"),
			 "Set particle reorder period")
		.def("set_rb_update_period",
			 &SimSystem::set_rb_update_period,
			 nb::arg("period"),
			 "Set rigid body update period")
		.def("set_estimated_particles",
			 &SimSystem::set_estimated_particles,
			 nb::arg("estimated_particles"),
			 "Hint the expected particle count for preallocation")
		.def("set_energy_output_period",
			 &SimSystem::set_energy_output_period,
			 nb::arg("period"),
			 "Set energy output period")
		.def("get_energy_output_period",
			 &SimSystem::get_energy_output_period,
			 "Get energy output period")
		.def("set_output_name",
			 &SimSystem::set_output_name,
			 nb::arg("name"),
			 "Set output file base name")
		.def("get_output_name", &SimSystem::get_output_name, "Get output file base name")
		.def("set_output_format",
			 &SimSystem::set_output_format,
			 nb::arg("format"),
			 "Set output file format")
		.def("get_output_format", &SimSystem::get_output_format, "Get output file format")
		// Type definitions (time-invariant configuration)
		.def("add_particle_type",
			 &SimSystem::add_particle_type,
			 nb::arg("type"),
			 "Register a particle type (copied into the system)")
		.def("add_rigid_body_type",
			 &SimSystem::add_rigid_body_type,
			 nb::arg("type"),
			 "Register a rigid body type (copied into the system)")
		// NOTE: these return a *copy* as a plain Python list - nanobind/stl/vector.h
		// converts std::vector<T> by value, so reference_internal cannot make
		// the result a live view and mutating it (e.g. .append()) does nothing
		// to the system. Use add_particle_type()/add_rigid_body_type() to
		// register types; these are for inspection only.
		.def("get_particle_types",
			 static_cast<const std::vector<ParticleType>& (SimSystem::*)() const>(
				 &SimSystem::get_particle_types),
			 "Get a copy of the registered particle types (read-only snapshot)")
		.def("get_rigid_body_types",
			 static_cast<const std::vector<RigidBodyType>& (SimSystem::*)() const>(
				 &SimSystem::get_rigid_body_types),
			 "Get a copy of the registered rigid body types (read-only snapshot)")
		.def("get_particle_type_id",
			 &SimSystem::get_particle_type_id,
			 nb::arg("name"),
			 "Resolve a particle type name to its assigned id (valid only after "
			 "SimManager.init())")
		.def("get_rigid_body_type_id",
			 &SimSystem::get_rigid_body_type_id,
			 nb::arg("name"),
			 "Resolve a rigid body type name to its assigned id (valid only after "
			 "SimManager.init())")
		.def("get_grid_manager",
			 static_cast<GridManager& (SimSystem::*)()>(&SimSystem::get_grid_manager),
			 nb::rv_policy::reference_internal,
			 "Get GridManager for unified grid management")
		// By reference: TablesRegistry owns device buffers, and its copy ctor
		// would reallocate every one of them.
		.def("get_tables_registry",
			 static_cast<TablesRegistry& (SimSystem::*)()>(&SimSystem::get_tables_registry),
			 nb::rv_policy::reference_internal,
			 "Get TablesRegistry - the tabulated potential cache that assigns function_index")
		.def("get_nonbonded_interactions",
			 static_cast<NonBondedInteractions& (SimSystem::*)()>(
				 &SimSystem::get_nonbonded_interactions),
			 nb::rv_policy::reference_internal,
			 "Get NonBondedInteractions - pair/long-range parameters are tied to particle "
			 "type definitions, so this lives on SimSystem rather than being staged into "
			 "SimManager like bonded interactions")
		// Validation
		.def("is_valid", &SimSystem::is_valid, "Check if system configuration is valid")
		.def("__repr__", [](const SimSystem& sys) {
			return std::string("SimSystem(temperature=") + std::to_string(sys.get_temperature()) +
				   ", cutoff=" + std::to_string(static_cast<float>(sys.get_cutoff())) +
				   ", box_size=" + sys.get_box_size().to_string() + ")";
		});

	//================================================================================
	// ConfigParser - Configuration File Loader
	//================================================================================
	// ConfigParser loads configuration files and temporarily holds initial topology data.
	// The initial data (particles, bonds, etc.) is retrieved once during initialization
	// and then discarded. It does NOT create SystemState - that's created separately.
	nb::class_<ConfigParser>(m, "ConfigParser")
		// keep_alive: ConfigParser holds a raw SimSystem*, so the system must
		// outlive the parser.
		.def(nb::init<SimSystem&, std::string_view>(),
			 nb::arg("sim_system"),
			 nb::arg("file_name"),
			 nb::keep_alive<1, 2>(),
			 "Load configuration from file and configure SimSystem")
		.def("get_sim_system",
			 static_cast<SimSystem& (ConfigParser::*)()>(&ConfigParser::get_sim_system),
			 nb::rv_policy::reference_internal,
			 "Get reference to loaded SimSystem")
		.def("get_init_particles",
			 static_cast<std::vector<ParticleIO>& (ConfigParser::*)()>(
				 &ConfigParser::get_init_particles),
			 nb::rv_policy::reference_internal,
			 "Get initial particles (temporary data)")
		.def("get_init_bonded_interactions",
			 &ConfigParser::get_init_bonded_interactions,
			 nb::rv_policy::reference_internal,
			 "Get parsed bonds/angles/dihedrals/exclusions/restraints (temporary data)")
		.def("get_init_rigid_bodies",
			 static_cast<std::vector<RigidBodyIO>& (ConfigParser::*)()>(
				 &ConfigParser::get_init_rigid_bodies),
			 nb::rv_policy::reference_internal,
			 "Get initial rigid bodies (temporary data)")
		.def("validate", &ConfigParser::validate, "Validate loaded configuration")
		.def("__repr__", [](const ConfigParser& parser) {
			const auto& sys = parser.get_sim_system();
			return std::string("ConfigParser(temperature=") +
				   std::to_string(sys.get_temperature()) +
				   ", cutoff=" + std::to_string(static_cast<float>(sys.get_cutoff())) +
				   ", box_size=" + sys.get_box_size().to_string() +
				   ", init_particles=" + std::to_string(parser.get_init_particles().size()) + ")";
		});
}
