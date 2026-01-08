#include "Backend/Resource.h"
#include "IO/ConfigParser.h"
#include "SimParam.h"
#include "System/SimSystem.h"

// pybind11 core
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Standard library
#include <string>

namespace py = pybind11;
using namespace ARBD;

/**
 * @brief Python bindings for SimSystem (time-invariant configuration) and ConfigParser
 *
 * ## Architecture:
 * - **SimSystem**: Time-invariant configuration (temperature, box, cutoff, particle types, etc.)
 * - **ConfigParser**: Loads config files, temporarily holds initial topology
 * - **SystemState**: Internal runtime state (NOT exposed to Python - managed by SimManager)
 *
 * ## Python Usage:
 *
 * ### Option 1: Load from existing config file (use ConfigParser)
 * ```python
 * from arbd2v import ConfigParser, SimSystem, Resource, ResourceType
 *
 * # Create resources (e.g., GPU or CPU)
 * resources = [Resource(ResourceType.CUDA, 0)]  # Use CUDA GPU 0
 * # Or for CPU: resources = [Resource(ResourceType.CPU)]
 *
 * # Create SimSystem with resources
 * sys = SimSystem(resources)
 *
 * # Load config file and configure the system
 * parser = ConfigParser(sys, "circovirus.bd")
 *
 * # Get initial topology (used once during initialization)
 * init_particles = parser.get_init_particles()
 *
 * # Runtime state is managed internally by SimManager (not exposed to Python)
 * # Results are written to output files (DCD, etc.)
 * ```
 *
 * ### Option 2: Pure Python configuration (no ConfigParser needed)
 * ```python
 * from arbd2v import SimSystem, ParticleType, Resource, ResourceType
 *
 * # Create resources
 * resources = [Resource(ResourceType.CUDA, 0)]  # Use CUDA GPU 0
 *
 * # Create and configure system directly
 * sys = SimSystem(resources)
 * sys.set_temperature(300.0)
 * sys.set_cutoff(10.0)
 * sys.set_box_size(100.0, 100.0, 100.0)
 * sys.set_timestep(0.01)
 * sys.set_num_steps(1000)
 *
 * # Define particle types
 * ptype = ParticleType("A")
 * ptype.mass = 1.0
 * ptype.charge = 0.0
 * sys.get_particle_types().append(ptype)
 *
 * # Initial particles can be provided via ConfigParser.get_init_particles()
 * # or generated programmatically and passed to SimManager
 * # Runtime state is managed internally - results written to output files
 * ```
 */
void init_pysystem(py::module_& m) {
	//================================================================================
	// Enum Bindings
	//================================================================================
	py::enum_<Periodicity>(m, "Periodicity")
		.value("AllPeriodic", Periodicity::AllPeriodic)
		.value("TwoDimensional", Periodicity::TwoDimensional)
		.value("OneDimensional", Periodicity::OneDimensional)
		.value("Open", Periodicity::Open);

	py::enum_<DecomposerType>(m, "DecomposerType")
		.value("Spatial", DecomposerType::Spatial)
		.value("RecursiveBisection", DecomposerType::RecursiveBisection)
		.value("Geometric", DecomposerType::Geometric);

	py::enum_<LongRangeMethod>(m, "LongRangeMethod")
		.value("CutoffAMR", LongRangeMethod::CutoffAMR)
		.value("PPPM", LongRangeMethod::PPPM)
		.value("PME", LongRangeMethod::PME)
		.value("FMM", LongRangeMethod::FMM)
		.value("Direct", LongRangeMethod::Direct)
		.value("None", LongRangeMethod::None);

	py::enum_<OutputFormat>(m, "OutputFormat")
		.value("DCD", OutputFormat::DCD)
		.value("PDB", OutputFormat::PDB)
		.value("HDF5", OutputFormat::HDF5);

	// Temperature format enum
	py::enum_<Temperature::Format>(m, "TemperatureFormat")
		.value("Value", Temperature::Format::Value)
		.value("Grid", Temperature::Format::Grid);

	// Resource type enum
	py::enum_<ResourceType>(m, "ResourceType")
		.value("CPU", ResourceType::CPU)
		.value("CUDA", ResourceType::CUDA)
		.value("SYCL", ResourceType::SYCL)
		.value("METAL", ResourceType::METAL);

	//================================================================================
	// Basic Structure Bindings
	//================================================================================
	// Length is a typedef for float, so we just expose it as float in Python
	// Users can use float directly for Length values

	py::class_<Temperature>(m, "Temperature")
		.def(py::init<>())
		.def_readwrite("format", &Temperature::format)
		.def_readwrite("value", &Temperature::value)
		.def("__repr__", [](const Temperature& t) {
			return std::string("Temperature(format=") +
				   (t.format == Temperature::Format::Value ? "Value" : "Grid") +
				   ", value=" + std::to_string(t.value) + ")";
		});

	py::class_<SimSteps>(m, "SimSteps")
		.def(py::init<float, float>(), py::arg("timestep"), py::arg("total_simulation_time"))
		.def(py::init<float, int>(), py::arg("timestep"), py::arg("steps"))
		.def_readwrite("timestep", &SimSteps::timestep)
		.def_readwrite("steps", &SimSteps::steps)
		.def_readwrite("total_simulation_time", &SimSteps::total_simulation_time)
		.def("set_total_steps", &SimSteps::set_total_steps)
		.def("__repr__", [](const SimSteps& ss) {
			return "SimSteps(timestep=" + std::to_string(ss.timestep) +
				   ", steps=" + std::to_string(ss.steps) +
				   ", total_simulation_time=" + std::to_string(ss.total_simulation_time) + ")";
		});

	py::class_<PeriodicBox>(m, "PeriodicBox")
		.def(py::init<>(), "Create non-periodic box")
		.def(py::init<Vector3, bool, bool, bool>(),
			 py::arg("box_size"),
			 py::arg("periodic_x") = true,
			 py::arg("periodic_y") = true,
			 py::arg("periodic_z") = true,
			 "Create periodic box from dimensions")
		.def(py::init<Vector3, Vector3, Vector3, Vector3, bool, bool, bool>(),
			 py::arg("origin"),
			 py::arg("basis1"),
			 py::arg("basis2"),
			 py::arg("basis3"),
			 py::arg("periodic1") = true,
			 py::arg("periodic2") = true,
			 py::arg("periodic3") = true,
			 "Create periodic box from basis vectors")
		.def("get_box_size", &PeriodicBox::get_box_size, "Get box dimensions")
		.def("get_periodicity", &PeriodicBox::get_periodicity, "Get periodicity flags")
		.def("get_origin", &PeriodicBox::get_origin, "Get origin point")
		.def("get_basis", &PeriodicBox::get_basis, "Get basis vectors")
		.def("set_box_size", &PeriodicBox::set_box_size, "Set box dimensions")
		.def("set_periodicity", &PeriodicBox::set_periodicity, "Set periodicity")
		.def("set_origin", &PeriodicBox::set_origin, "Set origin point")
		.def("set_basis", &PeriodicBox::set_basis, "Set basis vectors")
		.def("wrap_diff", &PeriodicBox::wrapDiff, "Apply minimum image convention")
		.def("is_periodic", &PeriodicBox::is_periodic, "Check if dimension is periodic")
		.def("get_volume", &PeriodicBox::get_volume, "Get box volume");

	//================================================================================
	// Resource Binding - Computational resources (CPU, GPU, etc.)
	//================================================================================
	py::class_<Resource>(m, "Resource")
		.def(py::init<>(), "Create default resource (CPU)")
		.def(py::init<short>(), py::arg("device_id"), "Create resource with device ID")
		.def(py::init<ResourceType, short>(),
			 py::arg("resource_type"),
			 py::arg("device_id") = 0,
			 "Create resource with type and device ID")
		.def("type", &Resource::type, "Get resource type")
		.def("id", &Resource::id, "Get device ID")
		.def("is_device", &Resource::is_device, "Check if resource is a device (GPU)")
		.def("is_host", &Resource::is_host, "Check if resource is host (CPU)")
		.def("supports_async",
			 &Resource::supports_async,
			 "Check if resource supports async operations")
		.def("validate", &Resource::validate, "Validate that resource exists and is accessible")
		.def("__repr__", &Resource::toString, "Get string representation")
		.def("__eq__", &Resource::operator==)
		.def("__ne__", &Resource::operator!=)
		.def("__lt__", &Resource::operator<);

	// Note: std::vector<Resource> is automatically bound by pybind11/stl.h
	// No manual binding needed - Python can use list[Resource] directly

	//================================================================================
	// SimSystem Binding - Time-immutable system configuration
	//================================================================================

	py::class_<SimSystem>(m, "SimSystem")
		.def(py::init<std::vector<Resource>>(), py::arg("resources"), "Create simulation system")
		// Physical parameters
		.def(
			"set_temperature_value",
			[](SimSystem& sys, float temp) { sys.set_temperature(temp); },
			py::arg("temperature"),
			"Set system temperature (constant value)")
		.def(
			"set_temperature_grid",
			[](SimSystem& sys, const BaseGrid<float>& grid) { sys.set_temperature(grid); },
			py::arg("grid"),
			"Set system temperature (spatial grid)")
		.def(
			"get_temperature",
			[](const SimSystem& sys, Vector3 position) { return sys.get_temperature(position); },
			py::arg("position") = Vector3{0, 0, 0},
			"Get temperature")
		.def(
			"set_cutoff",
			[](SimSystem& sys, float cutoff) { sys.set_cutoff(Length(cutoff)); },
			py::arg("cutoff"),
			"Set interaction cutoff distance")
		.def(
			"get_cutoff",
			[](const SimSystem& sys) -> float { return static_cast<float>(sys.get_cutoff()); },
			"Get interaction cutoff distance")
		.def("set_box_size",
			 &SimSystem::set_box_size,
			 py::arg("x"),
			 py::arg("y"),
			 py::arg("z"),
			 "Set box dimensions")
		.def("get_box_size", &SimSystem::get_box_size, "Get box dimensions")
		.def("set_periodicity",
			 &SimSystem::set_periodicity,
			 py::arg("px"),
			 py::arg("py"),
			 py::arg("pz"),
			 "Set periodicity")
		.def("get_boundary_conditions",
			 &SimSystem::get_boundary_conditions,
			 py::return_value_policy::reference_internal,
			 "Get boundary conditions (PeriodicBox)")
		// Method selection
		.def("set_decomposer_type",
			 &SimSystem::set_decomposer_type,
			 py::arg("type"),
			 "Set domain decomposition method")
		.def("get_decomposer_type",
			 &SimSystem::get_decomposer_type,
			 "Get domain decomposition method")
		.def("set_long_range_method",
			 &SimSystem::set_long_range_method,
			 py::arg("method"),
			 "Set long-range interaction method")
		.def("get_long_range_method",
			 &SimSystem::get_long_range_method,
			 "Get long-range interaction method")
		.def("set_particle_integrator_type",
			 &SimSystem::set_particle_integrator_type,
			 py::arg("type"),
			 "Set particle dynamics algorithm")
		.def("get_particle_algorithm",
			 &SimSystem::get_particle_algorithm,
			 "Get particle dynamics algorithm")
		.def("set_rigid_body_integrator_type",
			 &SimSystem::set_rigid_body_integrator_type,
			 py::arg("type"),
			 "Set rigid body dynamics algorithm")
		.def("get_rigid_body_algorithm",
			 &SimSystem::get_rigid_body_algorithm,
			 "Get rigid body dynamics algorithm")
		// Simulation control
		.def("set_timestep", &SimSystem::set_timestep, py::arg("dt"), "Set timestep")
		.def("get_timestep", &SimSystem::get_timestep, "Get timestep")
		.def("set_num_steps", &SimSystem::set_num_steps, py::arg("n"), "Set number of steps")
		.def("get_num_steps", &SimSystem::get_num_steps, "Get number of steps")
		.def("set_output_period",
			 &SimSystem::set_output_period,
			 py::arg("period"),
			 "Set trajectory output period")
		.def("get_output_period", &SimSystem::get_output_period, "Get trajectory output period")
		.def("set_energy_output_period",
			 &SimSystem::set_energy_output_period,
			 py::arg("period"),
			 "Set energy output period")
		.def("get_energy_output_period",
			 &SimSystem::get_energy_output_period,
			 "Get energy output period")
		.def("set_output_name",
			 &SimSystem::set_output_name,
			 py::arg("name"),
			 "Set output file base name")
		.def("get_output_name", &SimSystem::get_output_name, "Get output file base name")
		.def("set_output_format",
			 &SimSystem::set_output_format,
			 py::arg("format"),
			 "Set output file format")
		.def("get_output_format", &SimSystem::get_output_format, "Get output file format")
		// Type definitions (time-invariant configuration)
		.def("get_particle_types",
			 static_cast<std::vector<ParticleType>& (SimSystem::*)()>(
				 &SimSystem::get_particle_types),
			 py::return_value_policy::reference_internal,
			 "Get particle types")
		.def("get_rigid_body_types",
			 static_cast<std::vector<RigidBodyType>& (SimSystem::*)()>(
				 &SimSystem::get_rigid_body_types),
			 py::return_value_policy::reference_internal,
			 "Get rigid body types")
		.def("get_grid_manager",
			 static_cast<GridManager& (SimSystem::*)()>(&SimSystem::get_grid_manager),
			 py::return_value_policy::reference_internal,
			 "Get GridManager for unified grid management")
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
	py::class_<ConfigParser>(m, "ConfigParser")
		.def(py::init<SimSystem&, std::string_view>(),
			 py::arg("sim_system"),
			 py::arg("file_name"),
			 "Load configuration from file and configure SimSystem")
		.def("get_sim_system",
			 static_cast<SimSystem& (ConfigParser::*)()>(&ConfigParser::get_sim_system),
			 py::return_value_policy::reference_internal,
			 "Get reference to loaded SimSystem")
		.def("get_init_particles",
			 static_cast<std::vector<ParticleRead>& (ConfigParser::*)()>(
				 &ConfigParser::get_init_particles),
			 py::return_value_policy::reference_internal,
			 "Get initial particles (temporary data)")
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
