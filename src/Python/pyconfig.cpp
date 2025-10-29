#include "Configuration.h"
#include "IO/ConfigParser.h"
#include "SimParam.h"

// pybind11 core
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Standard library
#include <string>

namespace py = pybind11;
using namespace ARBD;

/**
 * @brief Python bindings for Configuration and ConfigParser classes
 *
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import Configuration, ConfigParser
 * >>> config = Configuration()
 * >>> config.temperature.value = 300.0
 * >>> parser = ConfigParser("config.txt")
 * >>> print(parser.get_config())
 * ```
 */
void init_pyconfig(py::module_& m) {
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

	py::enum_<DynamicType>(m, "DynamicType")
		.value("Brownian", DynamicType::Brownian)
		.value("Langevin", DynamicType::Langevin)
		.value("DPD", DynamicType::DPD);

	py::enum_<OutputFormat>(m, "OutputFormat")
		.value("DCD", OutputFormat::DCD)
		.value("PDB", OutputFormat::PDB)
		.value("HDF5", OutputFormat::HDF5);

	// Temperature format enum
	py::enum_<Temperature::Format>(m, "TemperatureFormat")
		.value("Value", Temperature::Format::Value)
		.value("Grid", Temperature::Format::Grid);

	//================================================================================
	// Basic Structure Bindings
	//================================================================================
	py::class_<Length>(m, "Length")
		.def(py::init<float>(), py::arg("value") = 0.0f)
		.def_readwrite("value", &Length::value)
		.def("__float__", [](const Length& l) { return l.value; })
		.def("__repr__", [](const Length& l) { return "Length(" + std::to_string(l.value) + ")"; });

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
	// Main Configuration Structure Binding
	//================================================================================
	py::class_<Configuration>(m, "Configuration")
		.def(py::init<>(), "Create default configuration")
		// Physical parameters
		.def_readwrite("temperature", &Configuration::temperature, "System temperature")
		.def_readwrite("cutoff", &Configuration::cutoff, "Interaction cutoff distance")
		.def_readwrite("sim_box", &Configuration::sim_box, "Simulation box configuration")
		// Method selection
		.def_readwrite("decomposer", &Configuration::decomposer, "Domain decomposition method")
		.def_readwrite("long_range_method",
					   &Configuration::long_range_method,
					   "Long-range interaction method")
		.def_readwrite("particle_dynamic_type",
					   &Configuration::ParticleDynamicType,
					   "Particle dynamics algorithm")
		.def_readwrite("rigid_body_dynamic_type",
					   &Configuration::RigidBodyDynamicType,
					   "Rigid body dynamics algorithm")
		// Simulation control
		.def_readwrite("steps", &Configuration::steps, "Simulation steps configuration")
		.def_readwrite("thermostat", &Configuration::thermostat, "Thermostat type")
		.def_readwrite("barostat", &Configuration::barostat, "Barostat type")
		.def_readwrite("output_period", &Configuration::output_period, "Trajectory output period")
		.def_readwrite("energy_output_period",
					   &Configuration::energy_output_period,
					   "Energy output period")
		.def_readwrite("output_name", &Configuration::output_name, "Output file base name")
		.def_readwrite("output_format", &Configuration::output_format, "Output file format")
		// System components
		.def_readwrite("reservoirs", &Configuration::reservoirs, "Grand canonical reservoirs")
		// Python-friendly methods
		.def("is_valid", &Configuration::is_valid, "Check if configuration is valid")
		.def("__repr__", [](const Configuration& c) {
			return "Configuration(temperature=" + std::to_string(c.temperature.value) + ")";
		});

	//================================================================================
	// ConfigParser - Configuration Manager Binding
	//================================================================================
	py::class_<ConfigParser>(m, "ConfigParser")
		.def(py::init<>())
		.def(py::init<std::string_view>(),
			 py::arg("file_name"),
			 "Construct ConfigParser from configuration file")
		.def(py::init<Configuration>(),
			 py::arg("config"),
			 "Construct ConfigParser from Configuration object")
		.def("parse_file",
			 &ConfigParser::parse_file,
			 py::arg("file_name"),
			 "Parse configuration from file")
		.def("get_config",
			 &ConfigParser::get_config,
			 py::return_value_policy::reference_internal,
			 "Get const reference to configuration")
		.def("get_mutable_config",
			 &ConfigParser::get_mutable_config,
			 py::return_value_policy::reference_internal,
			 "Get mutable reference to configuration")
		.def("validate", &ConfigParser::validate, "Validate current configuration")
		.def("__repr__", [](const ConfigParser& sc) {
			const auto& config = sc.get_config();
			return std::string("ConfigParser(temperature=") +
				   std::to_string(config.temperature.value) +
				   ", cutoff=" + std::to_string(config.cutoff.value) +
				   ", sim_box=" + std::to_string(config.sim_box.get_box_size().x) + "x" +
				   std::to_string(config.sim_box.get_box_size().y) + "x" +
				   std::to_string(config.sim_box.get_box_size().z) + ", decomposer=" +
				   (config.decomposer == DecomposerType::Spatial			  ? "Spatial"
					: config.decomposer == DecomposerType::RecursiveBisection ? "RecursiveBisection"
																			  : "Geometric") +
				   ", long_range_method=" +
				   (config.long_range_method == LongRangeMethod::CutoffAMR ? "CutoffAMR"
					: config.long_range_method == LongRangeMethod::PPPM	   ? "PPPM"
					: config.long_range_method == LongRangeMethod::PME	   ? "PME"
					: config.long_range_method == LongRangeMethod::FMM	   ? "FMM"
					: config.long_range_method == LongRangeMethod::Direct  ? "Direct"
																		   : "None") +
				   ")";
		});
}
