#include "Configuration.h"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

void init_pyconfig(py::module_& m) {
	// Enums
	py::enum_<Periodicity>(m, "Periodicity")
		.value("AllPeriodic", Periodicity::AllPeriodic)
		.value("TwoDimensional", Periodicity::TwoDimensional)
		.value("OneDimensional", Periodicity::OneDimensional)
		.value("Open", Periodicity::Open);

	py::enum_<DecomposerType>(m, "DecomposerType")
		.value("Cell", DecomposerType::Cell)
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

	// Basic structures
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

	py::class_<OutputPeriod>(m, "OutputPeriod")
		.def(py::init<>())
		.def_readwrite("period", &OutputPeriod::period)
		.def_readwrite("energy_period", &OutputPeriod::energy_period)
		.def("__repr__", [](const OutputPeriod& op) {
			return "OutputPeriod(period=" + std::to_string(op.period) +
				   ", energy_period=" + std::to_string(op.energy_period) + ")";
		});

	py::class_<SimSteps>(m, "SimSteps")
		.def(py::init<>())
		.def_readwrite("timestep", &SimSteps::timestep)
		.def_readwrite("steps", &SimSteps::steps)
		.def_readwrite("decomp_period", &SimSteps::decomp_period)
		.def("__repr__", [](const SimSteps& ss) {
			return "SimSteps(timestep=" + std::to_string(ss.timestep) +
				   ", steps=" + std::to_string(ss.steps) +
				   ", decomp_period=" + std::to_string(ss.decomp_period) + ")";
		});

	py::class_<BoundaryConditions>(m, "BoundaryConditions")
		.def(py::init<>())
		.def(py::init<Vector3, Vector3, Vector3, Vector3, bool, bool, bool>(),
			 py::arg("basis1"),
			 py::arg("basis2"),
			 py::arg("basis3"),
			 py::arg("origin") = Vector3{0, 0, 0},
			 py::arg("periodic1") = true,
			 py::arg("periodic2") = true,
			 py::arg("periodic3") = true)
		.def("get_origin",
			 &BoundaryConditions::get_origin,
			 py::return_value_policy::reference_internal)
		.def("get_basis",
			 &BoundaryConditions::get_basis,
			 py::return_value_policy::reference_internal)
		.def("get_periodicity",
			 &BoundaryConditions::get_periodicity,
			 py::return_value_policy::reference_internal)
		.def("set_origin", &BoundaryConditions::set_origin)
		.def("set_basis", &BoundaryConditions::set_basis)
		.def("set_periodicity", &BoundaryConditions::set_periodicity);

	// Main Configuration structure
	py::class_<Configuration>(m, "Configuration")
		.def(py::init<>())
		// Physical parameters
		.def_readwrite("temperature", &Configuration::temperature)
		.def_readwrite("cutoff", &Configuration::cutoff)
		.def_readwrite("box_lengths", &Configuration::box_lengths)
		// Method selection
		.def_readwrite("periodicity", &Configuration::periodicity)
		.def_readwrite("decomposer", &Configuration::decomposer)
		.def_readwrite("long_range_method", &Configuration::long_range_method)
		.def_readwrite("algorithm", &Configuration::ParticleDynamicType)
		// Simulation control
		.def_readwrite("steps", &Configuration::steps)
		.def_readwrite("output", &Configuration::output_period)
		.def_readwrite("output_format", &Configuration::output_format)
		.def_readwrite("output_name", &Configuration::output_name)
		// System components
		.def_readwrite("reservoirs", &Configuration::reservoirs)
		.def_readwrite("has_reaction", &Configuration::has_reaction)
		// Python-friendly methods
		.def("set_temperature", &Configuration::set_temperature, py::arg("temp"))
		.def("set_box_size", &Configuration::set_box_size, py::arg("x"), py::arg("y"), py::arg("z"))
		.def("set_timestep", &Configuration::set_timestep, py::arg("dt"))
		.def("set_num_steps", &Configuration::set_num_steps, py::arg("n"))
		.def("is_valid", &Configuration::is_valid)
		.def("__repr__", [](const Configuration& c) {
			return "Configuration(temperature=" + std::to_string(c.temperature.value) +
				   ", algorithm=" +
				   (c.algorithm == DynamicType::Langevin   ? "Langevin"
					: c.algorithm == DynamicType::Brownian ? "Brownian"
														   : "DPD") +
				   ", steps=" + std::to_string(c.steps.steps) + ")";
		});

	// SimConf - Configuration manager
	py::class_<SimConf>(m, "SimConf")
		.def(py::init<>())
		.def(py::init<std::string_view>(), py::arg("file_name"))
		.def(py::init<Configuration>(), py::arg("config"))
		.def("parse_file", &SimConf::parse_file, py::arg("file_name"))
		.def("get_config", &SimConf::get_config, py::return_value_policy::reference_internal)
		.def("get_mutable_config",
			 &SimConf::get_mutable_config,
			 py::return_value_policy::reference_internal)
		.def("validate", &SimConf::validate)
		.def("create_boundary_conditions", &SimConf::create_boundary_conditions)
		.def("get_sim_conf", &SimConf::get_sim_conf, py::return_value_policy::reference_internal)
		.def("__repr__", [](const SimConf& sc) {
			const auto& config = sc.get_config();
			return std::string("SimConf(algorithm=") +
				   (config.algorithm == DynamicType::Langevin	? "Langevin"
					: config.algorithm == DynamicType::Brownian ? "Brownian"
																: "DPD") +
				   ", temperature=" + std::to_string(config.temperature.value) + ")";
		});
}
