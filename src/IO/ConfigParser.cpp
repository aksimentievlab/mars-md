#include "ConfigParser.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/BondConfigReader.h"
#include "IO/Reader.h"
#include "Objects/Grid.h"
#include "SimParam.h"
#include "Types/Types.h"

// Standard library includes
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <unordered_map>
#include <vector>

// pybind11 includes (implementation only)
#ifdef USE_PYTHON
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif
namespace ARBD {

ConfigParser::ConfigParser(SimSystem& sim_system, std::string_view file_name)
	: sim_system_ref_(&sim_system), file_name_(file_name),
	  bond_config_reader_(init_bonded_interactions_, sim_system_ref_->get_tables_registry()) {
	parse_file(file_name);
}

#ifdef USE_PYTHON
ConfigParser::ConfigParser(SimSystem& sim_system,
						   const std::map<std::string, pybind11::object>& config_dict)
	: sim_system_ref_(&sim_system),
	  bond_config_reader_(init_bonded_interactions_, sim_system_ref_->get_tables_registry()) {
	apply_defaults();
	parse_dictionary(config_dict);
	validate();
}
#endif
void ConfigParser::parse_file(std::string_view file_name) {
	file_name_ = file_name;

	try {
		Reader reader(file_name);
		apply_defaults();
		parse_parameters(reader);
		get_elements(reader);
		validate();

		LOGINFO("ConfigParser: Successfully loaded configuration from '{}'", file_name);
	} catch (const std::exception& e) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"Failed to parse configuration file '%s': %s",
						file_name.data(),
						e.what());
	}
}

void ConfigParser::apply_defaults() {
	// Configuration struct already has sensible defaults
	sim_system_ref_->set_temperature(298.15f);
	sim_system_ref_->set_cutoff(10.0f);
	sim_system_ref_->set_timestep(1e-5f);
	sim_system_ref_->set_num_steps(1000);
	sim_system_ref_->set_neighbor_list_rebuild_period(100.0f);
	sim_system_ref_->set_output_period(10.0f);
	sim_system_ref_->set_energy_output_period(100.0f);
	sim_system_ref_->set_output_name("out");
}

void ConfigParser::parse_parameters(const Reader& reader) {
	// String to enum mapping helpers
	static const std::unordered_map<std::string, Periodicity> periodicity_map = {
		{"allperiodic", Periodicity::AllPeriodic},
		{"twodimensional", Periodicity::TwoDimensional},
		{"onedimensional", Periodicity::OneDimensional},
		{"open", Periodicity::Open}};

	static const std::unordered_map<std::string, DecomposerType> decomposer_map = {
		{"spatial", DecomposerType::Spatial},
		{"recursivebisection", DecomposerType::RecursiveBisection},
		{"geometric", DecomposerType::Geometric}};

	static const std::unordered_map<std::string, LongRangeMethod> longrange_map = {
		{"cutoffamr", LongRangeMethod::CutoffAMR},
		{"pppm", LongRangeMethod::PPPM},
		{"pme", LongRangeMethod::PME},
		{"fmm", LongRangeMethod::FMM},
		{"direct", LongRangeMethod::Direct},
		{"none", LongRangeMethod::None}};

	static const std::unordered_map<std::string, DynamicType> dynamic_type_map = {
		{"brownian", DynamicType::Brownian},
		{"langevin", DynamicType::Langevin},
		{"dpd", DynamicType::DPD}};

	static const std::unordered_map<std::string, OutputFormat> output_format_map = {
		{"dcd", OutputFormat::DCD},
		{"pdb", OutputFormat::PDB},
		{"hdf5", OutputFormat::HDF5}};

	auto to_lower = [](std::string s) {
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return s;
	};

	// Parse basic parameters
	if (reader.hasParameter("temperature_grid")) {
		std::string grid_file = reader.findValue("temperature_grid");
		BaseGrid<float> grid = DXReader::read_from_file<float>(grid_file);
		sim_system_ref_->set_temperature(grid);
	} else if (reader.hasParameter("temperature")) {
		sim_system_ref_->set_temperature(reader.parseValue<float>("temperature"));
	}

	if (reader.hasParameter("cutoff")) {
		sim_system_ref_->set_cutoff(Length(reader.parseValue<float>("cutoff")));
	}

	if (reader.hasParameter("timestep")) {
		sim_system_ref_->set_timestep(reader.parseValue<float>("timestep"));
	}

	if (reader.hasParameter("steps")) {
		sim_system_ref_->set_num_steps(reader.parseValue<int>("steps"));
	}

	if (reader.hasParameter("neighborListRebuildPeriod")) {
		sim_system_ref_->set_neighbor_list_rebuild_period(
			reader.parseValue<float>("neighborListRebuildPeriod"));
	}

	if (reader.hasParameter("outputPeriod")) {
		sim_system_ref_->set_output_period(reader.parseValue<float>("outputPeriod"));
	}

	if (reader.hasParameter("outputEnergyPeriod")) {
		sim_system_ref_->set_energy_output_period(reader.parseValue<float>("outputEnergyPeriod"));
	}

	if (reader.hasParameter("outputName")) {
		sim_system_ref_->set_output_name(reader.findValue("outputName"));
	}

	// Parse box dimensions
	if (reader.hasParameter("systemSize")) {
		Vector3 size = reader.parseVector3("systemSize");
		sim_system_ref_->set_box_size(size.x, size.y, size.z);
	}

	if (reader.hasParameter("decomposer")) {
		std::string val = to_lower(reader.findValue("decomposer"));
		auto it = decomposer_map.find(val);
		if (it != decomposer_map.end()) {
			sim_system_ref_->set_decomposer_type(it->second);
		} else {
			LOGWARN("Unknown decomposer '{}', using default", val);
		}
	}

	if (reader.hasParameter("longRangeMethod")) {
		std::string val = to_lower(reader.findValue("longRangeMethod"));
		auto it = longrange_map.find(val);
		if (it != longrange_map.end()) {
			sim_system_ref_->set_long_range_method(it->second);
		} else {
			LOGWARN("Unknown long range method '{}', using default", val);
		}
	}

	if (reader.hasParameter("algorithm")) {
		std::string val = to_lower(reader.findValue("algorithm"));
		auto it = dynamic_type_map.find(val);
		if (it != dynamic_type_map.end()) {
			sim_system_ref_->set_particle_dynamic_type(it->second);
		} else {
			LOGWARN("Unknown algorithm '{}', using default", val);
		}
	}

	if (reader.hasParameter("outputFormat")) {
		std::string val = to_lower(reader.findValue("outputFormat"));
		auto it = output_format_map.find(val);
		if (it != output_format_map.end()) {
			sim_system_ref_->set_output_format(it->second);
		} else {
			LOGWARN("Unknown output format '{}', using default", val);
		}
	}
}

//--------------------------------------------------------------------------------
// Element/block parsing and external list loaders
//--------------------------------------------------------------------------------
namespace {

static bool is_comment_or_blank(const std::string& s) {
	auto it = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
	return it == s.end() || *it == '#';
}

/**
 * @brief Resolve file path relative to the configuration file directory
 * @param file_path The file path from the config file
 * @param config_file_path The path to the configuration file
 * @return Resolved absolute path
 */
static std::string resolve_file_path(const std::string& file_path,
									 const std::string& config_file_path) {
	// If path is already absolute, return as-is
	if (file_path[0] == '/') {
		return file_path;
	}

	// If path starts with ~, expand home directory
	if (file_path[0] == '~') {
		const char* home = getenv("HOME");
		if (home) {
			return std::string(home) + file_path.substr(1);
		}
		return file_path; // Fallback if HOME not set
	}

	// For relative paths, resolve relative to config file directory
	std::string config_dir = config_file_path;
	size_t last_slash = config_dir.find_last_of('/');
	if (last_slash != std::string::npos) {
		config_dir = config_dir.substr(0, last_slash + 1);
	} else {
		config_dir = "./"; // Current directory if no path separators
	}

	return config_dir + file_path;
}

static std::vector<std::string> tokenize(const std::string& s) {
	std::istringstream iss(s);
	std::vector<std::string> out;
	std::string t;
	while (iss >> t)
		out.push_back(std::move(t));
	return out;
}

static void load_particles_file(const std::string& path,
								std::vector<ParticleRead>& out,
								const std::string& config_file_path) {
	std::string resolved_path = resolve_file_path(path, config_file_path);
	try {
		ARBD::FileHandle fh(resolved_path.c_str(), "r");
		FILE* fp = fh.get();
		char* line = nullptr;
		size_t len = 0;
		ssize_t rd;
		while ((rd = getline(&line, &len, fp)) != -1) {
			std::string s(line, static_cast<size_t>(rd));
			if (is_comment_or_blank(s))
				continue;
			auto toks = tokenize(s);
			if (toks.size() < 6)
				continue;
			ParticleRead p{};
			p.id = std::stoi(toks[1]);
			p.type_name = toks[2];
			p.position.x = std::stof(toks[3]);
			p.position.y = std::stof(toks[4]);
			p.position.z = std::stof(toks[5]);
			out.push_back(p);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read particles from '{}': {}", path, e.what());
	}
}
} // namespace

void ConfigParser::get_elements(const Reader& reader) {
	const auto params = reader.getParameters();

	auto is_block_header = [](std::string_view key) {
		return key == "particle"; // Extend with other block headers as needed
	};

	// First pass: parse sequential blocks and inline keys
	for (size_t i = 0; i < params.size();) {
		const auto& [key, value] = params[i];
		if (key == "particle") {
			std::string name = value;
			int num = 0;
			int type_index = static_cast<int>(sim_system_ref_->get_particle_types().size());
			ParticleType ptype(name);

			// Consume following field lines until next header
			++i;
			for (; i < params.size(); ++i) {
				const auto& [k, v] = params[i];
				if (is_block_header(k))
					break;
				if (k == "num") {
					LOGDEBUG("num: {}", v);
					ptype.num = std::stoi(v);
				} else if (k == "diffusion") {
					ptype.diffusion = std::stof(v);
					LOGDEBUG("diffusion: {}", v);
				} else if (k == "transDamping") {
					LOGDEBUG("transDamping: {}", v);
					auto toks = tokenize(v);
					if (toks.size() == 3) {
						ptype.transDamping.x = std::stof(toks[0]);
						ptype.transDamping.y = std::stof(toks[1]);
						ptype.transDamping.z = std::stof(toks[2]);
					} else if (toks.size() == 1) {
						ptype.transDamping.x = std::stof(toks[0]);
						ptype.transDamping.y = std::stof(toks[0]);
						ptype.transDamping.z = std::stof(toks[0]);
					} else {
						LOGWARN("Invalid transDamping format: {}", v);
						ptype.transDamping.x = 0.0f;
						ptype.transDamping.y = 0.0f;
						ptype.transDamping.z = 0.0f;
					}
				} else if (k == "mass") {
					LOGDEBUG("mass: {}", v);
					ptype.mass = std::stof(v);
				} else if (k == "gridFile") {
					LOGDEBUG("gridFile: {}", v);
					// Load grid using GridManager and store grid_id in ParticleType
					GridKey grid_key = sim_system_ref_->get_grid_manager().add_dense_grid(v);
					if (grid_key.is_valid()) {
						ptype.pmf_grid_id = grid_key.grid_id;
						LOGDEBUG("Assigned PMF grid '{}' with grid_id={}", v, grid_key.grid_id);
					} else {
						LOGWARN("Failed to load grid file '{}'", v);
					}
				} else if (k == "gridFileScale") {
					LOGDEBUG("gridFileScale: {}", v);
					ptype.pmf_scale = std::stof(v);
				} else if (k == "gridFileScaleSlope") {
					LOGDEBUG("gridFileScaleSlope: {}", v);
					ptype.pmf_scale_slope = std::stof(v);
				} else if (k == "gridFileSMD") {
					LOGDEBUG("gridFileSMD: {}", v);
					ptype.pmf_smd_freq = std::stoi(v);
				} else {
					// Recognized but not yet wired into ParticleType storage in this branch
					(void)0;
				}
			}

			sim_system_ref_->get_particle_types().push_back(std::move(ptype));
			// Create particles for this type
			for (int n = 0; n < num; ++n) {
				ParticleRead p{};
				p.id = static_cast<int>(init_particles_.size());
				p.type_name = name;
				init_particles_.push_back(p);
			}
			continue; // i already at next header or end
		}

		// External lists - load into temporary storage
		if (key == "inputBonds" || key == "inputAngles" || key == "inputDihedrals" ||
			key == "inputExcludes" || key == "inputRestraints" || key == "inputProductPotentials") {
			bond_config_reader_.read_file(value);
		} else if (key == "inputParticles") {
			load_particles_file(value, init_particles_, file_name_);
		}

		++i;
	}
}

#ifdef USE_PYTHON
void ConfigParser::parse_dictionary(const std::map<std::string, pybind11::object>& config_dict) {

	for (const auto& [key, value] : config_dict) {
		try {
			if (key == "temperature") {
				sim_system_ref_->set_temperature(pybind11::cast<float>(value));
			} else if (key == "cutoff") {
				sim_system_ref_->set_cutoff(Length(pybind11::cast<float>(value)));
			} else if (key == "timestep") {
				sim_system_ref_->set_timestep(pybind11::cast<float>(value));
			} else if (key == "num_steps" || key == "steps") {
				sim_system_ref_->set_num_steps(pybind11::cast<int>(value));
			} else if (key == "output_period") {
				sim_system_ref_->set_output_period(pybind11::cast<float>(value));
			} else if (key == "energy_output_period") {
				sim_system_ref_->set_energy_output_period(pybind11::cast<float>(value));
			} else if (key == "neighbor_list_rebuild_period") {
				sim_system_ref_->set_neighbor_list_rebuild_period(pybind11::cast<float>(value));
			} else if (key == "output_name") {
				sim_system_ref_->set_output_name(pybind11::cast<std::string>(value));
			} else if (key == "pressure") {
				// TODO: Add setter for pressure
				// sim_system_ref_->set_pressure(pybind11::cast<float>(value));
			} else if (key == "replicas") {
				// TODO: Add setter for replicas
				// sim_system_ref_->set_replicas(pybind11::cast<int>(value));
			} else if (key == "box_size") {
				// Handle box size as tuple/list of 3 floats
				auto box_list = pybind11::cast<std::vector<float>>(value);
				if (box_list.size() == 3) {
					sim_system_ref_->set_box_size(box_list[0], box_list[1], box_list[2]);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Box size must be a list/tuple of 3 floats");
				}
			} else if (key == "decomposer") {
				std::string decomposer_str = pybind11::cast<std::string>(value);
				if (decomposer_str == "Spatial") {
					sim_system_ref_->set_decomposer_type(DecomposerType::Spatial);
				} else if (decomposer_str == "RecursiveBisection") {
					sim_system_ref_->set_decomposer_type(DecomposerType::RecursiveBisection);
				} else if (decomposer_str == "Geometric") {
					sim_system_ref_->set_decomposer_type(DecomposerType::Geometric);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown decomposer type: {}",
									decomposer_str);
				}
			} else if (key == "long_range_method") {
				std::string method_str = pybind11::cast<std::string>(value);
				if (method_str == "CutoffAMR") {
					sim_system_ref_->set_long_range_method(LongRangeMethod::CutoffAMR);
				} else if (method_str == "PPPM") {
					sim_system_ref_->set_long_range_method(LongRangeMethod::PPPM);
				} else if (method_str == "PME") {
					sim_system_ref_->set_long_range_method(LongRangeMethod::PME);
				} else if (method_str == "FMM") {
					sim_system_ref_->set_long_range_method(LongRangeMethod::FMM);
				} else if (method_str == "Direct") {
					sim_system_ref_->set_long_range_method(LongRangeMethod::Direct);
				} else if (method_str == "None") {
					sim_system_ref_->set_long_range_method(LongRangeMethod::None);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown long range method: {}",
									method_str);
				}
			} else if (key == "particle_dynamic_type") {
				std::string dynamic_str = pybind11::cast<std::string>(value);
				if (dynamic_str == "Brownian") {
					sim_system_ref_->set_particle_dynamic_type(DynamicType::Brownian);
				} else if (dynamic_str == "Langevin") {
					sim_system_ref_->set_particle_dynamic_type(DynamicType::Langevin);
				} else if (dynamic_str == "DPD") {
					sim_system_ref_->set_particle_dynamic_type(DynamicType::DPD);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown particle dynamic type: {}",
									dynamic_str);
				}
			} else if (key == "rigid_body_dynamic_type") {
				std::string dynamic_str = pybind11::cast<std::string>(value);
				if (dynamic_str == "Brownian") {
					sim_system_ref_->set_rigid_body_dynamic_type(DynamicType::Brownian);
				} else if (dynamic_str == "Langevin") {
					sim_system_ref_->set_rigid_body_dynamic_type(DynamicType::Langevin);
				} else if (dynamic_str == "DPD") {
					sim_system_ref_->set_rigid_body_dynamic_type(DynamicType::DPD);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown rigid body dynamic type: {}",
									dynamic_str);
				}
			} else if (key == "output_format") {
				std::string format_str = pybind11::cast<std::string>(value);
				if (format_str == "DCD") {
					sim_system_ref_->set_output_format(OutputFormat::DCD);
				} else if (format_str == "PDB") {
					sim_system_ref_->set_output_format(OutputFormat::PDB);
				} else if (format_str == "HDF5") {
					sim_system_ref_->set_output_format(OutputFormat::HDF5);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown output format: {}",
									format_str);
				}
			} else {
				LOGINFO("ConfigParser: Ignoring unknown configuration parameter '{}'", key);
			}
		} catch (const pybind11::cast_error& e) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Failed to cast parameter '{}' to expected type: {}",
							key,
							e.what());
		}
	}

	LOGINFO("ConfigParser: Successfully parsed configuration from Python dictionary");
}
#endif
} // namespace ARBD
