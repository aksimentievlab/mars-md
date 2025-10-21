#include "ConfigParser.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/Reader.h"
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

ConfigParser::ConfigParser(std::string_view file_name) : file_name_(file_name) {
	parse_file(file_name);
}

ConfigParser::ConfigParser(Configuration config) : config_(std::move(config)) {
	validate();
}

#ifdef USE_PYTHON
ConfigParser::ConfigParser(const std::map<std::string, pybind11::object>& config_dict) {
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
	config_ = Configuration{};
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
	if (reader.hasParameter("temperature")) {
		config_.set_temperature(reader.parseValue<float>("temperature"));
	}

	if (reader.hasParameter("cutoff")) {
		config_.cutoff = Length(reader.parseValue<float>("cutoff"));
	}

	if (reader.hasParameter("timestep")) {
		config_.steps.timestep = reader.parseValue<float>("timestep");
	}

	if (reader.hasParameter("steps")) {
		config_.steps.steps = reader.parseValue<int>("steps");
	}

	if (reader.hasParameter("neighborListRebuildPeriod")) {
		config_.neighbor_list_rebuild_period = reader.parseValue<int>("neighborListRebuildPeriod");
	}

	if (reader.hasParameter("outputPeriod")) {
		config_.output_period = reader.parseValue<float>("outputPeriod");
	}

	if (reader.hasParameter("outputEnergyPeriod")) {
		config_.energy_output_period = reader.parseValue<float>("outputEnergyPeriod");
	}

	if (reader.hasParameter("outputName")) {
		config_.output_name = reader.findValue("outputName");
	}

	// Parse box dimensions
	if (reader.hasParameter("systemSize")) {
		Vector3 size = reader.parseVector3("systemSize");
		config_.set_box_size(size.x, size.y, size.z);
	}

	if (reader.hasParameter("decomposer")) {
		std::string val = to_lower(reader.findValue("decomposer"));
		auto it = decomposer_map.find(val);
		if (it != decomposer_map.end()) {
			config_.decomposer = it->second;
		} else {
			LOGWARN("Unknown decomposer '{}', using default", val);
		}
	}

	if (reader.hasParameter("longRangeMethod")) {
		std::string val = to_lower(reader.findValue("longRangeMethod"));
		auto it = longrange_map.find(val);
		if (it != longrange_map.end()) {
			config_.long_range_method = it->second;
		} else {
			LOGWARN("Unknown long range method '{}', using default", val);
		}
	}

	if (reader.hasParameter("algorithm")) {
		std::string val = to_lower(reader.findValue("algorithm"));
		auto it = dynamic_type_map.find(val);
		if (it != dynamic_type_map.end()) {
			config_.ParticleDynamicType = it->second;
		} else {
			LOGWARN("Unknown algorithm '{}', using default", val);
		}
	}

	if (reader.hasParameter("outputFormat")) {
		std::string val = to_lower(reader.findValue("outputFormat"));
		auto it = output_format_map.find(val);
		if (it != output_format_map.end()) {
			config_.output_format = it->second;
		} else {
			LOGWARN("Unknown output format '{}', using default", val);
		}
	}

	if (reader.hasParameter("hasReaction")) {
		config_.has_reaction = reader.parseValue<bool>("hasReaction");
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

static void load_bonds_file(const std::string& path,
							std::vector<Bond>& out,
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
			if (toks.size() < 5)
				continue;
			Bond b{};
			b.flag = static_cast<BondFlag>(std::stoi(toks[1]));
			b.ind1 = std::stoi(toks[2]);
			b.ind2 = std::stoi(toks[3]);
			b.name = toks[4];
			if (b.name.find(".dat") != std::string::npos) {
				b.form = InteractionForm::Tabulated;
			} else {
				b.form = InteractionForm::Analytical;
			}
			out.push_back(b);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read bonds from '{}': {}", path, e.what());
	}
}

static void load_angles_file(const std::string& path,
							 std::vector<Angle>& out,
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
			if (toks.size() < 3)
				continue;
			Angle a{};
			a.ind1 = std::stoi(toks[0]);
			a.ind2 = std::stoi(toks[1]);
			a.ind3 = std::stoi(toks[2]);
			a.name = toks[3];
			a.form = InteractionForm::Tabulated;
			a.functionIndex = 0;
			out.push_back(a);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read angles from '{}': {}", path, e.what());
	}
}

static void load_dihedrals_file(const std::string& path,
								std::vector<Dihedral>& out,
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
			if (toks.size() < 4)
				continue;
			Dihedral d{};
			d.ind1 = std::stoi(toks[0]);
			d.ind2 = std::stoi(toks[1]);
			d.ind3 = std::stoi(toks[2]);
			d.ind4 = std::stoi(toks[3]);
			d.name = toks[4];
			d.form = InteractionForm::Tabulated;
			d.functionIndex = 0;
			out.push_back(d);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read dihedrals from '{}': {}", path, e.what());
	}
}

static void load_excludes_file(const std::string& path,
							   std::vector<Exclude>& out,
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
			if (toks.size() < 2)
				continue;
			int i1 = std::stoi(toks[0]);
			int i2 = std::stoi(toks[1]);
			out.emplace_back(i1, i2);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read excludes from '{}': {}", path, e.what());
	}
}

static void load_restraints_file(const std::string& path,
								 std::vector<Restraint>& out,
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
			// Accept: id k x0 y0 z0
			if (toks.size() != 5)
				continue;
			int id = std::stoi(toks[0]);
			float k1 = std::stof(toks[1]);

			float x0 = std::stof(toks[2]);
			float y0 = std::stof(toks[3]);
			float z0 = std::stof(toks[4]);
			out.emplace_back(id, Vector3{x0, y0, z0}, k1);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read restraints from '{}': {}", path, e.what());
	}
}
static void load_particles_file(const std::string& path,
								std::vector<Particle>& out,
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
			Particle p{};
			p.id = std::stoi(toks[1]);
			p.type_id = std::stoi(toks[2]);
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
	auto& objects = config_.objects;

	auto is_block_header = [](std::string_view key) {
		return key == "particle"; // Extend with other block headers as needed
	};

	// First pass: parse sequential blocks and inline keys
	for (size_t i = 0; i < params.size();) {
		const auto& [key, value] = params[i];
		if (key == "particle") {
			std::string name = value;
			int num = 0;
			int type_index = static_cast<int>(objects.particle_types.size());
			ParticleType ptype(name);

			// Consume following field lines until next header
			++i;
			for (; i < params.size(); ++i) {
				const auto& [k, v] = params[i];
				if (is_block_header(k))
					break;
				if (k == "num") {
					std::cout << "num: " << v << std::endl;
					ptype.num = std::stoi(v);
				} else if (k == "diffusion") {
					ptype.diffusion = std::stof(v);
					std::cout << "diffusion: " << v << std::endl;
				} else if (k == "transDamping") {
					std::cout << "transDamping: " << v << std::endl;
					ptype.transDamping = Vector3(std::stof(v));
				} else if (k == "mass") {
					std::cout << "mass: " << v << std::endl;
					ptype.mass = std::stof(v);
				} else if (k == "gridFile" || k == "charge" || k == "radius" || k == "eps" ||
						   k == "mu") {
					// Recognized but not yet wired into ParticleType storage in this branch
					(void)0;
				} else {
					// Non-field, ignore here
				}
			}

			objects.particle_types.push_back(std::move(ptype));
			// Create particles for this type
			for (int n = 0; n < num; ++n) {
				Particle p{};
				p.id = static_cast<int>(objects.particles.size());
				p.type_id = type_index;
				objects.particles.push_back(p);
			}
			continue; // i already at next header or end
		}

		// External lists
		if (key == "inputBonds") {
			load_bonds_file(value, objects.bonds, file_name_);
		} else if (key == "inputAngles") {
			load_angles_file(value, objects.angles, file_name_);
		} else if (key == "inputDihedrals") {
			load_dihedrals_file(value, objects.dihedrals, file_name_);
		} else if (key == "inputExcludes") {
			load_excludes_file(value, objects.exclusions, file_name_);
		} else if (key == "inputRestraints") {
			load_restraints_file(value, objects.restraints, file_name_);
		} else if (key == "inputParticles") {
			load_particles_file(value, objects.particles, file_name_);
		}

		++i;
	}
}

void ConfigParser::validate() const {
	validate_physical_parameters();
	validate_method_parameters();
	validate_output_parameters();

	if (!config_.is_valid()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Configuration failed basic validation checks");
	}
}

void ConfigParser::validate_physical_parameters() const {
	if (config_.temperature.value <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Temperature must be positive (got {})",
						config_.temperature.value);
	}

	if (config_.cutoff.value <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Cutoff distance must be positive (got {})",
						config_.cutoff.value);
	}
}

void ConfigParser::validate_method_parameters() const {
	if (config_.steps.timestep <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Timestep must be positive (got {})",
						config_.steps.timestep);
	}

	if (config_.steps.steps <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Number of steps must be positive (got {})",
						config_.steps.steps);
	}

	if (config_.neighbor_list_rebuild_period <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Neighbor list rebuild period must be positive (got {})",
						config_.neighbor_list_rebuild_period);
	}
}

void ConfigParser::validate_output_parameters() const {
	if (config_.output_period <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Output period must be positive (got {})",
						config_.output_period);
	}

	if (config_.energy_output_period <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Energy output period must be positive (got {})",
						config_.energy_output_period);
	}

	if (config_.output_name.empty()) {
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Output name cannot be empty");
	}
}
#ifdef USE_PYTHON
void ConfigParser::parse_dictionary(const std::map<std::string, pybind11::object>& config_dict) {

	for (const auto& [key, value] : config_dict) {
		try {
			if (key == "temperature") {
				config_.set_temperature(pybind11::cast<float>(value));
			} else if (key == "cutoff") {
				config_.cutoff.value = pybind11::cast<float>(value);
			} else if (key == "timestep") {
				config_.set_timestep(pybind11::cast<float>(value));
			} else if (key == "num_steps" || key == "steps") {
				config_.set_num_steps(pybind11::cast<int>(value));
			} else if (key == "output_period") {
				config_.output_period = pybind11::cast<float>(value);
			} else if (key == "energy_output_period") {
				config_.energy_output_period = pybind11::cast<float>(value);
			} else if (key == "neighbor_list_rebuild_period") {
				config_.neighbor_list_rebuild_period = pybind11::cast<float>(value);
			} else if (key == "output_name") {
				config_.output_name = pybind11::cast<std::string>(value);
			} else if (key == "pressure") {
				config_.pressure.value = pybind11::cast<float>(value);
			} else if (key == "replicas") {
				config_.replicas = pybind11::cast<int>(value);
			} else if (key == "box_size") {
				// Handle box size as tuple/list of 3 floats
				auto box_list = pybind11::cast<std::vector<float>>(value);
				if (box_list.size() == 3) {
					config_.set_box_size(box_list[0], box_list[1], box_list[2]);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Box size must be a list/tuple of 3 floats");
				}
			} else if (key == "decomposer") {
				std::string decomposer_str = pybind11::cast<std::string>(value);
				if (decomposer_str == "Spatial") {
					config_.decomposer = DecomposerType::Spatial;
				} else if (decomposer_str == "RecursiveBisection") {
					config_.decomposer = DecomposerType::RecursiveBisection;
				} else if (decomposer_str == "Geometric") {
					config_.decomposer = DecomposerType::Geometric;
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown decomposer type: {}",
									decomposer_str);
				}
			} else if (key == "long_range_method") {
				std::string method_str = pybind11::cast<std::string>(value);
				if (method_str == "CutoffAMR") {
					config_.long_range_method = LongRangeMethod::CutoffAMR;
				} else if (method_str == "PPPM") {
					config_.long_range_method = LongRangeMethod::PPPM;
				} else if (method_str == "PME") {
					config_.long_range_method = LongRangeMethod::PME;
				} else if (method_str == "FMM") {
					config_.long_range_method = LongRangeMethod::FMM;
				} else if (method_str == "Direct") {
					config_.long_range_method = LongRangeMethod::Direct;
				} else if (method_str == "None") {
					config_.long_range_method = LongRangeMethod::None;
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown long range method: {}",
									method_str);
				}
			} else if (key == "particle_dynamic_type") {
				std::string dynamic_str = pybind11::cast<std::string>(value);
				if (dynamic_str == "Brownian") {
					config_.ParticleDynamicType = DynamicType::Brownian;
				} else if (dynamic_str == "Langevin") {
					config_.ParticleDynamicType = DynamicType::Langevin;
				} else if (dynamic_str == "DPD") {
					config_.ParticleDynamicType = DynamicType::DPD;
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown particle dynamic type: {}",
									dynamic_str);
				}
			} else if (key == "rigid_body_dynamic_type") {
				std::string dynamic_str = pybind11::cast<std::string>(value);
				if (dynamic_str == "Brownian") {
					config_.RigidBodyDynamicType = DynamicType::Brownian;
				} else if (dynamic_str == "Langevin") {
					config_.RigidBodyDynamicType = DynamicType::Langevin;
				} else if (dynamic_str == "DPD") {
					config_.RigidBodyDynamicType = DynamicType::DPD;
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown rigid body dynamic type: {}",
									dynamic_str);
				}
			} else if (key == "output_format") {
				std::string format_str = pybind11::cast<std::string>(value);
				if (format_str == "DCD") {
					config_.output_format = OutputFormat::DCD;
				} else if (format_str == "PDB") {
					config_.output_format = OutputFormat::PDB;
				} else if (format_str == "HDF5") {
					config_.output_format = OutputFormat::HDF5;
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
