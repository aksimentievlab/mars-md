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
#include <unordered_set>
#include <vector>

// pybind11 includes (implementation only)
#ifdef USE_PYTHON
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif
namespace ARBD {

namespace {
// Helper function to check for parameter with both camelCase and snake_case variants
std::string
findParameterVariant(const Reader& reader, std::string_view camelCase, std::string_view snakeCase) {
	if (reader.hasParameter(std::string(camelCase))) {
		return std::string(camelCase);
	}
	if (reader.hasParameter(std::string(snakeCase))) {
		return std::string(snakeCase);
	}
	return {};
}

// Helper function to get value for parameter with both camelCase and snake_case variants
std::string
findValueVariant(const Reader& reader, std::string_view camelCase, std::string_view snakeCase) {
	std::string key = findParameterVariant(reader, camelCase, snakeCase);
	if (!key.empty()) {
		return reader.findValue(key);
	}
	return {};
}

// Helper function to check if parameter exists in either camelCase or snake_case
bool hasParameterVariant(const Reader& reader,
						 std::string_view camelCase,
						 std::string_view snakeCase) {
	return reader.hasParameter(std::string(camelCase)) ||
		   reader.hasParameter(std::string(snakeCase));
}
} // namespace

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

	static const std::unordered_map<std::string, IntegratorType> integrator_type_map = {
		{"brownian", IntegratorType::Brownian},
		{"langevin", IntegratorType::Langevin},
		{"velocityverlet", IntegratorType::VelocityVerlet}};

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
	if (hasParameterVariant(reader, "temperature_grid", "temperature_grid")) {
		std::string key = findParameterVariant(reader, "temperature_grid", "temperature_grid");
		std::string grid_file = reader.findValue(key);
		BaseGrid<float> grid = DXReader::read_from_file<float>(grid_file);
		sim_system_ref_->set_temperature(grid);
	} else if (hasParameterVariant(reader, "temperature", "temperature")) {
		std::string key = findParameterVariant(reader, "temperature", "temperature");
		sim_system_ref_->set_temperature(reader.parseValue<float>(key));
	}

	if (hasParameterVariant(reader, "cutoff", "cutoff")) {
		std::string key = findParameterVariant(reader, "cutoff", "cutoff");
		sim_system_ref_->set_cutoff(Length(reader.parseValue<float>(key)));
	}

	// pairlistDistance is a skin/padding distance added to the interaction
	// cutoff to get the actual neighbor-list (pairlist) cutoff. Always set
	// pairlist_cutoff_ here (using a zero skin if unspecified) so it tracks
	// the configured cutoff rather than leaving SimSystem's hardcoded default.
	{
		float pairlist_distance = 0.0f;
		if (hasParameterVariant(reader, "pairlistDistance", "pairlist_distance")) {
			std::string key = findParameterVariant(reader, "pairlistDistance", "pairlist_distance");
			pairlist_distance = reader.parseValue<float>(key);
		}
		sim_system_ref_->set_pairlist_cutoff(
			Length(static_cast<float>(sim_system_ref_->get_cutoff()) + pairlist_distance));
	}

	if (hasParameterVariant(reader, "timestep", "timestep")) {
		std::string key = findParameterVariant(reader, "timestep", "timestep");
		sim_system_ref_->set_timestep(reader.parseValue<float>(key));
	}

	if (hasParameterVariant(reader, "steps", "steps")) {
		std::string key = findParameterVariant(reader, "steps", "steps");
		sim_system_ref_->set_num_steps(reader.parseValue<int>(key));
	}

	if (hasParameterVariant(reader, "neighborListRebuildPeriod", "neighbor_list_rebuild_period")) {
		std::string key = findParameterVariant(reader,
											   "neighborListRebuildPeriod",
											   "neighbor_list_rebuild_period");
		sim_system_ref_->set_neighbor_list_rebuild_period(reader.parseValue<float>(key));
	}

	if (hasParameterVariant(reader, "outputPeriod", "output_period")) {
		std::string key = findParameterVariant(reader, "outputPeriod", "output_period");
		sim_system_ref_->set_output_period(reader.parseValue<float>(key));
	}

	if (hasParameterVariant(reader, "outputEnergyPeriod", "output_energy_period")) {
		std::string key =
			findParameterVariant(reader, "outputEnergyPeriod", "output_energy_period");
		sim_system_ref_->set_energy_output_period(reader.parseValue<float>(key));
	}

	if (hasParameterVariant(reader, "outputName", "output_name")) {
		std::string key = findParameterVariant(reader, "outputName", "output_name");
		sim_system_ref_->set_output_name(reader.findValue(key));
	}

	// Parse box dimensions
	if (hasParameterVariant(reader, "systemSize", "system_size")) {
		std::string key = findParameterVariant(reader, "systemSize", "system_size");
		Vector3 size = reader.parseVector3(key);
		sim_system_ref_->set_box_size(size.x, size.y, size.z);
	}

	if (hasParameterVariant(reader, "origin", "origin")) {
		std::string key = findParameterVariant(reader, "origin", "origin");
		Vector3 origin = reader.parseVector3(key);
		sim_system_ref_->set_origin(origin.x, origin.y, origin.z);
	}

	if (hasParameterVariant(reader, "decomposer", "decomposer")) {
		std::string key = findParameterVariant(reader, "decomposer", "decomposer");
		std::string val = to_lower(reader.findValue(key));
		auto it = decomposer_map.find(val);
		if (it != decomposer_map.end()) {
			sim_system_ref_->set_decomposer_type(it->second);
		} else {
			LOGWARN("Unknown decomposer '{}', using default", val);
		}
	}

	if (hasParameterVariant(reader, "longRangeMethod", "long_range_method")) {
		std::string key = findParameterVariant(reader, "longRangeMethod", "long_range_method");
		std::string val = to_lower(reader.findValue(key));
		auto it = longrange_map.find(val);
		if (it != longrange_map.end()) {
			sim_system_ref_->set_long_range_method(it->second);
		} else {
			LOGWARN("Unknown long range method '{}', using default", val);
		}
	}

	// Handle algorithm/particleDynamicType/ParticleDynamicType variants
	if (reader.hasParameter("ParticleDynamicType") ||
		hasParameterVariant(reader, "particleDynamicType", "particle_dynamic_type") ||
		hasParameterVariant(reader, "algorithm", "particle_dynamic_type")) {
		std::string key;
		if (reader.hasParameter("ParticleDynamicType")) {
			key = "ParticleDynamicType";
		} else if (reader.hasParameter("particleDynamicType")) {
			key = "particleDynamicType";
		} else {
			key = findParameterVariant(reader, "algorithm", "particle_dynamic_type");
		}
		std::string val = to_lower(reader.findValue(key));
		auto it = integrator_type_map.find(val);
		if (it != integrator_type_map.end()) {
			sim_system_ref_->set_particle_integrator_type(it->second);
		} else {
			LOGWARN("Unknown algorithm/particle dynamic type '{}', using default", val);
		}
	}

	// Handle rigidBodyAlgorithm/RigidBodyDynamicType variants
	if (reader.hasParameter("RigidBodyDynamicType") ||
		hasParameterVariant(reader, "rigidBodyDynamicType", "rigid_body_dynamic_type") ||
		hasParameterVariant(reader, "rigidBodyAlgorithm", "rigid_body_algorithm")) {
		std::string key;
		if (reader.hasParameter("RigidBodyDynamicType")) {
			key = "RigidBodyDynamicType";
		} else if (reader.hasParameter("rigidBodyDynamicType")) {
			key = "rigidBodyDynamicType";
		} else {
			key = findParameterVariant(reader, "rigidBodyAlgorithm", "rigid_body_algorithm");
		}
		std::string val = to_lower(reader.findValue(key));
		auto it = integrator_type_map.find(val);
		if (it != integrator_type_map.end()) {
			sim_system_ref_->set_rigid_body_integrator_type(it->second);
		} else {
			LOGWARN("Unknown rigid body algorithm '{}', using default", val);
		}
	}

	if (hasParameterVariant(reader, "outputFormat", "output_format")) {
		std::string key = findParameterVariant(reader, "outputFormat", "output_format");
		std::string val = to_lower(reader.findValue(key));
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
			p.momentum.x = std::stof(toks[6]);
			p.momentum.y = std::stof(toks[7]);
			p.momentum.z = std::stof(toks[8]);
			out.push_back(p);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		LOGWARN("get_elements: Failed to read particles from '{}': {}", path, e.what());
	}
}

static void load_restart_file(const std::string& path,
							  std::vector<ParticleRead>& out,
							  const std::string& config_file_path,
							  const std::vector<ParticleType>& particle_types) {
	// Format: type_id coord_x coord_y coord_z
	// particle_id is the line order (0-based index)
	std::string resolved_path = resolve_file_path(path, config_file_path);
	try {
		ARBD::FileHandle fh(resolved_path.c_str(), "r");
		FILE* fp = fh.get();
		char* line = nullptr;
		size_t len = 0;
		ssize_t rd;
		int line_number = 0;
		while ((rd = getline(&line, &len, fp)) != -1) {
			std::string s(line, static_cast<size_t>(rd));
			if (is_comment_or_blank(s))
				continue;
			auto toks = tokenize(s);
			if (toks.size() < 4) {
				LOGWARN("load_restart_file: Invalid line (expected 4 tokens): {}", s);
				continue;
			}
			ParticleRead p{};
			p.id = line_number; // particle_id is the line order
			int type_id = std::stoi(toks[0]);
			p.position.x = std::stof(toks[1]);
			p.position.y = std::stof(toks[2]);
			p.position.z = std::stof(toks[3]);

			// Map type_id to type_name using particle_types vector
			if (type_id >= 0 && static_cast<size_t>(type_id) < particle_types.size()) {
				p.type_name = particle_types[type_id].name;
			} else {
				LOGWARN("load_restart_file: Invalid particle type_id {} (max: {})",
						type_id,
						particle_types.size() - 1);
				continue;
			}
			out.push_back(p);
			line_number++;
		}
		if (line)
			free(line);
		LOGINFO("load_restart_file: Loaded {} particles from '{}'", out.size(), resolved_path);
	} catch (const std::exception& e) {
		LOGWARN("load_restart_file: Failed to read restart file from '{}': {}", path, e.what());
	}
}
} // namespace

void ConfigParser::get_elements(const Reader& reader) {
	const auto params = reader.getParameters();

	auto is_block_header = [](std::string_view key) {
		return key == "particle"; // Extend with other block headers as needed
	};

	// Keys recognized as fields of a "particle" block. The inner loop below
	// must stop as soon as it sees anything outside this set - otherwise it
	// silently swallows top-level directives (inputParticles, inputBonds,
	// tabulatedFile, ...) that follow the last particle block, since it only
	// used to break on the next "particle" header.
	static const std::unordered_set<std::string_view> particle_field_keys = {
		"num",
		"diffusion",
		"mass",
		"gridFile",
		"grid_file",
		"diffusionGridFile",
		"diffusion_grid_file",
		"forceGridFiles",
		"force_grid_files",
		"gridFileScale",
		"grid_file_scale",
		"gridFileScaleSlope",
		"grid_file_scale_slope",
		"gridFileSMD",
		"grid_file_smd",
	};

	// When the file also supplies explicit per-particle data, that data is
	// authoritative and the "num"-based placeholders below (created at the
	// origin, with no real position) must be skipped - otherwise particles
	// end up duplicated: N placeholders plus the N real ones from the file.
	const bool has_explicit_particle_source =
		std::any_of(params.begin(), params.end(), [](const auto& kv) {
			const auto& k = kv.first;
			return k == "inputParticles" || k == "input_particles" ||
				   k == "restartCoordinates" || k == "restart_coordinates";
		});

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
				if (is_block_header(k) || !particle_field_keys.count(k))
					break;
				// Helper lambda to check for variant keys
				auto check_key = [&k](const std::string& camel, const std::string& snake) {
					return k == camel || k == snake;
				};

				if (k == "num") {
					LOGDEBUG("num: {}", v);
					num = std::stoi(v);
					ptype.num = num;
				} else if (k == "diffusion") {
					auto toks = tokenize(v);
					if (toks.size() == 3) {
						ptype.diffusion.x = std::stof(toks[0]);
						ptype.diffusion.y = std::stof(toks[1]);
						ptype.diffusion.z = std::stof(toks[2]);
					} else if (toks.size() == 1) {
						ptype.diffusion.x = std::stof(toks[0]);
						ptype.diffusion.y = std::stof(toks[0]);
						ptype.diffusion.z = std::stof(toks[0]);
					} else {
						LOGWARN("Invalid diffusion format: {}", v);
						ptype.diffusion = Vector3(0.0f, 0.0f, 0.0f);
					}
					LOGDEBUG("diffusion: {}, {}, {}",
							 ptype.diffusion.x,
							 ptype.diffusion.y,
							 ptype.diffusion.z);
				} else if (k == "mass") {
					LOGDEBUG("mass: {}", v);
					ptype.mass = std::stof(v);
				} else if (check_key("gridFile", "grid_file")) {
					LOGDEBUG("gridFile/grid_file: {}", v);
					// Load grid using GridManager and store grid_id in ParticleType
					GridKey grid_key = sim_system_ref_->get_grid_manager().add_dense_grid(v);
					if (grid_key.is_valid()) {
						ptype.pmf_grid_id = grid_key.grid_id;
						ptype.pmf_grid_name = v;
						LOGDEBUG("Assigned PMF grid '{}' with grid_id={}", v, grid_key.grid_id);
					} else {
						LOGWARN("Failed to load grid file '{}'", v);
					}
				} else if (check_key("diffusionGridFile", "diffusion_grid_file")) {
					LOGDEBUG("diffusionGridFile/diffusion_grid_file: {}", v);
					GridKey grid_key = sim_system_ref_->get_grid_manager().add_dense_grid(v);
					if (grid_key.is_valid()) {
						ptype.diffusion_grid_id = grid_key.grid_id;
						ptype.diffusion_grid_name = v;
						LOGDEBUG("Assigned diffusion grid '{}' with grid_id={}",
								 v,
								 grid_key.grid_id);
					} else {
						LOGWARN("Failed to load grid file '{}'", v);
					}
				} else if (check_key("forceGridFiles", "force_grid_files")) {
					LOGDEBUG("forceGridFiles/force_grid_files: {}", v);
					auto toks = tokenize(v);
					if (toks.size() == 3) {
						ptype.force_grid_names[0] = toks[0];
						ptype.force_grid_names[1] = toks[1];
						ptype.force_grid_names[2] = toks[2];
						GridKey grid_key_x =
							sim_system_ref_->get_grid_manager().add_dense_grid(toks[0]);
						if (grid_key_x.is_valid()) {
							ptype.force_grid_id[0] = grid_key_x.grid_id;
						} else {
							LOGWARN("Failed to load grid file '{}'", toks[0]);
						}
						GridKey grid_key_y =
							sim_system_ref_->get_grid_manager().add_dense_grid(toks[1]);
						if (grid_key_y.is_valid()) {
							ptype.force_grid_id[1] = grid_key_y.grid_id;
						} else {
							LOGWARN("Failed to load grid file '{}'", toks[1]);
						}
						GridKey grid_key_z =
							sim_system_ref_->get_grid_manager().add_dense_grid(toks[2]);
						if (grid_key_z.is_valid()) {
							ptype.force_grid_id[2] = grid_key_z.grid_id;
						} else {
							LOGWARN("Failed to load grid file '{}'", toks[2]);
						}
					} else {
						LOGWARN("Invalid force grid file format: {}", v);
						ptype.force_grid_names = {"", "", ""};
						ptype.force_grid_id = {-1, -1, -1};
					}
				} else if (check_key("gridFileScale", "grid_file_scale")) {
					LOGDEBUG("gridFileScale/grid_file_scale: {}", v);
					ptype.pmf_scale = std::stof(v);
				} else if (check_key("gridFileScaleSlope", "grid_file_scale_slope")) {
					LOGDEBUG("gridFileScaleSlope/grid_file_scale_slope: {}", v);
					ptype.pmf_scale_slope = std::stof(v);
				} else if (check_key("gridFileSMD", "grid_file_smd")) {
					LOGDEBUG("gridFileSMD/grid_file_smd: {}", v);
					ptype.pmf_smd_freq = std::stoi(v);
				} else {
					// Recognized but not yet wired into ParticleType storage in this branch
					(void)0;
				}
			}

			sim_system_ref_->get_particle_types().push_back(std::move(ptype));
			// Create placeholder particles for this type, unless explicit
			// per-particle data is supplied elsewhere in the file (see
			// has_explicit_particle_source above).
			if (!has_explicit_particle_source) {
				for (int n = 0; n < num; ++n) {
					ParticleRead p{};
					p.id = static_cast<int>(init_particles_.size());
					p.type_name = name;
					init_particles_.push_back(p);
				}
			}
			continue; // i already at next header or end
		}

		// Parse tabulated nonbonded interactions: format is i@j@file
		if (key == "tabulatedFile" || key == "tabulated_file") {
			// Parse the format: type_id_1@type_id_2@file_path
			size_t at_pos1 = value.find('@');
			if (at_pos1 == std::string::npos) {
				LOGWARN("Invalid tabulatedFile format (missing first @): {}", value);
				++i;
				continue;
			}
			size_t at_pos2 = value.find('@', at_pos1 + 1);
			if (at_pos2 == std::string::npos) {
				LOGWARN("Invalid tabulatedFile format (missing second @): {}", value);
				++i;
				continue;
			}

			try {
				int type_id_1 = std::stoi(value.substr(0, at_pos1));
				int type_id_2 = std::stoi(value.substr(at_pos1 + 1, at_pos2 - at_pos1 - 1));
				std::string file_path = value.substr(at_pos2 + 1);

				// Resolve file path relative to config file
				std::string resolved_path = resolve_file_path(file_path, file_name_);

				// Load the pair nonbonded interaction
				sim_system_ref_->get_tables_registry().load_pair_nonbonded(type_id_1,
																		   type_id_2,
																		   resolved_path);
				LOGDEBUG("Loaded tabulatedFile: type {}@{} from '{}'",
						 type_id_1,
						 type_id_2,
						 resolved_path);
			} catch (const std::exception& e) {
				LOGWARN("Failed to parse tabulatedFile '{}': {}", value, e.what());
			}
		}
		// External lists - load into temporary storage
		else if (key == "inputBonds" || key == "input_bonds" || key == "inputAngles" ||
				 key == "input_angles" || key == "inputDihedrals" || key == "input_dihedrals" ||
				 key == "inputExcludes" || key == "input_excludes" || key == "inputRestraints" ||
				 key == "input_restraints" || key == "inputProductPotentials" ||
				 key == "input_product_potentials") {
			bond_config_reader_.read_file(value, file_name_);
		} else if (key == "inputParticles" || key == "input_particles") {
			load_particles_file(value, init_particles_, file_name_);
		} else if (key == "restartCoordinates" || key == "restart_coordinates") {
			// Load restart coordinates: format is particle_id type_id coord_x coord_y coord_z
			load_restart_file(value,
							  init_particles_,
							  file_name_,
							  sim_system_ref_->get_particle_types());
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
					sim_system_ref_->set_particle_integrator_type(IntegratorType::Brownian);
				} else if (dynamic_str == "Langevin") {
					sim_system_ref_->set_particle_integrator_type(IntegratorType::Langevin);
				} else if (dynamic_str == "VelocityVerlet") {
					sim_system_ref_->set_particle_integrator_type(IntegratorType::VelocityVerlet);
				} else {
					throw Exception(ExceptionType::ValueError,
									SourceLocation(),
									"Unknown particle dynamic type: {}",
									dynamic_str);
				}
			} else if (key == "rigid_body_dynamic_type") {
				std::string dynamic_str = pybind11::cast<std::string>(value);
				if (dynamic_str == "Brownian") {
					sim_system_ref_->set_rigid_body_integrator_type(IntegratorType::Brownian);
				} else if (dynamic_str == "Langevin") {
					sim_system_ref_->set_rigid_body_integrator_type(IntegratorType::Langevin);
				} else if (dynamic_str == "VelocityVerlet") {
					sim_system_ref_->set_rigid_body_integrator_type(IntegratorType::VelocityVerlet);
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

void ConfigParser::validate() const {
	// Basic validation checks
	if (!sim_system_ref_) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"ConfigParser: SimSystem reference is null");
	}

	// Validate temperature
	float temp = sim_system_ref_->get_temperature();
	if (temp <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"ConfigParser: Temperature must be positive, got {}",
						temp);
	}

	// Validate timestep
	float dt = sim_system_ref_->get_timestep();
	if (dt <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"ConfigParser: Timestep must be positive, got {}",
						dt);
	}

	// Validate cutoff
	float cutoff = sim_system_ref_->get_cutoff();
	if (cutoff <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"ConfigParser: Cutoff must be positive, got {}",
						cutoff);
	}

	// Validate box size
	Vector3 box_size = sim_system_ref_->get_box_size();
	if (box_size.x <= 0.0f || box_size.y <= 0.0f || box_size.z <= 0.0f) {
		throw Exception(
			ExceptionType::ValueError,
			SourceLocation(),
			"ConfigParser: Box size must be positive in all dimensions, got ({}, {}, {})",
			box_size.x,
			box_size.y,
			box_size.z);
	}

	// Validate output period
	float output_period = sim_system_ref_->get_output_period();
	if (output_period <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"ConfigParser: Output period must be positive, got {}",
						output_period);
	}

	LOGINFO("ConfigParser: Configuration validation passed");
}
} // namespace ARBD
