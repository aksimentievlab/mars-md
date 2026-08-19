#include "ConfigParser.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/BondConfigReader.h"
#include "IO/Reader.h"
#include "IO/RigidBodyCoordReader.h"
#include "IO/RigidBodyVisualize.h"
#include "Objects/Grid.h"
#include "SimParam.h"
#include "Types/Types.h"

// Standard library includes
#include <algorithm>
#include <array>
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
	sim_system_ref_->set_neighbor_list_rebuild_period(1000.0f);
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
		BaseGrid<arbd_real> grid = DXReader::read_from_file<float>(grid_file);
		sim_system_ref_->set_temperature(grid);
	} else if (hasParameterVariant(reader, "temperature", "temperature")) {
		std::string key = findParameterVariant(reader, "temperature", "temperature");
		sim_system_ref_->set_temperature(reader.parseValue<float>(key));
	}
	if (hasParameterVariant(reader, "seed", "seed")) {
		std::string key = findParameterVariant(reader, "seed", "seed");
		sim_system_ref_->set_base_seed(static_cast<size_t>(reader.parseValue<long long>(key)));
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

	if (hasParameterVariant(reader, "decompPeriod", "pairlist_rebuild_period")) {
		std::string key = findParameterVariant(reader, "decompPeriod", "pairlist_rebuild_period");
		sim_system_ref_->set_neighbor_list_rebuild_period(reader.parseValue<float>(key));
	}

	if (hasParameterVariant(reader, "reorderPeriod", "reorder_period")) {
		std::string key = findParameterVariant(reader, "reorderPeriod", "reorder_period");
		sim_system_ref_->set_reorder_period(reader.parseValue<int>(key));
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

/// Parse "x y z" into a Vector3; used by rigidBody's position/momentum/etc.
static Vector3 parse_vector3(const std::string& s, const Vector3& fallback = Vector3(0.0f)) {
	const auto toks = tokenize(s);
	if (toks.size() != 3) {
		LOGWARN("Expected 3 values for a Vector3, got '{}' - using default", s);
		return fallback;
	}
	return Vector3(std::stof(toks[0]), std::stof(toks[1]), std::stof(toks[2]));
}

/// Parse 9 whitespace-separated values as 3 human-readable rows into a
/// Matrix3 - Matrix3's own constructor takes column vectors, so this
/// transposes at construction time.
static Matrix3 parse_matrix3_rows(const std::string& s) {
	const auto toks = tokenize(s);
	if (toks.size() != 9) {
		LOGWARN("Expected 9 values (3x3, row-major) for orientation, got '{}' - using identity", s);
		return Matrix3(1.0f);
	}
	std::array<float, 9> m;
	for (int i = 0; i < 9; ++i) {
		m[i] = std::stof(toks[i]);
	}
	return Matrix3(Vector3(m[0], m[3], m[6]), Vector3(m[1], m[4], m[7]), Vector3(m[2], m[5], m[8]));
}

/**
 * @brief Apply a per-entry config list (gridFileScale, ...) to a type's PMF terms
 * @details Legacy semantics: a single value applies to every gridFile entry
 *          (Configuration.cpp default-fills the scale array that way), while a
 *          list must line up one-to-one with the gridFile entries. Deferred
 *          until the whole particle block is parsed so these keys may appear
 *          before or after gridFile.
 */
template<typename Assign>
static void apply_pmf_grid_list(const std::string& key,
								const std::vector<std::string>& toks,
								std::vector<GridTerm>& terms,
								Assign&& assign) {
	if (toks.empty())
		return;
	if (terms.empty()) {
		LOGWARN("{} given but this particle type has no gridFile entries - ignoring", key);
		return;
	}
	if (toks.size() != 1 && toks.size() != terms.size()) {
		LOGWARN("{}: got {} values for {} gridFile entries - ignoring",
				key,
				toks.size(),
				terms.size());
		return;
	}
	for (size_t j = 0; j < terms.size(); ++j) {
		assign(terms[j], toks.size() == 1 ? toks[0] : toks[j]);
	}
}

/**
 * @brief One deferred rigidBody grid scale line: which grid, and the value.
 * @details An empty `grid_key` means the line gave a bare value, which applies
 *          to every grid of that kind (the particle-block gridFileScale form).
 */
struct RbGridScale {
	std::string grid_key;
	float value;
};

/// Parse a rigidBody scale line: `<gridKey> <value>` or a bare `<value>`.
static bool
parse_rb_grid_scale(const std::string& value, const std::string& config_key, RbGridScale& out) {
	const auto toks = tokenize(value);
	try {
		if (toks.size() == 1) {
			out = RbGridScale{{}, std::stof(toks[0])};
			return true;
		}
		if (toks.size() == 2) {
			out = RbGridScale{toks[0], std::stof(toks[1])};
			return true;
		}
	} catch (const std::exception&) {
		LOGWARN("{}: could not parse a numeric scale from '{}' - ignoring", config_key, value);
		return false;
	}
	LOGWARN("{}: expected '<gridKey> <value>' or '<value>', got '{}' - ignoring",
			config_key,
			value);
	return false;
}

/**
 * @brief Apply rigidBody grid scale lines, which may be keyed or positional.
 * @details Legacy scalePotentialGrid/scaleDensityGrid/scalePMF take
 *          `<gridKey> <value>` and apply the value to the grid registered under
 *          that key (RigidBodyType.cu addScaleFactor). Deferred until the block
 *          is fully parsed so these keys may appear either side of the grid
 *          they scale.
 */
template<typename Assign>
static void apply_rb_grid_scales(const std::string& config_key,
								 const std::vector<RbGridScale>& scales,
								 std::vector<GridTerm>& terms,
								 const std::vector<std::string>& grid_keys,
								 Assign&& assign) {
	for (const RbGridScale& s : scales) {
		if (terms.empty()) {
			LOGWARN("{} given but this rigidBody type has no matching grid entries - ignoring",
					config_key);
			continue;
		}
		if (s.grid_key.empty()) {
			for (auto& term : terms) {
				assign(term, s.value);
			}
			continue;
		}
		const auto it = std::find(grid_keys.begin(), grid_keys.end(), s.grid_key);
		if (it == grid_keys.end()) {
			LOGWARN("{}: no grid registered under key '{}' for this rigidBody type - ignoring",
					config_key,
					s.grid_key);
			continue;
		}
		assign(terms[static_cast<size_t>(std::distance(grid_keys.begin(), it))], s.value);
	}
}

/**
 * @brief Parse one rigidBody grid line into (key, filename).
 * @details Legacy syntax is `<gridKey> <file>` (RigidBodyType.cu addGrid), where
 *          the key is the logical name two types must share to be paired for a
 *          grid-grid force - deliberately independent of the filename. A bare
 *          `<file>` is also accepted, with the filename doubling as the key.
 *          Returns false if the line has neither shape.
 */
static bool parse_rb_grid_entry(const std::string& value,
								const std::string& config_key,
								std::string& out_key,
								std::string& out_file) {
	const auto toks = tokenize(value);
	if (toks.size() == 1) {
		out_key = toks[0];
		out_file = toks[0];
		return true;
	}
	if (toks.size() == 2) {
		out_key = toks[0];
		out_file = toks[1];
		return true;
	}
	LOGWARN("{}: expected '<gridKey> <file>' or '<file>', got '{}' - ignoring", config_key, value);
	return false;
}

/// Parse one gridFileBoundaryConditions token; <0 means "keep the grid's own".
static int parse_boundary_condition(std::string tok) {
	std::transform(tok.begin(), tok.end(), tok.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	if (tok == "dirichlet")
		return static_cast<int>(GridBoundaryCondition::Dirichlet);
	if (tok == "neumann")
		return static_cast<int>(GridBoundaryCondition::Neumann);
	if (tok == "periodic")
		return static_cast<int>(GridBoundaryCondition::Periodic);
	LOGWARN("Unrecognized gridFileBoundaryConditions value '{}' - using the grid's own", tok);
	return -1;
}

static void load_particles_file(const std::string& path,
								std::vector<ParticleIO>& out,
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
			ParticleIO p{};
			p.id = std::stoi(toks[1]);
			p.type_name = toks[2];
			p.position.x = std::stof(toks[3]);
			p.position.y = std::stof(toks[4]);
			p.position.z = std::stof(toks[5]);
			// Momentum columns are optional: legacy writes 'ATOM id type x y z'
			// (exactly the 6 tokens the guard above admits).
			if (toks.size() >= 9) {
				p.momentum.x = std::stof(toks[6]);
				p.momentum.y = std::stof(toks[7]);
				p.momentum.z = std::stof(toks[8]);
			}
			out.push_back(p);
		}
		if (line)
			free(line);
	} catch (const std::exception& e) {
		throw_value_error("get_elements: Failed to read particles from '{}': {}", path, e.what());
	}
}

static void load_restart_file(const std::string& path,
							  std::vector<ParticleIO>& out,
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
			ParticleIO p{};
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
		return key == "particle" || key == "rigidBody";
	};

	static const std::unordered_set<std::string_view> particle_field_keys = {
		"num",
		"diffusion",
		"transDamping",
		"mass",
		"gridFile",
		"diffusionGridFile",
		"forceGridFiles",
		"forceGridScale",
		"gridFileScale",
		"gridFileScaleSlope",
		"gridFileBoundaryConditions",
		"gridFileSMD",
		"rigidBodyPotential",
	};

	static const std::unordered_set<std::string_view> rigid_body_field_keys = {
		"mass",
		"num",
		"inertia",
		"transDamping",
		"rotDamping",
		"inputPsf",
		"inputPdb",
		// Point in the template PDB's own frame that becomes this type's body
		// origin; load() subtracts it to get body-frame offsets. Per-type, so
		// every instance shares it - see the fold-in pass below.
		"referencePoint",
		"densityGrid",
		"potentialGrid",
		"pmf",
		"densityGridScale",
		"potentialGridScale",
		"pmfScale",
		"position",
		"orientation",
		"momentum",
		"angularMomentum",
		// Per-instance, like position/orientation - NOT per-type. Legacy stored
		// these on RigidBodyType, which cannot express distinct loads on two
		// instances of the same type.
		"constantForce",
		"constantTorque",
	};

	// Set by the top-level inputRBCoordinates key, applied after the loop (see
	// there for why it cannot be handled inline).
	std::string rb_coordinates_file;

	const bool has_explicit_particle_source =
		std::any_of(params.begin(), params.end(), [](const auto& kv) {
			const auto& k = kv.first;
			return k == "inputParticles" || k == "input_particles" || k == "restartCoordinates" ||
				   k == "restart_coordinates";
		});

	// First pass: parse sequential blocks and inline keys
	for (size_t i = 0; i < params.size();) {
		const auto& [key, value] = params[i];
		if (key == "particle") {
			std::string name = value;
			int num = 0;
			int type_index = static_cast<int>(sim_system_ref_->get_particle_types().size());
			ParticleType ptype(name);
			// Per-gridFile-entry lists, applied once the block is fully parsed so
			// they don't depend on appearing after gridFile (see
			// apply_pmf_grid_list).
			std::vector<std::string> pending_pmf_scale;
			std::vector<std::string> pending_pmf_scale_slope;
			std::vector<std::string> pending_pmf_bc;
			std::vector<std::string> pending_rigid_body_potential_key;

			// Consume following field lines until next header
			++i;
			for (; i < params.size(); ++i) {
				const auto& [k, v] = params[i];
				if (is_block_header(k) || !particle_field_keys.count(k))
					break;
				// Helper lambda to check for variant keys
				auto check_key = [&k](const std::string& camel) { return k == camel; };

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
				} else if (check_key("transDamping")) {
					auto toks = tokenize(v);
					if (toks.size() == 3) {
						ptype.trans_damping.x = std::stof(toks[0]);
						ptype.trans_damping.y = std::stof(toks[1]);
						ptype.trans_damping.z = std::stof(toks[2]);
					} else if (toks.size() == 1) {
						ptype.trans_damping.x = std::stof(toks[0]);
						ptype.trans_damping.y = std::stof(toks[0]);
						ptype.trans_damping.z = std::stof(toks[0]);
					} else {
						LOGWARN("Invalid transDamping format: {}", v);
						ptype.trans_damping = Vector3(0.0f, 0.0f, 0.0f);
					}
					LOGDEBUG("transDamping: {}, {}, {}",
							 ptype.trans_damping.x,
							 ptype.trans_damping.y,
							 ptype.trans_damping.z);
				} else if (k == "mass") {
					LOGDEBUG("mass: {}", v);
					ptype.mass = std::stof(v);
				} else if (check_key("gridFile")) {
					LOGDEBUG("gridFile/grid_file: {}", v);
					for (const auto& fname : tokenize(v)) {
						GridKey grid_key =
							sim_system_ref_->get_grid_manager().add_dense_grid(fname, file_name_);
						if (grid_key.is_valid()) {
							GridTerm term;
							term.grid_id = grid_key.grid_id;
							ptype.pmf_grids.push_back(term);
							ptype.pmf_grid_names.push_back(fname);
							LOGDEBUG("Assigned PMF grid '{}' with grid_id={}",
									 fname,
									 grid_key.grid_id);
						} else {
							LOGWARN("Failed to load grid file '{}'", fname);
						}
					}
				} else if (check_key("diffusionGridFile")) {
					LOGDEBUG("diffusionGridFile/diffusion_grid_file: {}", v);
					GridKey grid_key =
						sim_system_ref_->get_grid_manager().add_dense_grid(v, file_name_);
					if (grid_key.is_valid()) {
						ptype.diffusion_grid_id = grid_key.grid_id;
						ptype.diffusion_grid_name = v;
						LOGDEBUG("Assigned diffusion grid '{}' with grid_id={}",
								 v,
								 grid_key.grid_id);
					} else {
						LOGWARN("Failed to load grid file '{}'", v);
					}
				} else if (check_key("forceGridFiles")) {
					LOGDEBUG("forceGridFiles/force_grid_files: {}", v);
					auto toks = tokenize(v);
					if (toks.size() == 3) {
						ptype.force_grid_names[0] = toks[0];
						ptype.force_grid_names[1] = toks[1];
						ptype.force_grid_names[2] = toks[2];
						GridKey grid_key_x =
							sim_system_ref_->get_grid_manager().add_dense_grid(toks[0], file_name_);
						if (grid_key_x.is_valid()) {
							ptype.force_grid_id[0] = grid_key_x.grid_id;
						} else {
							LOGWARN("Failed to load grid file '{}'", toks[0]);
						}
						GridKey grid_key_y =
							sim_system_ref_->get_grid_manager().add_dense_grid(toks[1], file_name_);
						if (grid_key_y.is_valid()) {
							ptype.force_grid_id[1] = grid_key_y.grid_id;
						} else {
							LOGWARN("Failed to load grid file '{}'", toks[1]);
						}
						GridKey grid_key_z =
							sim_system_ref_->get_grid_manager().add_dense_grid(toks[2], file_name_);
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
				} else if (check_key("forceGridScale")) {
					LOGDEBUG("forceGridScale/force_grid_scale: {}", v);
					auto toks = tokenize(v);
					if (toks.size() == 3) {
						ptype.force_grid_scale = {std::stof(toks[0]),
												  std::stof(toks[1]),
												  std::stof(toks[2])};
					} else if (toks.size() == 1) {
						const float s = std::stof(toks[0]);
						LOGINFO("forceGridScale: got one value '{}', applying it to x, y and z", v);
						ptype.force_grid_scale = {s, s, s};
					} else {
						LOGINFO("forceGridScale: expected three (or one) floating point scale "
								"values for x,y,z but got '{}' - leaving force grids unscaled",
								v);
					}
				} else if (check_key("gridFileScale")) {
					LOGDEBUG("gridFileScale/grid_file_scale: {}", v);
					pending_pmf_scale = tokenize(v);
				} else if (check_key("gridFileScaleSlope")) {
					LOGDEBUG("gridFileScaleSlope/grid_file_scale_slope: {}", v);
					pending_pmf_scale_slope = tokenize(v);
				} else if (check_key("gridFileBoundaryConditions")) {
					LOGDEBUG("gridFileBoundaryConditions: {}", v);
					pending_pmf_bc = tokenize(v);
				} else if (check_key("gridFileSMD")) {
					LOGDEBUG("gridFileSMD/grid_file_smd: {}", v);
					ptype.pmf_smd_freq = std::stoi(v);
				} else if (check_key("rigidBodyPotential")) {
					LOGDEBUG("rigidBodyPotential/rigid_body_potential: {}", v);
					ptype.rigid_body_potential_keys.push_back(v);
				} else {
					(void)0;
				}
			}
			apply_pmf_grid_list("gridFileScale",
								pending_pmf_scale,
								ptype.pmf_grids,
								[](GridTerm& t, const std::string& s) { t.scale = std::stof(s); });
			apply_pmf_grid_list(
				"gridFileScaleSlope",
				pending_pmf_scale_slope,
				ptype.pmf_grids,
				[](GridTerm& t, const std::string& s) { t.scale_slope = std::stof(s); });
			apply_pmf_grid_list("gridFileBoundaryConditions",
								pending_pmf_bc,
								ptype.pmf_grids,
								[](GridTerm& t, const std::string& s) {
									t.boundary_condition = parse_boundary_condition(s);
								});

			sim_system_ref_->get_particle_types().push_back(std::move(ptype));
			// Create placeholder particles for this type, unless explicit
			// per-particle data is supplied elsewhere in the file (see
			// has_explicit_particle_source above).
			if (!has_explicit_particle_source) {
				for (int n = 0; n < num; ++n) {
					ParticleIO p{};
					p.id = static_cast<int>(init_particles_.size());
					p.type_name = name;
					init_particles_.push_back(p);
				}
			}
			continue; // i already at next header or end
		}

		if (key == "rigidBody") {
			std::string name = value;
			RigidBodyType rbtype;
			rbtype.name = name;
			RigidBodyIO rb_io{};
			rb_io.orientation = Matrix3(1.0f);

			// Deferred like particle's gridFileScale/gridFileScaleSlope - these
			// keys may appear before or after the grid they scale, and each of
			// the three grid lists (density/potential/pmf) is scaled
			// independently. One entry per config line, not one per token: a
			// type normally scales several grids with one `<gridKey> <value>`
			// line each, so these keys legitimately repeat within a block.
			std::vector<RbGridScale> pending_density_scale;
			std::vector<RbGridScale> pending_potential_scale;
			std::vector<RbGridScale> pending_pmf_scale;

			// Attached/cosmetic particle template. Deferred to the end of the
			// block because referencePoint may appear either side of the file
			// keys, and load() needs all three at once.
			int instance_count = 1;
			std::string template_pdb;
			std::string template_psf;
			Vector3 reference_point{};

			++i;
			for (; i < params.size(); ++i) {
				const auto& [k, v] = params[i];
				if (is_block_header(k) || !rigid_body_field_keys.count(k))
					break;
				auto check_key = [&k](const std::string& camel) { return k == camel; };

				if (k == "mass") {
					rbtype.mass = std::stof(v);
				} else if (k == "num") {
					instance_count = std::stoi(v);
					if (instance_count < 0) {
						throw_value_error("rigidBody '%s': num must be >= 0, got '%s'",
										  name.c_str(),
										  v.c_str());
					}
				} else if (check_key("inputPdb")) {
					template_pdb = resolve_file_path(v, file_name_);
				} else if (check_key("inputPsf")) {
					template_psf = resolve_file_path(v, file_name_);
				} else if (check_key("referencePoint")) {
					reference_point = parse_vector3(v);
				} else if (k == "inertia") {
					rbtype.inertia = parse_vector3(v);
				} else if (check_key("transDamping")) {
					rbtype.trans_damping = parse_vector3(v);
				} else if (check_key("rotDamping")) {
					rbtype.rot_damping = parse_vector3(v);
				} else if (check_key("densityGrid")) {
					// Legacy syntax is `<gridKey> <file>`: the key is the logical
					// name two types must share to be paired for a grid-grid
					// force, independent of which file each one loads. A bare
					// `<file>` still works, with the filename doubling as the key.
					std::string gkey, gfile;
					if (parse_rb_grid_entry(v, "densityGrid", gkey, gfile)) {
						GridKey grid_key =
							sim_system_ref_->get_grid_manager().add_dense_grid(gfile, file_name_);
						if (grid_key.is_valid()) {
							rbtype.density_grids.push_back(GridTerm{grid_key.grid_id});
							rbtype.density_grid_keys.push_back(gkey);
						} else {
							LOGWARN("Failed to load rigid body density grid file '{}'", gfile);
						}
					}
				} else if (check_key("potentialGrid")) {
					std::string gkey, gfile;
					if (parse_rb_grid_entry(v, "potentialGrid", gkey, gfile)) {
						GridKey grid_key =
							sim_system_ref_->get_grid_manager().add_dense_grid(gfile, file_name_);
						if (grid_key.is_valid()) {
							rbtype.potential_grids.push_back(GridTerm{grid_key.grid_id});
							rbtype.potential_grid_keys.push_back(gkey);
						} else {
							LOGWARN("Failed to load rigid body potential grid file '{}'", gfile);
						}
					}
				} else if (check_key("pmf")) {
					std::string gkey, gfile;
					if (parse_rb_grid_entry(v, "pmf", gkey, gfile)) {
						GridKey grid_key =
							sim_system_ref_->get_grid_manager().add_dense_grid(gfile, file_name_);
						if (grid_key.is_valid()) {
							rbtype.pmf_grids.push_back(GridTerm{grid_key.grid_id});
							rbtype.pmf_keys.push_back(gkey);
						} else {
							LOGWARN("Failed to load rigid body pmf grid file '{}'", gfile);
						}
					}
				} else if (check_key("densityGridScale")) {
					RbGridScale s;
					if (parse_rb_grid_scale(v, "densityGridScale", s))
						pending_density_scale.push_back(s);
				} else if (check_key("potentialGridScale")) {
					RbGridScale s;
					if (parse_rb_grid_scale(v, "potentialGridScale", s))
						pending_potential_scale.push_back(s);
				} else if (check_key("pmfScale")) {
					RbGridScale s;
					if (parse_rb_grid_scale(v, "pmfScale", s))
						pending_pmf_scale.push_back(s);
				} else if (k == "position") {
					rb_io.position = parse_vector3(v);
				} else if (k == "orientation") {
					rb_io.orientation = parse_matrix3_rows(v);
					rb_io.has_orientation = true;
				} else if (k == "momentum") {
					rb_io.momentum = parse_vector3(v);
				} else if (check_key("angularMomentum")) {
					rb_io.angular_momentum = parse_vector3(v);
				} else if (check_key("constantForce")) {
					rb_io.external_force = parse_vector3(v);
				} else if (check_key("constantTorque")) {
					rb_io.external_torque = parse_vector3(v);
				} else {
					// Recognized but not yet wired into RigidBodyType storage in this branch
					(void)0;
				}
			}

			apply_rb_grid_scales("densityGridScale",
								 pending_density_scale,
								 rbtype.density_grids,
								 rbtype.density_grid_keys,
								 [](GridTerm& t, float s) { t.scale = s; });
			apply_rb_grid_scales("potentialGridScale",
								 pending_potential_scale,
								 rbtype.potential_grids,
								 rbtype.potential_grid_keys,
								 [](GridTerm& t, float s) { t.scale = s; });
			apply_rb_grid_scales("pmfScale",
								 pending_pmf_scale,
								 rbtype.pmf_grids,
								 rbtype.pmf_keys,
								 [](GridTerm& t, float s) { t.scale = s; });

			// Attached/cosmetic particles come from a PDB+PSF template pair.
			// load() resolves each ATT atom's resname against the already-declared
			// ParticleTypes, so particle blocks must precede rigidBody blocks;
			// fail with that explanation rather than the raw "unknown resname".
			if (!template_pdb.empty() || !template_psf.empty()) {
				if (template_pdb.empty() || template_psf.empty()) {
					throw_value_error("rigidBody '%s': inputPdb and inputPsf must be given "
									  "together (got only one)",
									  name.c_str());
				}
				if (sim_system_ref_->get_particle_types().empty()) {
					throw_value_error("rigidBody '%s': inputPdb/inputPsf need particle types to "
									  "resolve atom resnames against, but none are declared yet - "
									  "move the particle blocks above this rigidBody block",
									  name.c_str());
				}
				RigidBodyPdbPsfReader::load(rbtype,
											template_pdb,
											template_psf,
											reference_point,
											sim_system_ref_->get_particle_types());
			}

			// Insertion order here is exactly what assign_rigid_body_type_ids()
			// later assigns as RigidBodyType::id, so this is safe to resolve
			// immediately (unlike ParticleIO, which defers via type_name since
			// its data can arrive from a separate file, not necessarily in
			// insertion order).
			const int type_id = static_cast<int>(sim_system_ref_->get_rigid_body_types().size());
			sim_system_ref_->get_rigid_body_types().push_back(std::move(rbtype));

			// One type, `num` instances. They share the block's position/
			// orientation/momentum here; inputRBCoordinates (a top-level key,
			// applied after all blocks are parsed) overwrites them per instance
			// when present, matching legacy loadRBCoordinates.
			for (int n = 0; n < instance_count; ++n) {
				RigidBodyIO inst = rb_io;
				inst.id = static_cast<int>(init_rigid_bodies_.size());
				inst.type_id = type_id;
				init_rigid_bodies_.push_back(inst);
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
		} else if (key == "inputRBCoordinates" || key == "input_rb_coordinates") {
			rb_coordinates_file = resolve_file_path(value, file_name_);
		} else if (key == "rigidBodyGridGridPeriod" || key == "rigid_body_grid_grid_period") {
			sim_system_ref_->set_rb_update_period(std::stoi(value));
		}

		++i;
	}

	if (!rb_coordinates_file.empty()) {
		RigidBodyCoordReader::load(init_rigid_bodies_, rb_coordinates_file);
	}

	fold_in_attached_particles();
}

void ConfigParser::fold_in_attached_particles() {
	const auto& types = sim_system_ref_->get_rigid_body_types();

	size_t total_attached = 0;
	for (const RigidBodyIO& rb : init_rigid_bodies_) {
		if (rb.type_id >= 0 && static_cast<size_t>(rb.type_id) < types.size()) {
			total_attached += types[rb.type_id].attached_particle.size();
		}
	}
	if (total_attached == 0) {
		return;
	}

	const size_t num_regular = init_particles_.size();
	init_particles_.reserve(num_regular + total_attached);

	for (RigidBodyIO& rb : init_rigid_bodies_) {
		if (rb.type_id < 0 || static_cast<size_t>(rb.type_id) >= types.size()) {
			continue;
		}
		const RigidBodyType& type = types[rb.type_id];
		if (type.attached_particle.empty()) {
			continue;
		}

		rb.attached_start = static_cast<int>(init_particles_.size());
		rb.attached_count = static_cast<int>(type.attached_particle.size());

		for (const ParticleIO& tmpl : type.attached_particle) {
			ParticleIO p = tmpl;
			p.id = static_cast<int>(init_particles_.size());
			p.attached_rigid_body_id = rb.id;
			p.position = rb.orientation * tmpl.position + rb.position;
			init_particles_.push_back(p);
		}
	}

	LOGINFO("ConfigParser: {} regular particle(s) + {} rigid-body attached particle(s) = {} total",
			num_regular,
			total_attached,
			init_particles_.size());
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
			} else if (key == "seed") {
				sim_system_ref_->set_base_seed(pybind11::cast<long long>(value));
			} else if (key == "num_steps" || key == "steps") {
				sim_system_ref_->set_num_steps(pybind11::cast<int>(value));
			} else if (key == "output_period") {
				sim_system_ref_->set_output_period(pybind11::cast<float>(value));
			} else if (key == "energy_output_period") {
				sim_system_ref_->set_energy_output_period(pybind11::cast<float>(value));
			} else if (key == "neighbor_list_rebuild_period") {
				sim_system_ref_->set_neighbor_list_rebuild_period(pybind11::cast<float>(value));
			} else if (key == "reorder_period") {
				sim_system_ref_->set_reorder_period(pybind11::cast<int>(value));
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
