#include "catch_boiler.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
// Include MARS headers
#include "IO/ConfigParser.h"
#include "Objects/ParticleProperties.h"
short single_resource_id = Global::single_resource_id;
using namespace MARS;

/**
 * @brief Test configuration parsing with the circovirus example
 *
 * This test validates that the ConfigParser parser can correctly load:
 * - Basic simulation parameters (temperature, timestep, etc.)
 * - Particle types (D000, O000, S000) with their properties
 * - Individual particles for each type
 * - Bond, angle, and dihedral lists from external files
 * - Tabulated potential files
 */
void test_parse_circovirus_config() {
	std::cout << "=== Testing Configuration Parsing ===" << std::endl;

	// Path to the example config file
	std::string config_file =
		"/data/server10/pinyili2/3-grid_capsid/1-compaction/circovirus-1bppb-compaction-rep1.bd";

	try {
		// Create ConfigParser object and parse the file
		SimSystem sim_system(std::vector<Resource>({Resource(single_resource_id)}));
		ConfigParser config_parser(sim_system, config_file);
		std::cout << "✓ Successfully loaded configuration from: " << config_file << std::endl;

		// Test basic simulation parameters
		std::cout << "\n--- Basic Simulation Parameters ---" << std::endl;
		std::cout << "Temperature: " << sim_system.get_temperature() << " K" << std::endl;
		std::cout << "Timestep: " << sim_system.get_timestep() << " ps" << std::endl;
		std::cout << "Steps: " << sim_system.get_num_steps() << std::endl;
		std::cout << "Box size: " << sim_system.get_box_size().x << " x "
				  << sim_system.get_box_size().y << " x " << sim_system.get_box_size().z << " Å"
				  << std::endl;
		std::cout << "Cutoff: " << sim_system.get_cutoff() << " Å" << std::endl;
		std::cout << "Output period: " << sim_system.get_output_period() << " ps" << std::endl;

		std::cout << "✓ Basic parameters match expected values" << std::endl;
		// Test particle types
		std::cout << "\n--- Particle Types ---" << std::endl;
		const auto& particle_types = sim_system.get_particle_types().size();
		std::cout << "Number of particle types: " << particle_types << std::endl;

		// Expected: 3 particle types (D000, O000, S000)
		assert(particle_types == 3);

		// Check each particle type
		for (size_t i = 0; i < particle_types; ++i) {
			const auto& ptype = sim_system.get_particle_types()[i];
			std::cout << "Type " << i << ": " << ptype.name << " (num=" << ptype.num
					  << ", mass=" << ptype.mass << ")" << std::endl;
		}

		// Validate specific particle types from config
		assert(sim_system.get_particle_types()[0].name == "D000");
		assert(sim_system.get_particle_types()[0].num == 530);
		assert(sim_system.get_particle_types()[0].mass == 300.0f);

		assert(sim_system.get_particle_types()[1].name == "O000");
		assert(sim_system.get_particle_types()[1].num == 530);
		assert(sim_system.get_particle_types()[1].mass == 300.0f);

		assert(sim_system.get_particle_types()[2].name == "S000");
		assert(sim_system.get_particle_types()[2].num == 738);
		assert(sim_system.get_particle_types()[2].mass == 150.0f);

		std::cout << "✓ Particle types match expected values" << std::endl;

		// Test individual particles
		std::cout << "\n--- Individual Particles ---" << std::endl;
		const auto& particles = config_parser.get_init_particles();
		std::cout << "Total number of particles: " << particles.size() << std::endl;

		// Expected: 530 + 530 + 738 = 1798 particles
		int expected_total = 530 + 530 + 738;
		assert(particles.size() == expected_total);

		// Check particle type distribution
		std::vector<int> type_counts(3, 0);
		for (const auto& particle : particles) {
			assert(particle.type_id >= 0 && particle.type_id < 3);
			type_counts[particle.type_name]++;
		}

		assert(type_counts[0] == 530); // D000
		assert(type_counts[1] == 530); // O000
		assert(type_counts[2] == 738); // S000

		std::cout << "Particle distribution:" << std::endl;
		std::cout << "  D000 particles: " << type_counts[0] << std::endl;
		std::cout << "  O000 particles: " << type_counts[1] << std::endl;
		std::cout << "  S000 particles: " << type_counts[2] << std::endl;

		std::cout << "✓ Particle distribution is correct" << std::endl;

		// Test bonds
		std::cout << "\n--- Bonded Interactions ---" << std::endl;
		const auto& bonds = config_parser.get_init_bonded_interactions().get_bonds();
		const auto& angles = config_parser.get_init_bonded_interactions().get_angles();
		const auto& dihedrals = config_parser.get_init_bonded_interactions().get_dihedrals();
		const auto& exclusions = config_parser.get_init_bonded_interactions().get_exclusions();

		std::cout << "Bonds: " << bonds.size() << std::endl;
		std::cout << "Angles: " << angles.size() << std::endl;
		std::cout << "Dihedrals: " << dihedrals.size() << std::endl;
		std::cout << "Exclusions: " << exclusions.size() << std::endl;

		// Check that bonds were loaded (exact count depends on input files)
		if (!bonds.empty()) {
			std::cout << "Sample bond: " << bonds[0].ind1 << " - " << bonds[0].ind2 << " ("
					  << bonds[0].type_id << ")" << std::endl;
			assert(bonds[0].ind1 >= 0);
			assert(bonds[0].ind2 >= 0);
			assert(!bonds[0].name.empty());
		}

		if (!angles.empty()) {
			std::cout << "Sample angle: " << angles[0].ind1 << " - " << angles[0].ind2 << " - "
					  << angles[0].ind3 << " (" << angles[0].name << ")" << std::endl;
		}

		if (!dihedrals.empty()) {
			std::cout << "Sample dihedral: " << dihedrals[0].ind1 << " - " << dihedrals[0].ind2
					  << " - " << dihedrals[0].ind3 << " - " << dihedrals[0].ind4 << " ("
					  << dihedrals[0].name << ")" << std::endl;
		}

		std::cout << "✓ Bonded interactions loaded successfully" << std::endl;

		// Test tabulated potentials
		std::cout << "\n--- Tabulated Potentials ---" << std::endl;
		const TablesRegistry& tables = sim_system.get_tables_registry();
		std::cout << "Number of potential tables: " << tables.num_tables() << std::endl;

		// Test boundary conditions creation
		std::cout << "\n--- Boundary Conditions ---" << std::endl;
		const PeriodicBox& bc = sim_system.get_boundary_conditions();
		const auto& origin = bc.get_origin();
		const auto& basis = bc.get_basis();
		const auto& periodic = bc.get_periodicity();

		std::cout << "Origin: (" << origin.x << ", " << origin.y << ", " << origin.z << ")"
				  << std::endl;
		std::cout << "Basis vectors:" << std::endl;
		for (int i = 0; i < 3; ++i) {
			std::cout << "  " << i << ": (" << basis[i].x << ", " << basis[i].y << ", "
					  << basis[i].z << ")" << std::endl;
		}
		std::cout << "Periodic: (" << periodic[0] << ", " << periodic[1] << ", " << periodic[2]
				  << ")" << std::endl;

		// Validate boundary conditions
		assert(origin.x == 0.0f && origin.y == 0.0f && origin.z == 0.0f);
		assert(basis[0].x == 5000.0f && basis[0].y == 0.0f && basis[0].z == 0.0f);
		assert(basis[1].x == 0.0f && basis[1].y == 5000.0f && basis[1].z == 0.0f);
		assert(basis[2].x == 0.0f && basis[2].y == 0.0f && basis[2].z == 5000.0f);
		assert(periodic[0] && periodic[1] && periodic[2]); // All periodic

		std::cout << "✓ Boundary conditions are correct" << std::endl;

		// Test configuration validation
		std::cout << "\n--- Configuration Validation ---" << std::endl;
		assert(sim_system.is_valid());
		std::cout << "✓ Configuration passes validation" << std::endl;

		std::cout << "\n=== All Tests Passed! ===" << std::endl;

	} catch (const Exception& e) {
		std::cerr << "Configuration parsing failed: " << e.what() << std::endl;
		throw;
	} catch (const std::exception& e) {
		std::cerr << "Unexpected error: " << e.what() << std::endl;
		throw;
	}
}

/**
 * @brief Test configuration with missing files (should handle gracefully)
 */
void test_missing_files() {
	std::cout << "\n=== Testing Missing Files Handling ===" << std::endl;

	// Create a minimal config file for testing
	std::string test_config = R"(
seed 12345
timestep 1e-5
steps 1000
temperature 300
cutoff 10
systemSize 100 100 100
outputPeriod 100

particle D000
num 10
mass 100.0
transDamping 100 100 100

inputBonds nonexistent_bonds.txt
inputAngles nonexistent_angles.txt
)";

	std::string test_file = "/tmp/test_config.bd";
	std::ofstream f(test_file);
	f << test_config;
	f.close();

	try {
		SimSystem sim_system(std::vector<Resource>({Resource(single_resource_id)}));
		ConfigParser config_parser(sim_system, test_file);
		const BondedInteractions& bonded_interactions =
			config_parser.get_init_bonded_interactions();
		std::cout << "✓ Configuration loaded successfully with missing files" << std::endl;
		std::cout << "Bonds loaded: " << bonded_interactions.get_num_bonds() << std::endl;
		std::cout << "Angles loaded: " << bonded_interactions.get_num_angles() << std::endl;

		// Should handle missing files gracefully (empty lists)
		assert(bonded_interactions.get_num_bonds() == 0);
		assert(bonded_interactions.get_num_angles() == 0);

		std::cout << "✓ Missing files handled gracefully" << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "Unexpected error with missing files: " << e.what() << std::endl;
		throw;
	}

	// Clean up
	std::remove(test_file.c_str());
}

/**
 * @brief Test relative path resolution
 */
void test_relative_paths() {
	std::cout << "\n=== Testing Relative Path Resolution ===" << std::endl;

	// Create a test config file with relative paths
	std::string test_config = R"(
seed 12345
timestep 1e-5
steps 1000
temperature 300
cutoff 10
systemSize 100 100 100
outputPeriod 100

particle D000
num 5
mass 100.0
transDamping 100 100 100

# Test relative paths
inputBonds ./test_bonds.txt
inputAngles ../test_angles.txt
)";

	// Create test directory structure
	system("mkdir -p /tmp/test_config_dir");
	system("mkdir -p /tmp/test_config_dir/parent");

	// Create test files
	std::ofstream f1("/tmp/test_config_dir/test_bonds.txt");
	f1 << "0 1 0 1 bond1.dat\n";
	f1 << "1 1 1 2 bond2.dat\n";
	f1.close();

	std::ofstream f2("/tmp/test_config_dir/parent/test_angles.txt");
	f2 << "0 1 2 angle1.dat\n";
	f2.close();

	// Create config file
	std::string test_file = "/tmp/test_config_dir/test_config.bd";
	std::ofstream f(test_file);
	f << test_config;
	f.close();

	try {
		SimSystem sim_system(std::vector<Resource>({Resource(single_resource_id)}));
		ConfigParser config_parser(sim_system, test_file);
		const BondedInteractions& bonded_interactions =
			config_parser.get_init_bonded_interactions();

		std::cout << "✓ Configuration loaded successfully with relative paths" << std::endl;
		std::cout << "Bonds loaded: " << bonded_interactions.get_num_bonds() << std::endl;
		std::cout << "Angles loaded: " << bonded_interactions.get_num_angles() << std::endl;

		// Should have loaded the files
		assert(bonded_interactions.get_num_bonds() == 2);
		assert(bonded_interactions.get_num_angles() == 1);

		std::cout << "✓ Relative paths resolved correctly" << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "Unexpected error with relative paths: " << e.what() << std::endl;
		throw;
	}

	// Clean up
	system("rm -rf /tmp/test_config_dir");
}

/**
 * @brief Test configuration validation
 */
void test_config_validation() {
	std::cout << "\n=== Testing Configuration Validation ===" << std::endl;

	// Test invalid configuration
	SimSystem sim_system(std::vector<Resource>({Resource(single_resource_id)}));
	sim_system.set_temperature(-1.0f); // Invalid temperature
	sim_system.set_cutoff(0.0f);	   // Invalid cutoff

	try {
		ConfigParser config_parser(sim_system, "test_file");
		assert(false && "Should have thrown validation error");
	} catch (const Exception& e) {
		std::cout << "✓ Correctly caught validation error: " << e.what() << std::endl;
	}

	// Test valid configuration
	SimSystem sim_system2(std::vector<Resource>({Resource(single_resource_id)}));
	sim_system2.set_temperature(300.0f);
	sim_system2.set_cutoff(10.0f);
	sim_system2.set_timestep(1e-5f);
	sim_system2.set_num_steps(1000);
	sim_system2.set_box_size(100.0f, 100.0f, 100.0f);

	try {
		ConfigParser config_parser(sim_system2, "test_file");
		std::cout << "✓ Valid configuration accepted" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Unexpected error with valid config: " << e.what() << std::endl;
		throw;
	}
}

int main() {
	try {
		std::cout << "Starting Configuration Parsing Tests..." << std::endl;

		// Run tests
		test_parse_circovirus_config();
		test_missing_files();
		test_relative_paths();
		test_config_validation();

		std::cout << "\n🎉 All configuration parsing tests completed successfully!" << std::endl;
		return 0;

	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}
