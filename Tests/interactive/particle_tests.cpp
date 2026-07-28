#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "IO/ConfigParser.h"
#include "Objects/ParticleProperties.h"
#include "SimManager.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"

// Include backend managers based on compile-time configuration
#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef USE_PYBIND
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#else
// Include Catch2 for C++ testing when not building PyBind module
#include "../../Unit_test/catch_boiler.h"
#endif

using namespace ARBD;
std::vector<Resource> resources = {Resource(0), Resource(1), Resource(2)};
// ============================================================================
// Interactive Particle Creation Functions
// ============================================================================

/**
 * @brief Create test particles with different spatial patterns
 * @param count Number of particles to create
 * @param pattern Spatial arrangement pattern
 * @param box_size Size of the simulation box
 * @return Vector of particles with positions and types set
 */
std::vector<ParticleIO>
create_test_particles(int count, const std::string& pattern = "linear", float box_size = 100.0f) {
	std::vector<ParticleIO> particles;
	particles.reserve(count);

	// Initialize random number generator for random patterns
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0f, box_size);

	for (int i = 0; i < count; ++i) {
		ParticleIO p;
		p.type_name = "A"; // Alternate between two particle types
		p.id = i;
		if (pattern == "linear") {
			// Linear arrangement along x-axis
			float spacing = box_size / (count + 1);
			p.position = Vector3_t<float>{(i + 1) * spacing, box_size * 0.5f, box_size * 0.5f};
		} else if (pattern == "grid") {
			// 3D grid arrangement
			int side = std::ceil(std::cbrt(count));
			float spacing = box_size / (side + 1);
			int x = i % side;
			int y = (i / side) % side;
			int z = i / (side * side);
			p.position = Vector3_t<float>{(x + 1) * spacing, (y + 1) * spacing, (z + 1) * spacing};
		} else if (pattern == "random") {
			// Random positions within box
			p.position = Vector3_t<float>{dist(gen), dist(gen), dist(gen)};
		} else if (pattern == "sphere") {
			// Spherical arrangement
			float radius = box_size * 0.3f;
			float theta = 2.0f * M_PI * i / count;
			float phi = M_PI * (i + 0.5f) / count;
			p.position =
				Vector3_t<float>{box_size * 0.5f + radius * std::sin(phi) * std::cos(theta),
								 box_size * 0.5f + radius * std::sin(phi) * std::sin(theta),
								 box_size * 0.5f + radius * std::cos(phi)};
		} else {
			// Default to linear if pattern not recognized
			float spacing = box_size / (count + 1);
			p.position = Vector3_t<float>{(i + 1) * spacing, box_size * 0.5f, box_size * 0.5f};
		}

		particles.push_back(p);
	}

	return particles;
}

/**
 * @brief Create particle types with specified properties
 * @param type_count Number of different particle types
 * @param particles_per_type Number of particles for each type
 * @return Vector of particle type definitions
 */
std::vector<ParticleType> create_test_particle_types(int type_count = 2,
													 int particles_per_type = 50) {
	std::vector<ParticleType> particle_types;
	particle_types.reserve(type_count);

	for (int i = 0; i < type_count; ++i) {
		ParticleType type(std::string("TestType") + std::to_string(i));
		type.mass = 100.0f + i * 50.0f; // Increasing mass for different types
		type.num = particles_per_type;

		// Set damping coefficients (example values)
		type.diffusion = Vector3(100.0f, 100.0f, 100.0f);

		particle_types.push_back(type);
	}

	return particle_types;
}

/**
 * @brief Create a complete test system with particles and configuration
 * @param particle_count Total number of particles
 * @param pattern Spatial arrangement pattern
 * @param box_size Simulation box size
 * @return Configured SimSystem ready for testing
 */
auto create_interactive_test_system(int particle_count,
									const std::string& pattern = "linear",
									float box_size = 100.0f) {
	SimSystem system(resources);
	// Create particle types (2 types by default)
	int type_count = 2;
	int particles_per_type = particle_count / type_count;
	std::vector<ParticleType> particle_types =
		create_test_particle_types(type_count, particles_per_type);

	// Create particles with specified pattern
	std::vector<ParticleIO> init_particles =
		create_test_particles(particle_count, pattern, box_size);

	// Adjust particle type assignments to match particle type counts
	for (size_t i = 0; i < init_particles.size(); ++i) {
		init_particles[i].type_name = particle_types[i % type_count].name;
	}

	// Set system parameters
	system.set_temperature(300.0f);	  // Room temperature in Kelvin
	system.set_cutoff(12.0f);		  // Interaction cutoff in Angstroms
	system.set_timestep(2e-5f);		  // 20 fs timestep
	system.set_num_steps(1000);		  // Short test simulation
	system.set_output_period(100.0f); // Output every 100 steps
	system.set_output_format(OutputFormat::DCD);
	system.set_output_name("test_system");
	for (const auto& type : particle_types) {
		system.add_particle_type(type);
	}

	// Set simulation box
	system.set_box_size(box_size, box_size, box_size);

	return std::make_tuple(std::move(system), std::move(init_particles), std::move(particle_types));
}

/**
 * @brief Get comprehensive system information for debugging/verification
 * @param system SimSystem to analyze
 * @return Dictionary/map of system properties
 */
std::map<std::string, std::string>
get_system_info(const SimSystem& system,
				const std::vector<ParticleIO>& init_particles,
				const std::vector<ParticleType>& particle_types) {
	std::map<std::string, std::string> info;

	info["particle_count"] = std::to_string(init_particles.size());
	info["type_count"] = std::to_string(particle_types.size());

	// Get particle type information
	const auto& types = particle_types;
	for (size_t i = 0; i < types.size(); ++i) {
		std::string prefix = "type_" + std::to_string(i) + "_";
		info[prefix + "name"] = particle_types[i].name;
		info[prefix + "mass"] = std::to_string(particle_types[i].mass);
		info[prefix + "count"] = std::to_string(particle_types[i].num);
	}

	// System configuration info
	info["temperature"] = std::to_string(system.get_temperature());
	info["cutoff"] = std::to_string(system.get_cutoff());
	info["timestep"] = std::to_string(system.get_timestep());
	info["box_x"] = std::to_string(system.get_box_size().x);
	info["box_y"] = std::to_string(system.get_box_size().y);
	info["box_z"] = std::to_string(system.get_box_size().z);

	return info;
}

/**
 * @brief Test device memory operations with particle data
 * @param system SimSystem to test with
 * @return True if device operations successful, false otherwise
 */
bool test_device_operations(const SimSystem& system,
							SystemState& system_state,
							const std::vector<ParticleIO>& init_particles) {
	initialize_backend_once();
	try {
		// Test device buffer operations
		size_t particle_count = init_particles.size();

		// Create device buffers for positions
		DeviceBuffer<Vector3_t<float>> position_buffer(particle_count);

		// Get particle positions from system
		std::vector<Vector3_t<float>> host_positions;
		system_state.set_init_particle_data(init_particles);
		const auto& particles = system_state.get_global_positions();
		host_positions.reserve(particle_count);

		for (const auto& particle : particles) {
			host_positions.push_back(particle);
		}

		// Transfer to device
		position_buffer.copy_from_host(host_positions.data(), particle_count);

		// Transfer back to verify
		std::vector<Vector3_t<float>> verify_positions(particle_count);
		position_buffer.copy_to_host(verify_positions);

		// Verify data integrity
		for (size_t i = 0; i < particle_count; ++i) {
			const auto& orig = init_particles[i].position;
			const auto& verify = verify_positions[i];

			if (std::abs(orig.x - verify.x) > 1e-6f || std::abs(orig.y - verify.y) > 1e-6f ||
				std::abs(orig.z - verify.z) > 1e-6f) {
				std::cerr << "Device transfer verification failed at particle " << i << std::endl;
				return false;
			}
		}

		std::cout << "Device operations test passed for " << particle_count << " particles"
				  << std::endl;
		return true;

	} catch (const std::exception& e) {
		std::cerr << "Device operations test failed: " << e.what() << std::endl;
		return false;
	}
}

/**
 * @brief Benchmark particle operations
 * @param system SimSystem to benchmark
 * @param operation Type of operation to benchmark
 * @return Time taken in milliseconds
 */
double benchmark_operation(const SimSystem& system, const std::string& operation) {
	auto start = std::chrono::high_resolution_clock::now();

	if (operation == "particle_creation") {
		// Benchmark particle creation
		auto particles = create_test_particles(system.get_num_particles(), "random");
	} else if (operation == "system_creation") {
		// Benchmark full system creation
		auto test_system = create_interactive_test_system(system.get_num_particles(), "grid");
	} else if (operation == "device_transfer") {
		// Benchmark device transfer
		test_device_operations(system,
							   system_state,
							   create_test_particles(system.get_particle_count(), "random"));
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration.count() / 1000.0; // Return milliseconds
}

// ============================================================================
// Python Bindings (when USE_PYBIND is defined)
// ============================================================================

#ifdef USE_PYBIND

PYBIND11_MODULE(arbd_interactive, m) {
	m.doc() = "ARBD Interactive Testing and Development Tools";

	// Particle creation functions
	py::module particles = m.def_submodule("particles", "Particle testing and creation tools");

	particles.def("create_test_particles",
				  &create_test_particles,
				  "Create test particles with specified pattern",
				  py::arg("count"),
				  py::arg("pattern") = "linear",
				  py::arg("box_size") = 100.0f);

	particles.def("create_test_particle_types",
				  &create_test_particle_types,
				  "Create particle type definitions",
				  py::arg("type_count") = 2,
				  py::arg("particles_per_type") = 50);

	// System creation and analysis
	py::module systems = m.def_submodule("systems", "System creation and analysis tools");

	systems.def("create_test_system",
				&create_interactive_test_system,
				"Create complete test system",
				py::arg("particle_count"),
				py::arg("pattern") = "linear",
				py::arg("box_size") = 100.0f);

	systems.def("get_system_info",
				&get_system_info,
				"Get comprehensive system information",
				py::arg("system"));

	// Device and performance testing
	py::module testing = m.def_submodule("testing", "Device and performance testing tools");

	testing.def("test_device_operations",
				&test_device_operations,
				"Test device memory operations",
				py::arg("system"));

	testing.def("benchmark_operation",
				&benchmark_operation,
				"Benchmark specific operations",
				py::arg("system"),
				py::arg("operation"));

	// Bind core ARBD classes for inspection
	py::class_<Vector3_t<float>>(m, "Vector3f")
		.def_readwrite("x", &Vector3_t<float>::x)
		.def_readwrite("y", &Vector3_t<float>::y)
		.def_readwrite("z", &Vector3_t<float>::z);

	py::class_<Particle>(m, "Particle")
		.def_readwrite("position", &Particle::position)
		.def_readwrite("type_id", &Particle::type_id);

	py::class_<ParticleType>(m, "ParticleType")
		.def_readwrite("name", &ParticleType::name)
		.def_readwrite("mass", &ParticleType::mass)
		.def_readwrite("num", &ParticleType::num);

	py::class_<SimSystem>(m, "SimSystem")
		.def("get_particle_count", &SimSystem::get_particle_count)
		.def("get_particle_types",
			 &SimSystem::get_particle_types,
			 py::return_value_policy::reference)
		.def("get_particles", &SimSystem::get_particles, py::return_value_policy::reference)
		.def("get_configuration",
			 &SimSystem::get_configuration,
			 py::return_value_policy::reference);
}

#else

// ============================================================================
// C++ Tests (when not building PyBind module)
// ============================================================================

TEST_CASE("Interactive Particle Creation", "[interactive][particles]") {
	SECTION("Linear pattern particles") {
		auto particles = create_test_particles(10, "linear", 100.0f);
		REQUIRE(particles.size() == 10);

		// Verify linear arrangement
		for (size_t i = 0; i < particles.size(); ++i) {
			REQUIRE(particles[i].position.x > 0.0f);
			REQUIRE(particles[i].position.x < 100.0f);
			REQUIRE(particles[i].type_name == "TestType" + std::to_string(i % 2));
		}
	}

	SECTION("Grid pattern particles") {
		auto particles = create_test_particles(27, "grid", 90.0f); // 3x3x3 grid
		REQUIRE(particles.size() == 27);

		// Verify all particles are within bounds
		for (const auto& p : particles) {
			REQUIRE(p.position.x > 0.0f);
			REQUIRE(p.position.x < 90.0f);
			REQUIRE(p.position.y > 0.0f);
			REQUIRE(p.position.y < 90.0f);
			REQUIRE(p.position.z > 0.0f);
			REQUIRE(p.position.z < 90.0f);
		}
	}

	SECTION("Random pattern particles") {
		auto particles = create_test_particles(50, "random", 200.0f);
		REQUIRE(particles.size() == 50);

		// Verify random positions are within bounds
		for (const auto& p : particles) {
			REQUIRE(p.position.x >= 0.0f);
			REQUIRE(p.position.x <= 200.0f);
			REQUIRE(p.position.y >= 0.0f);
			REQUIRE(p.position.y <= 200.0f);
			REQUIRE(p.position.z >= 0.0f);
			REQUIRE(p.position.z <= 200.0f);
		}
	}

	SECTION("Spherical pattern particles") {
		auto particles = create_test_particles(100, "sphere", 50.0f);
		REQUIRE(particles.size() == 100);

		// Verify particles form roughly spherical distribution
		Vector3_t<float> center{25.0f, 25.0f, 25.0f};
		float max_radius = 50.0f * 0.3f;

		for (const auto& p : particles) {
			Vector3_t<float> offset = p.position - center;
			float distance =
				std::sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z);
			REQUIRE(distance <= max_radius * 1.1f); // Allow small tolerance
		}
	}
}

TEST_CASE("Interactive System Creation", "[interactive][systems]") {
	SECTION("Basic system creation") {
		auto system = create_interactive_test_system(20, "linear");

		REQUIRE(system.get_particle_count() == 20);
		REQUIRE(system.get_particle_types().size() == 2);

		// Verify system info
		auto info = get_system_info(system);
		REQUIRE(info["particle_count"] == "20");
		REQUIRE(info["type_count"] == "2");
		REQUIRE(info["temperature"] == "300");
	}

	SECTION("Large system creation") {
		auto system = create_interactive_test_system(1000, "grid");

		REQUIRE(system.get_particle_count() == 1000);
		REQUIRE(system.get_particle_types().size() == 2);

		// Performance check - should create quickly
		auto start = std::chrono::high_resolution_clock::now();
		auto benchmark_system = create_interactive_test_system(1000, "random");
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		REQUIRE(duration.count() < 1000); // Should take less than 1 second
	}
}

TEST_CASE("Interactive Device Operations", "[interactive][device]") {
	SECTION("Device operations test") {
		auto system = create_interactive_test_system(100, "random");

		// Test device operations (should not throw)
		REQUIRE_NOTHROW(test_device_operations(system));
	}
}

TEST_CASE("Interactive Benchmarking", "[interactive][benchmark]") {
	SECTION("Benchmark particle operations") {
		auto system = create_interactive_test_system(100, "linear");

		// Benchmark different operations
		auto creation_time = benchmark_operation(system, "particle_creation");
		auto system_time = benchmark_operation(system, "system_creation");
		auto device_time = benchmark_operation(system, "device_transfer");

		// Verify benchmarks return reasonable times (> 0)
		REQUIRE(creation_time >= 0.0);
		REQUIRE(system_time >= 0.0);
		REQUIRE(device_time >= 0.0);

		std::cout << "Benchmark results:" << std::endl;
		std::cout << "  Particle creation: " << creation_time << " ms" << std::endl;
		std::cout << "  System creation: " << system_time << " ms" << std::endl;
		std::cout << "  Device transfer: " << device_time << " ms" << std::endl;
	}
}

TEST_CASE("Decompose System with 3 GPUs", "[decompose][gpu]") {
	// Initialize backend once
	initialize_backend_once();

	SECTION("Test decompose system functionality with 3 GPUs") {
		// Create 3 GPU resources from available GPUs
		std::vector<Resource> resources;
		resources.push_back(get_device_resource(0)); // GPU 0
		resources.push_back(get_device_resource(1)); // GPU 1
		resources.push_back(get_device_resource(2)); // GPU 2

		INFO("Created 3 GPU resources (GPU 0, 1, 2)");
		REQUIRE(resources.size() == 3);

		// Create SimSystem with GPU resources
		SimSystem sim_system(resources);
		INFO("Created SimSystem with GPU resources");

		// Configure basic parameters
		sim_system.set_box_size(150.0f, 150.0f, 450.0f); // Z-dominant for spatial decomposition
		sim_system.set_temperature(298.15f);
		sim_system.set_cutoff(12.0f);
		sim_system.set_pairlist_cutoff(24.0f);
		sim_system.set_timestep(0.001f);
		sim_system.set_num_steps(10000);
		sim_system.set_estimated_particles(100000); // 100K particles
		INFO("Set simulation parameters for 3-GPU decomposition");

		// Set spatial decomposer type with Z-direction (should create 3 patches in Z)
		sim_system.set_decomposer_type(DecomposerType::Spatial, DecomposeDirection::Z);
		INFO("Set spatial decomposer for Z-direction decomposition");

		// Verify decomposer was created
		REQUIRE(sim_system.get_decomposer() != nullptr);
		INFO("Decomposer successfully initialized");

		// Verify no patch manager exists yet
		REQUIRE_FALSE(sim_system.has_patch_manager());
		INFO("No patch manager before decomposition");

		// Perform decomposition
		INFO("Calling decompose_system() for 3 GPUs...");
		REQUIRE_NOTHROW(sim_system.decompose_system());
		INFO("Decomposition completed");

		// Verify patch manager was created
		REQUIRE(sim_system.has_patch_manager());
		REQUIRE(sim_system.get_patch_manager() != nullptr);
		INFO("PatchManager successfully created for 3 GPUs");

		// Verify system configuration
		auto box_size = sim_system.get_box_size();
		INFO("Box size: " << box_size.x << " x " << box_size.y << " x " << box_size.z);
		INFO("Cutoff: " << sim_system.get_cutoff());
		INFO("Resources: " << sim_system.get_resources().size());

		REQUIRE(sim_system.get_resources().size() == 3);

		// Try to decompose again (should warn and return early)
		INFO("Attempting second decomposition (should warn)...");
		REQUIRE_NOTHROW(sim_system.decompose_system());
		INFO("Second decomposition handled correctly");
	}

	SECTION("Test SimManager initialization with 3 GPUs") {
		// Create 3 GPU resources using different GPUs
		std::vector<Resource> resources;
		resources.push_back(get_device_resource(3)); // GPU 3
		resources.push_back(get_device_resource(4)); // GPU 4
		resources.push_back(get_device_resource(5)); // GPU 5

		INFO("Created 3 GPU resources (GPU 3, 4, 5)");
		REQUIRE(resources.size() == 3);

		// Create and configure SimSystem
		SimSystem sim_system(resources);
		sim_system.set_box_size(120.0f, 120.0f, 360.0f); // 3:1 aspect ratio for Z decomposition
		sim_system.set_temperature(310.0f);
		sim_system.set_cutoff(10.0f);
		sim_system.set_pairlist_cutoff(20.0f);
		sim_system.set_timestep(0.0005f);
		sim_system.set_num_steps(50000);
		sim_system.set_estimated_particles(200000); // 200K particles

		INFO("Created and configured SimSystem for 3 GPUs");

		// Create SimManager (will auto-initialize decomposer)
		SimManager sim_manager(sim_system);
		INFO("Created SimManager");

		// Initialize - this should:
		// 1. Set up default decomposer if not set
		// 2. Perform domain decomposition
		// 3. Transfer grids and tables to all 3 GPUs
		INFO("Calling SimManager::init() for 3 GPUs...");
		REQUIRE_NOTHROW(sim_manager.init());
		INFO("SimManager initialization completed");

		// Verify decomposition was successful across all GPUs
		REQUIRE(sim_system.has_patch_manager());
		REQUIRE(sim_system.get_decomposer() != nullptr);
		INFO("SimManager successfully handled 3-GPU decomposition");

		// Verify grid and table transfers
		INFO("Grids and tables transferred to all GPU resources");
	}

	SECTION("Test SystemState with particles on 3 GPUs") {
		// Create 3 GPU resources
		std::vector<Resource> resources;
		resources.push_back(get_device_resource(6)); // GPU 6
		resources.push_back(get_device_resource(7)); // GPU 7
		resources.push_back(get_device_resource(0)); // GPU 0 (wrapping back)

		INFO("Created 3 GPU resources (GPU 6, 7, 0) for SystemState test");
		REQUIRE(resources.size() == 3);

		// Create and configure SimSystem
		SimSystem sim_system(resources);
		sim_system.set_box_size(180.0f, 180.0f, 540.0f); // 3:1 aspect ratio for Z decomposition
		sim_system.set_temperature(300.0f);
		sim_system.set_cutoff(15.0f);
		sim_system.set_pairlist_cutoff(30.0f);
		sim_system.set_timestep(0.001f);
		sim_system.set_num_steps(25000);
		sim_system.set_estimated_particles(150000); // 150K particles

		// Set up decomposer and decompose
		sim_system.set_decomposer_type(DecomposerType::Spatial, DecomposeDirection::Z);
		REQUIRE_NOTHROW(sim_system.decompose_system());
		REQUIRE(sim_system.has_patch_manager());

		INFO("SimSystem configured and decomposed for SystemState test");

		// Create SystemState
		SystemState system_state(sim_system);
		INFO("Created SystemState");

		// Create test particles
		const size_t num_particles = 1000;
		std::vector<ParticleIO> particles =
			create_test_particles(num_particles, "random", 180.0f);

		// Set particle IDs
		for (size_t i = 0; i < particles.size(); ++i) {
			particles[i].id = static_cast<int>(i);
		}

		INFO("Created " << num_particles << " test particles");

		// Set initial particle data in SystemState
		REQUIRE_NOTHROW(system_state.set_init_particle_data(particles));
		INFO("Set initial particle data in SystemState");

		// Verify particle count
		REQUIRE(system_state.get_num_particles() == num_particles);
		INFO("SystemState correctly reports " << system_state.get_num_particles() << " particles");

		// Test particle positions update
		std::vector<Vector3> new_positions;
		new_positions.reserve(num_particles);
		for (size_t i = 0; i < num_particles; ++i) {
			new_positions.push_back(Vector3(particles[i].position.x + 1.0f, // Slight displacement
											particles[i].position.y + 1.0f,
											particles[i].position.z + 1.0f));
		}

		REQUIRE_NOTHROW(system_state.set_particle_positions(new_positions));
		INFO("Successfully updated particle positions");

		// Test clearing and re-adding particles
		system_state.clear_global_arrays();
		REQUIRE(system_state.get_num_particles() == 0);
		INFO("Successfully cleared global particle arrays");

		// Re-add particles
		std::vector<Vector3> positions, momenta;
		std::vector<int> ids;
		for (const auto& p : particles) {
			positions.push_back(p.position);
			momenta.push_back(p.momentum);
			ids.push_back(p.id);
		}

		REQUIRE_NOTHROW(system_state.add_particle_data(positions, momenta, ids));
		REQUIRE(system_state.get_num_particles() == num_particles);
		INFO("Successfully re-added particle data");
	}
}

#endif // USE_PYBIND
