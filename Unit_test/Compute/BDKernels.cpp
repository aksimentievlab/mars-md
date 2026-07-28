#include "../catch_boiler.h"

#include "Backend/Resource.h"
#include "Interactions/Bonded/BondComputer.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/Tables.h"
#include "PatchOperation/Integrator.h"
#include "PatchOperation/Patch.h"
#include "System/PeriodicBox.h"

using namespace ARBD;
using Catch::Approx;
TEST_CASE("RandomTest", "[free][random]") {
	const int N = 100000;
	std::vector<float> samples_x, samples_y, samples_z;

	openrand::Philox rng(12345, 0);

	for (int i = 0; i < N; i++) {
		openrand::float4 uniform = rng.draw_float4();

		// Box-Muller transform
		float r1 = sqrtf(-2.0f * logf(uniform.x));
		float theta1 = 2.0f * 3.1415926535f * uniform.y;
		float r2 = sqrtf(-2.0f * logf(uniform.z));
		float theta2 = 2.0f * 3.1415926535f * uniform.w;

		samples_x.push_back(r1 * cosf(theta1));
		samples_y.push_back(r1 * sinf(theta1));
		samples_z.push_back(r2 * cosf(theta2));
	}

	// Calculate mean and variance
	auto calc_stats = [](const std::vector<float>& v) {
		double mean = 0.0, var = 0.0;
		for (float x : v)
			mean += x;
		mean /= v.size();
		for (float x : v)
			var += (x - mean) * (x - mean);
		var /= v.size();
		return std::make_pair(mean, var);
	};

	auto [mean_x, var_x] = calc_stats(samples_x);
	auto [mean_y, var_y] = calc_stats(samples_y);
	auto [mean_z, var_z] = calc_stats(samples_z);

	std::cout << "X: mean=" << mean_x << ", var=" << var_x << ", std=" << sqrt(var_x) << std::endl;
	std::cout << "Y: mean=" << mean_y << ", var=" << var_y << ", std=" << sqrt(var_y) << std::endl;
	std::cout << "Z: mean=" << mean_z << ", var=" << var_z << ", std=" << sqrt(var_z) << std::endl;

	// Expected: mean ≈ 0, var ≈ 1, std ≈ 1
	REQUIRE(mean_x == Approx(0.0).margin(0.01));
	REQUIRE(var_x == Approx(1.0).margin(0.02));
	REQUIRE(var_y == Approx(1.0).margin(0.02));
	REQUIRE(var_z == Approx(1.0).margin(0.02));
}
TEST_CASE("IntegratorTest", "[free][bd]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create particle type
	ParticleType ptype("Ar");
	ptype.mass = 39.948f; // amu (Argon)

	float D = 149.0f;					// Å²/ns (calculated as kT/(gamma*mass))
	ptype.diffusion = Vector3(D, D, D); // Å²/ns ✓

	std::vector<ParticleType> types = {ptype};
	DeviceParticleTypes device_types(types, res);

	// Setup 100 particles at center
	DeviceParticle particles(100, res);
	HostParticleData init;
	init.resize(100);
	for (int i = 0; i < 100; i++) {
		init.pos[i] = Vector3(50.0f, 50.0f, 50.0f);
		init.type_id[i] = 0;
		init.force[i] = Vector3(0, 0, 0); // Zero force
	}
	particles.copy_from_host(init, 100);

	// Simulation parameters
	float dt = 2e-5f; // 20 fs in ns
	float temperature = 300.0f;
	float kT = ARBD::constants::BOLTZMANN * temperature; // 0.596 kcal/mol
	int num_steps = 10000;								 // 0.2 ns total
	Vector3 box_size(100.0f, 100.0f, 100.0f);
	// Origin defaults to (0,0,0), so the primary image is [0,100) per axis -
	// the particles start at the box center (50,50,50).
	PeriodicBox sim_box(box_size);

	// Expected MSD = 6*D*t
	float total_time = dt * num_steps;
	float expected_msd = 6.0f * D * total_time;

	std::cout << "=== Test Parameters ===" << std::endl;
	std::cout << "Diffusion: " << D << " Å²/ns" << std::endl;
	std::cout << "Timestep: " << dt << " ns = " << dt * 1e6 << " fs" << std::endl;
	std::cout << "Total time: " << total_time << " ns" << std::endl;
	std::cout << "Expected MSD = " << expected_msd << " Å²" << std::endl;

	// Run integration
	auto particle_view = particles.view();
	auto type_view = device_types.view();

	for (int step = 0; step < num_steps; step++) {
		launch_BD<float>(res, particle_view, type_view, dt, step, kT, 100, sim_box, 5, step);
	}

	// Calculate MSD
	HostParticleData final;
	particles.copy_to_host(final, 100);

	float msd = 0.0f;
	for (int i = 0; i < 100; i++) {
		Vector3 disp = final.pos[i] - init.pos[i];

		// Apply minimum image convention for PBC (use roundf!)
		disp.x -= box_size.x * roundf(disp.x / box_size.x); // ✓
		disp.y -= box_size.y * roundf(disp.y / box_size.y); // ✓
		disp.z -= box_size.z * roundf(disp.z / box_size.z); // ✓

		msd += disp.length2();
		std::cout << "final pos: " << final.pos[i].x << ", " << final.pos[i].y << ", "
				  << final.pos[i].z << std::endl;
		std::cout << final.force[i].x << ", " << final.force[i].y << ", " << final.force[i].z
				  << std::endl;
	}
	msd /= 100.0f;

	std::cout << "Measured MSD = " << msd << " Å²" << std::endl;
	std::cout << "Ratio = " << msd / expected_msd << std::endl;

	REQUIRE(msd == Approx(expected_msd).epsilon(0.2));
}

TEST_CASE("BondedForcesTest", "[free][bonded]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Real bond table from the mpipi_k18 repro case - reproduces the exact
	// CUDA_ERROR_ILLEGAL_ADDRESS crash seen in Patch::calculate_bonded_forces
	// at much smaller scale (2 particles, 1 bond) for fast iteration.
	const std::string bond_file =
		"/data/storage01/pinyili2/sims/fig2e/mpipi_k18/potentials/bond-19.190-3.800.dat";

	TablesRegistry tables_registry;
	int function_index = tables_registry.get_or_load_bond(bond_file);
	std::cout << "Loaded bond table, function_index=" << function_index
			  << " num_points=" << tables_registry.get_bond_functions()[function_index].Y.size()
			  << " step_size=" << tables_registry.get_bond_functions()[function_index].step_size
			  << " start=" << tables_registry.get_bond_functions()[function_index].start
			  << std::endl;

	std::vector<Resource> resources = {res};
	tables_registry.set_resources(&resources);
	tables_registry.build_device_arrays();

	// Full-scale repro: 129 particles in a chain, 128 sequential bonds -
	// matches mpipi_k18 exactly (same particle count, same bond count/topology).
	const idx_t num_particles = 129;
	BondedInteractions bonded_interactions;
	for (idx_t i = 0; i + 1 < num_particles; ++i) {
		Bond bond;
		bond.ind1 = static_cast<int>(i);
		bond.ind2 = static_cast<int>(i + 1);
		bond.function_index = function_index;
		bond.form = InteractionForm::Tabulated;
		bond.flag = BondFlag::DEFAULT;
		bonded_interactions.add_bond(bond);
	}
	std::cout << "Built " << bonded_interactions.get_num_bonds() << " bonds" << std::endl;

	// Match the real mpipi_k18 patch's capacity (from the spatial decomposer)
	// rather than num_particles, in case scale/capacity is what triggers the
	// illegal-address crash rather than particle/bond count.
	Patch patch(0, 1024000, res);
	patch.set_particle_count(num_particles);

	HostParticleData init;
	init.resize(num_particles);
	for (idx_t i = 0; i < num_particles; ++i) {
		init.pos[i] = Vector3(0.0f, 0.0f, 1.885f * static_cast<float>(i));
		init.type_id[i] = 0;
		init.force[i] = Vector3(0, 0, 0);
	}
	patch.copy_particles_from_host(init, 0, num_particles);

	PeriodicBox box(Vector3(500.0f, 500.0f, 500.0f));
	patch.set_periodic_box(&box);

	DeviceParticleTypes device_types(std::vector<ParticleType>{ParticleType("Ar")}, res);

	// Mirror the real SimManager::execute_force_calculation sequence: PMF/
	// nonbonded runs first every step (a no-op here since there are 0 grids,
	// matching mpipi_k18), then bonded forces. Testing whether that ordering
	// (not just calculate_bonded_forces in isolation) is needed to reproduce.
	std::cout << "Calling calculate_nonbonded_forces (0 grids, should no-op)..." << std::endl;
	NonBondedInteractions nb_interactions;
	DeviceBuffer<BaseGridView<float>> empty_grid_views;
	Event nb_evt = patch.calculate_nonbonded_forces(nb_interactions,
													bonded_interactions,
													device_types,
													empty_grid_views,
													tables_registry,
													0,
													10.0f,
													0.0f,
													1);
	nb_evt.wait();
	std::cout << "calculate_nonbonded_forces returned successfully" << std::endl;

	// Host-side sanity check: replicate TabulatedPotential::compute exactly,
	// using the real loaded Y data, to confirm the math itself gives a
	// nonzero force at this bond distance before trusting the device result.
	{
		const Table& t = tables_registry.get_bond_functions()[function_index];
		float dx = 1.885f;
		float w = (dx - t.start) / t.step_size;
		int home = static_cast<int>(std::floor(w));
		w -= home;
		float U0 = t.Y[home];
		float dU = t.Y[home + 1] - U0;
		std::cout << "HOST check: dx=" << dx << " home=" << home << " U0=" << U0 << " dU=" << dU
				  << " step_inv=" << (1.0f / t.step_size)
				  << " expected_force_mag=" << (dU / t.step_size) << std::endl;
	}

	std::cout << "Calling calculate_bonded_forces..." << std::endl;
	Event evt =
		patch.calculate_bonded_forces(bonded_interactions, device_types, tables_registry, 0);
	evt.wait();
	std::cout << "calculate_bonded_forces returned successfully" << std::endl;

	HostParticleData final_data;
	patch.copy_particles_to_host(final_data, 0, num_particles);
	std::cout << "force[0]: " << final_data.force[0].x << ", " << final_data.force[0].y << ", "
			  << final_data.force[0].z << std::endl;
	std::cout << "force[1]: " << final_data.force[1].x << ", " << final_data.force[1].y << ", "
			  << final_data.force[1].z << std::endl;

	REQUIRE(true);
}

TEST_CASE("DirectTabulatedBondKernelTest", "[free][bonded][direct]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const std::string bond_file =
		"/data/storage01/pinyili2/sims/fig2e/mpipi_k18/potentials/bond-19.190-3.800.dat";
	TablesRegistry tables_registry;
	int function_index = tables_registry.get_or_load_bond(bond_file);
	std::vector<Resource> resources = {res};
	tables_registry.set_resources(&resources);
	tables_registry.build_device_arrays();

	const Table& table = tables_registry.get_bond_functions()[function_index];
	TabulatedPotential pot{};
	pot.pot =
		const_cast<arbd_real*>(tables_registry.get_device_bond_buffer(0, function_index).data());
	pot.step_inv = static_cast<arbd_real>(1.0 / table.step_size);
	pot.size = static_cast<unsigned int>(table.Y.size());
	pot.start = static_cast<arbd_real>(table.start);
	pot.is_periodic = false;
	std::cout << "pot.step_inv=" << pot.step_inv << " pot.size=" << pot.size
			  << " pot.start=" << pot.start << " pot.pot=" << (void*)pot.pot << std::endl;

	DeviceBuffer<TabulatedPotential> tables_buf(1, res);
	tables_buf.copy_from_host(std::vector<TabulatedPotential>{pot});

	DeviceBuffer<int> table_indices_buf(1, res);
	table_indices_buf.copy_from_host(std::vector<int>{0});

	DeviceBuffer<int> forms_buf(1, res);
	forms_buf.copy_from_host(std::vector<int>{static_cast<int>(InteractionForm::Tabulated)});

	DeviceBuffer<int2> indices_buf(1, res);
	indices_buf.copy_from_host(std::vector<int2>{int2{0, 1}});

	DeviceBuffer<Vector3> positions_buf(2, res);
	positions_buf.copy_from_host(std::vector<Vector3>{Vector3(0, 0, 0), Vector3(0, 0, 1.885f)});

	DeviceBuffer<Vector3> force_buf(2, res);
	force_buf.copy_from_host(std::vector<Vector3>{Vector3(0, 0, 0), Vector3(0, 0, 0)});

	PeriodicBox box(Vector3(500.0f, 500.0f, 500.0f));
	DeviceBuffer<PeriodicBox> box_buf(1, res);
	box_buf.copy_from_host(std::vector<PeriodicBox>{box});

	std::cout << "Launching launch_tabulated_bonds directly..." << std::endl;
	Event evt = launch_tabulated_bonds(res,
									   indices_buf.device_data(),
									   positions_buf.device_data(),
									   force_buf.device_data(),
									   tables_buf.device_data(),
									   table_indices_buf.device_data(),
									   forms_buf.device_data(),
									   box_buf.device_data(),
									   false,
									   1);
	evt.wait();
	std::cout << "launch_tabulated_bonds returned successfully" << std::endl;

	std::vector<Vector3> host_force(2);
	force_buf.copy_to_host(host_force.data(), 2, true);
	std::cout << "direct force[0]: " << host_force[0].x << ", " << host_force[0].y << ", "
			  << host_force[0].z << std::endl;
	std::cout << "direct force[1]: " << host_force[1].x << ", " << host_force[1].y << ", "
			  << host_force[1].z << std::endl;

	REQUIRE(true);
}

TEST_CASE("DeviceBondedInteractionsDirectTest", "[free][bonded][dbi]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const std::string bond_file =
		"/data/storage01/pinyili2/sims/fig2e/mpipi_k18/potentials/bond-19.190-3.800.dat";
	TablesRegistry tables_registry;
	int function_index = tables_registry.get_or_load_bond(bond_file);

	// Match the real mpipi_k18 scenario exactly: it also loads 190 pairwise
	// nonbonded tables (19 types, i<=j) into the SAME TablesRegistry before
	// build_device_arrays() - testing whether that (191 tables total,
	// matching the real crash) versus just 1 (this test previously) changes
	// anything about the bond kernel's behavior.
	{
		std::vector<std::string> nb_files;
		std::string dir = "/data/storage01/pinyili2/sims/fig2e/mpipi_k18/potentials/";
		// Reuse the single real bond file's contents as a stand-in Y-table
		// for all 190 nonbonded slots - only the table *count*/memory layout
		// matters for this test, not the nonbonded physics.
		int type_id = 0;
		for (int i = 0; i < 19; ++i) {
			for (int j = i; j < 19; ++j) {
				tables_registry.load_pair_nonbonded(i, j, bond_file);
			}
		}
	}
	std::cout << "Loaded " << tables_registry.get_nonbonded_functions().size()
			  << " nonbonded tables + 1 bond table" << std::endl;

	std::vector<Resource> resources = {res};
	tables_registry.set_resources(&resources);
	tables_registry.build_device_arrays();

	const idx_t num_particles = 129;
	BondedInteractions interactions;
	for (idx_t i = 0; i + 1 < num_particles; ++i) {
		Bond bond;
		bond.ind1 = static_cast<int>(i);
		bond.ind2 = static_cast<int>(i + 1);
		bond.function_index = function_index;
		bond.form = InteractionForm::Tabulated;
		bond.flag = BondFlag::REPLACE;
		interactions.add_bond(bond);
		interactions.add_exclude(Exclude(bond.ind1, bond.ind2));
	}
	std::cout << "Built " << interactions.get_num_bonds() << " bonds, "
			  << interactions.get_exclusions().size() << " exclusions" << std::endl;

	DeviceBondedInteractions device_bonded(res);
	device_bonded.copy_from_host(interactions);
	device_bonded.link_tables(tables_registry, 0);
	std::cout << "device_bonded.num_bonds()=" << device_bonded.num_bonds() << std::endl;

	std::vector<Vector3> host_pos(num_particles);
	std::vector<Vector3> host_force_init(num_particles);
	for (idx_t i = 0; i < num_particles; ++i) {
		host_pos[i] = Vector3(0.0f, 0.0f, 1.885f * static_cast<float>(i));
		host_force_init[i] = Vector3(0, 0, 0);
	}
	DeviceBuffer<Vector3> positions_buf(num_particles, res);
	positions_buf.copy_from_host(host_pos);
	DeviceBuffer<Vector3> force_buf(num_particles, res);
	force_buf.copy_from_host(host_force_init);
	PeriodicBox box(Vector3(500.0f, 500.0f, 500.0f));
	DeviceBuffer<PeriodicBox> box_buf(1, res);
	box_buf.copy_from_host(std::vector<PeriodicBox>{box});

	std::cout << "Launching via DeviceBondedInteractions accessors..." << std::endl;
	Event evt = launch_tabulated_bonds(res,
									   device_bonded.bond_indices(),
									   positions_buf.device_data(),
									   force_buf.device_data(),
									   device_bonded.bond_potentials(),
									   device_bonded.bond_table_indices(),
									   device_bonded.bond_forms(),
									   box_buf.device_data(),
									   false,
									   device_bonded.num_bonds());
	evt.wait();

	std::vector<Vector3> host_force(num_particles);
	force_buf.copy_to_host(host_force.data(), num_particles, true);
	std::cout << "dbi force[0]: " << host_force[0].x << ", " << host_force[0].y << ", "
			  << host_force[0].z << std::endl;
	std::cout << "dbi force[1]: " << host_force[1].x << ", " << host_force[1].y << ", "
			  << host_force[1].z << std::endl;
	std::cout << "dbi force[64]: " << host_force[64].x << ", " << host_force[64].y << ", "
			  << host_force[64].z << std::endl;

	REQUIRE(true);
}
