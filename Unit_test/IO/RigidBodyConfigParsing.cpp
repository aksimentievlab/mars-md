/**
 * @file RigidBodyConfigParsing.cpp
 * @brief Phase 5 test for ConfigParser's "rigidBody" block parsing
 * (IO/ConfigParser.cpp): pure parsing, no SimManager/GPU kernels involved -
 * writes a small temp .bd config (mirroring the "particle" block's style)
 * plus a temp .dx grid file, parses it, and checks the resulting
 * RigidBodyType/RigidBodyIO fields.
 */

#include "../catch_boiler.h"
#include "IO/ConfigParser.h"
#include "IO/DxIO.h"
#include "System/SimSystem.h"
#include "Types/BaseGrid.h"
#include <filesystem>
#include <fstream>

using namespace ARBD;

TEST_CASE("ConfigParser parses rigidBody blocks into RigidBodyType + RigidBodyIO",
		  "[configparser][rigidbody]") {
	initialize_backend_once();
	std::vector<Resource> resources = {Resource(Global::single_resource_id)};

	// A trivial grid - only used to exercise add_dense_grid()/grid_id
	// resolution here, not sampled.
	const auto tmp_dir = std::filesystem::temp_directory_path();
	const std::string grid_path = (tmp_dir / "arbd_test_rb_config_grid.dx").string();
	BaseGrid<float> grid(Matrix3(1.0f), Vector3(0.0f), 4, 4, 4);
	grid.zero();
	DXReader::write_grid(grid, grid_path);

	const std::string config_path = (tmp_dir / "arbd_test_rb_config.bd").string();
	{
		std::ofstream out(config_path);
		out << "rigidBody ProteinA\n";
		out << "mass 10.0\n";
		out << "inertia 1.0 2.0 3.0\n";
		out << "transDamping 0.1 0.1 0.1\n";
		out << "rotDamping 0.2 0.2 0.2\n";
		out << "densityGrid " << grid_path << "\n";
		out << "densityGridScale 2.0\n";
		out << "position 1.0 2.0 3.0\n";
		out << "orientation 1 0 0 0 1 0 0 0 1\n";
		out << "momentum 0.5 0.0 0.0\n";
		out << "angularMomentum 0.0 0.1 0.0\n";
		out << "rigidBody MembraneB\n";
		out << "mass 20.0\n";
		out << "inertia 4.0 5.0 6.0\n";
		out << "potentialGrid " << grid_path << "\n";
	}

	SimSystem sys(resources);
	ConfigParser parser(sys, config_path);

	std::filesystem::remove(config_path);
	std::filesystem::remove(grid_path);

	const auto& types = sys.get_rigid_body_types();
	REQUIRE(types.size() == 2);

	const RigidBodyType& protein = types[0];
	REQUIRE(protein.name == "ProteinA");
	REQUIRE(protein.mass == Catch::Approx(10.0f));
	REQUIRE(protein.inertia.x == Catch::Approx(1.0f));
	REQUIRE(protein.inertia.y == Catch::Approx(2.0f));
	REQUIRE(protein.inertia.z == Catch::Approx(3.0f));
	REQUIRE(protein.trans_damping.x == Catch::Approx(0.1f));
	REQUIRE(protein.rot_damping.x == Catch::Approx(0.2f));
	REQUIRE(protein.density_grids.size() == 1);
	REQUIRE(protein.density_grids[0].grid_id >= 0);
	REQUIRE(protein.density_grids[0].scale == Catch::Approx(2.0f));
	REQUIRE(protein.density_grid_keys.size() == 1);
	REQUIRE(protein.density_grid_keys[0] == grid_path);

	const RigidBodyType& membrane = types[1];
	REQUIRE(membrane.name == "MembraneB");
	REQUIRE(membrane.mass == Catch::Approx(20.0f));
	REQUIRE(membrane.potential_grids.size() == 1);
	REQUIRE(membrane.potential_grid_keys[0] == grid_path);
	// Same grid file for both types -> same grid_id, matching the "grid
	// filename doubles as the pairing key" convention (see ConfigParser.cpp).
	REQUIRE(membrane.potential_grids[0].grid_id == protein.density_grids[0].grid_id);

	const auto& instances = parser.get_init_rigid_bodies();
	REQUIRE(instances.size() == 2);

	const RigidBodyIO& protein_io = instances[0];
	REQUIRE(protein_io.type_id == 0);
	REQUIRE(protein_io.position.x == Catch::Approx(1.0f));
	REQUIRE(protein_io.position.y == Catch::Approx(2.0f));
	REQUIRE(protein_io.position.z == Catch::Approx(3.0f));
	REQUIRE(protein_io.momentum.x == Catch::Approx(0.5f));
	REQUIRE(protein_io.angular_momentum.y == Catch::Approx(0.1f));
	REQUIRE(protein_io.orientation.ex() == Vector3(1.0f, 0.0f, 0.0f));
	REQUIRE(protein_io.orientation.ey() == Vector3(0.0f, 1.0f, 0.0f));
	REQUIRE(protein_io.orientation.ez() == Vector3(0.0f, 0.0f, 1.0f));

	const RigidBodyIO& membrane_io = instances[1];
	REQUIRE(membrane_io.type_id == 1);
	// No position/orientation given for MembraneB - must default to origin/identity.
	REQUIRE(membrane_io.position == Vector3(0.0f, 0.0f, 0.0f));
	REQUIRE(membrane_io.orientation.ex() == Vector3(1.0f, 0.0f, 0.0f));
}
