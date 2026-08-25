#include "../catch_boiler.h"
#include "Backend/Resource.h"
#include "Objects/DeviceRigidBodyManager.h"
#include "Objects/RigidBodyProperties.h"
using namespace MARS;

namespace {

HostRigidBodyData create_test_rigid_bodies(int count) {
	HostRigidBodyData host;
	host.reserve(count);
	for (int i = 0; i < count; ++i) {
		RigidBodyIO rb{};
		rb.id = i;
		rb.type_id = i % 2;
		rb.position = Vector3(float(i), 2.0f * i, 3.0f * i);
		rb.orientation = Matrix3(Vector3(0.0f, 1.0f, 0.0f),
								 Vector3(-1.0f, 0.0f, 0.0f),
								 Vector3(0.0f, 0.0f, 1.0f));
		rb.momentum = Vector3(0.1f * i, 0.0f, 0.0f);
		rb.angular_momentum = Vector3(0.0f, 0.2f * i, 0.0f);
		rb.force = Vector3(1.0f, 2.0f, 3.0f);
		rb.torque = Vector3(4.0f, 5.0f, 6.0f);
		host.push_back(rb);
	}
	return host;
}

std::vector<RigidBodyType> create_test_rigid_body_types() {
	std::vector<RigidBodyType> types(2);

	types[0].name = "A";
	types[0].id = 0;
	types[0].mass = 10.0f;
	types[0].inertia = Vector3(1.0f, 2.0f, 3.0f);
	types[0].potential_grids = {GridTerm{0, 1.0f}, GridTerm{1, 2.0f}};
	types[0].density_grids = {GridTerm{2, 3.0f}};

	types[1].name = "B";
	types[1].id = 1;
	types[1].mass = 20.0f;
	types[1].inertia = Vector3(4.0f, 5.0f, 6.0f);
	// Default-scale GridTerms (scale == 1.0), unlike type 0's explicitly-scaled ones.
	types[1].density_grids = {GridTerm{3}, GridTerm{4}};
	types[1].pmf_grids = {GridTerm{5, 0.5f}};

	return types;
}

} // namespace

TEST_CASE("Device Rigid Body Copy From Host To Device", "[device][rigidbody]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	constexpr idx_t count = 8;
	DeviceRigidBody bodies(16, res);
	HostRigidBodyData host = create_test_rigid_bodies(count);

	bodies.copy_from_host(host, count);
	REQUIRE(bodies.size() == count);

	HostRigidBodyData result;
	bodies.copy_to_host(result, count);

	REQUIRE(result.global_id[3] == host.global_id[3]);
	REQUIRE(result.type_id[3] == host.type_id[3]);
	REQUIRE(result.position[3] == host.position[3]);
	REQUIRE(result.momentum[3] == host.momentum[3]);
	REQUIRE(result.angular_momentum[3] == host.angular_momentum[3]);
	REQUIRE(result.force[3] == host.force[3]);
	REQUIRE(result.torque[3] == host.torque[3]);
	REQUIRE(result.orientation[3].ex() == host.orientation[3].ex());
	REQUIRE(result.orientation[3].ey() == host.orientation[3].ey());
	REQUIRE(result.orientation[3].ez() == host.orientation[3].ez());

	bodies.clear_forces();
	bodies.copy_to_host(result, count);
	REQUIRE(result.force[3] == Vector3(0.0f, 0.0f, 0.0f));
	REQUIRE(result.torque[3] == Vector3(0.0f, 0.0f, 0.0f));
}

TEST_CASE("Device Rigid Body Types flat grid-term buffer", "[device][rigidbody]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	std::vector<RigidBodyType> types = create_test_rigid_body_types();
	DeviceRigidBodyTypes device_types(types, res);

	REQUIRE(device_types.size() == 2);

	std::vector<float> mass(2);
	device_types.mass().copy_to_host(mass.data(), 2);
	REQUIRE(mass[0] == 10.0f);
	REQUIRE(mass[1] == 20.0f);

	// Layout is [type0: potential, density, pmf][type1: potential, density,
	// pmf] contiguously - type0 contributes 3 entries, type1 contributes 3.
	const size_t total = 6;
	std::vector<GridTerm> grid_terms(total);
	device_types.grid_terms().copy_to_host(grid_terms.data(), total);

	std::vector<int> grid_ids;
	std::vector<float> grid_scales;
	for (const GridTerm& term : grid_terms) {
		grid_ids.push_back(term.grid_id);
		grid_scales.push_back(term.scale);
	}

	REQUIRE(grid_ids == std::vector<int>{0, 1, 2, 3, 4, 5});
	// type1's density grids used the default-scale GridTerm, i.e. 1.0.
	REQUIRE(grid_scales == std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 0.5f});
}
