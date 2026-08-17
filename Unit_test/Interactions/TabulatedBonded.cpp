/**
 * @file TabulatedBonded.cpp
 * @brief Tests for TabulatedAngleComputer and TabulatedDihedralComputer.
 * @see TabulatedBonded.md
 */

#include "../catch_boiler.h"
#include "Interactions/DeviceBondedInteraction.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

#include <array>
#include <cmath>
#include <vector>

using Catch::Approx;
using namespace ARBD;

namespace {

constexpr size_t TABLE_N = 4096;
constexpr double SLOPE = 2.5;
constexpr double FD_STEP = 1e-4;

/// @brief Tabulated potential U(x) = slope * x over [lo, hi).
struct RampTable {
	std::vector<arbd_real> pot;
	arbd_real start;
	arbd_real step;
	bool periodic;

	RampTable(double lo, double hi, size_t n, double slope, bool is_periodic)
		: pot(n), start(static_cast<arbd_real>(lo)),
		  step(static_cast<arbd_real>((hi - lo) / static_cast<double>(n))), periodic(is_periodic) {
		for (size_t i = 0; i < n; ++i) {
			pot[i] = static_cast<arbd_real>(
				slope * (lo + static_cast<double>(i) * (hi - lo) / static_cast<double>(n)));
		}
	}

	TabulatedPotential descriptor() const {
		TabulatedPotential t{};
		t.pot = const_cast<arbd_real*>(pot.data());
		t.step_inv = arbd_real(1) / step;
		t.size = static_cast<unsigned int>(pot.size());
		t.start = start;
		t.is_periodic = periodic;
		return t;
	}
};

/// @brief Interior angle at p[1], from the dot product.
double reference_angle(const std::array<Vector3, 3>& p) {
	const double ux = p[0].x - p[1].x, uy = p[0].y - p[1].y, uz = p[0].z - p[1].z;
	const double vx = p[2].x - p[1].x, vy = p[2].y - p[1].y, vz = p[2].z - p[1].z;
	const double nu = std::sqrt(ux * ux + uy * uy + uz * uz);
	const double nv = std::sqrt(vx * vx + vy * vy + vz * vz);
	double c = (ux * vx + uy * vy + uz * vz) / (nu * nv);
	c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
	return std::acos(c);
}

/// @brief IUPAC dihedral, atan2((n1 x n2).b2hat, n1.n2).
double reference_dihedral(const std::array<Vector3, 4>& p) {
	const double b1[3] = {p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z};
	const double b2[3] = {p[2].x - p[1].x, p[2].y - p[1].y, p[2].z - p[1].z};
	const double b3[3] = {p[3].x - p[2].x, p[3].y - p[2].y, p[3].z - p[2].z};

	auto cross = [](const double a[3], const double b[3], double out[3]) {
		out[0] = a[1] * b[2] - a[2] * b[1];
		out[1] = a[2] * b[0] - a[0] * b[2];
		out[2] = a[0] * b[1] - a[1] * b[0];
	};
	auto dot = [](const double a[3], const double b[3]) {
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	};

	double n1[3], n2[3], m[3];
	cross(b1, b2, n1);
	cross(b2, b3, n2);
	cross(n1, n2, m);

	return std::atan2(dot(m, b2) / std::sqrt(dot(b2, b2)), dot(n1, n2));
}

/// @brief -grad of SLOPE * coord(p), by central differences.
template<size_t N, typename CoordFn>
std::array<Vector3, N> reference_forces(std::array<Vector3, N> p, CoordFn coord) {
	std::array<Vector3, N> f{};
	for (size_t i = 0; i < N; ++i) {
		for (int axis = 0; axis < 3; ++axis) {
			float& c = axis == 0 ? p[i].x : (axis == 1 ? p[i].y : p[i].z);
			const float saved = c;

			c = static_cast<float>(saved + FD_STEP);
			const double plus = coord(p);
			c = static_cast<float>(saved - FD_STEP);
			const double minus = coord(p);
			c = saved;

			float& out = axis == 0 ? f[i].x : (axis == 1 ? f[i].y : f[i].z);
			out = static_cast<float>(-SLOPE * (plus - minus) / (2.0 * FD_STEP));
		}
	}
	return f;
}

template<size_t N>
Vector3 sum_force(const std::array<Vector3, N>& f) {
	Vector3 s(0.0f, 0.0f, 0.0f);
	for (const auto& v : f)
		s = s + v;
	return s;
}

template<size_t N>
Vector3 sum_torque(const std::array<Vector3, N>& p, const std::array<Vector3, N>& f) {
	Vector3 s(0.0f, 0.0f, 0.0f);
	for (size_t i = 0; i < N; ++i)
		s = s + p[i].cross(f[i]);
	return s;
}

std::array<Vector3, 3> generic_angle_config() {
	return {Vector3(1.3f, 0.4f, -0.7f), Vector3(0.0f, 0.0f, 0.0f), Vector3(-0.6f, 1.1f, 0.9f)};
}

std::array<Vector3, 4> generic_dihedral_config() {
	return {Vector3(1.2f, 1.0f, -0.3f),
			Vector3(1.1f, 0.0f, 0.0f),
			Vector3(0.0f, 0.0f, 0.0f),
			Vector3(-0.4f, 0.7f, 0.8f)};
}

/// @brief Runs TabulatedAngleComputer once, host-side, over a single angle.
std::array<Vector3, 3> run_angle(const std::array<Vector3, 3>& p,
								 const RampTable& table,
								 bool get_energy,
								 InteractionForm form = InteractionForm::Tabulated) {
	std::array<Vector3, 3> positions = p;
	std::array<Vector3, 3> force_energy{};
	const TabulatedPotential desc = table.descriptor();
	const int table_index = 0;
	const int form_value = static_cast<int>(form);
	const PeriodicBox box;
	const ARBD::int3 indices{0, 1, 2};

	TabulatedAngleComputer computer(&indices,
									positions.data(),
									force_energy.data(),
									&desc,
									&table_index,
									&form_value,
									&box,
									get_energy,
									1);
	computer(0);
	return force_energy;
}

/// @brief Runs TabulatedDihedralComputer once, host-side, over a single dihedral.
std::array<Vector3, 4> run_dihedral(const std::array<Vector3, 4>& p,
									const RampTable& table,
									bool get_energy,
									InteractionForm form = InteractionForm::Tabulated) {
	std::array<Vector3, 4> positions = p;
	std::array<Vector3, 4> force_energy{};
	const TabulatedPotential desc = table.descriptor();
	const int table_index = 0;
	const int form_value = static_cast<int>(form);
	const PeriodicBox box;
	const ARBD::int4 indices{0, 1, 2, 3};

	TabulatedDihedralComputer computer(&indices,
									   positions.data(),
									   force_energy.data(),
									   &desc,
									   &table_index,
									   &form_value,
									   &box,
									   get_energy,
									   1);
	computer(0);
	return force_energy;
}

} // namespace

TEST_CASE("Tabulated angle force matches finite differences", "[force][bonded][angle]") {
	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const auto p = generic_angle_config();

	const auto got = run_angle(p, table, false);
	const auto want =
		reference_forces<3>(p, [](const std::array<Vector3, 3>& q) { return reference_angle(q); });

	for (size_t i = 0; i < 3; ++i) {
		INFO("particle " << i);
		REQUIRE(got[i].x == Approx(want[i].x).margin(1e-3));
		REQUIRE(got[i].y == Approx(want[i].y).margin(1e-3));
		REQUIRE(got[i].z == Approx(want[i].z).margin(1e-3));
	}
}

TEST_CASE("Tabulated angle conserves momentum and angular momentum", "[force][bonded][angle]") {
	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const auto p = generic_angle_config();
	const auto f = run_angle(p, table, false);

	const Vector3 net = sum_force<3>(f);
	REQUIRE(net.x == Approx(0.0f).margin(1e-4));
	REQUIRE(net.y == Approx(0.0f).margin(1e-4));
	REQUIRE(net.z == Approx(0.0f).margin(1e-4));

	const Vector3 torque = sum_torque<3>(p, f);
	REQUIRE(torque.x == Approx(0.0f).margin(1e-4));
	REQUIRE(torque.y == Approx(0.0f).margin(1e-4));
	REQUIRE(torque.z == Approx(0.0f).margin(1e-4));
}

TEST_CASE("Tabulated angle force lies in the plane of the three particles",
		  "[force][bonded][angle]") {
	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const auto p = generic_angle_config();
	const auto f = run_angle(p, table, false);

	const Vector3 normal = (p[0] - p[1]).cross(p[2] - p[1]);
	const float scale = normal.length();
	for (size_t i = 0; i < 3; ++i) {
		INFO("particle " << i);
		REQUIRE(f[i].dot(normal) / scale == Approx(0.0f).margin(1e-4));
	}
}

TEST_CASE("Tabulated angle with a rising potential closes the angle", "[force][bonded][angle]") {
	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const std::array<Vector3, 3> p{Vector3(1.0f, 0.0f, 0.0f),
								   Vector3(0.0f, 0.0f, 0.0f),
								   Vector3(0.0f, 1.0f, 0.0f)};
	const auto f = run_angle(p, table, false);
	const float s = static_cast<float>(SLOPE);

	REQUIRE(f[0].x == Approx(0.0f).margin(1e-4));
	REQUIRE(f[0].y == Approx(s).epsilon(1e-3));
	REQUIRE(f[1].x == Approx(-s).epsilon(1e-3));
	REQUIRE(f[1].y == Approx(-s).epsilon(1e-3));
	REQUIRE(f[2].x == Approx(s).epsilon(1e-3));
	REQUIRE(f[2].y == Approx(0.0f).margin(1e-4));
}

TEST_CASE("Tabulated angle energy is the potential split three ways",
		  "[force][bonded][angle][energy]") {
	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const auto p = generic_angle_config();
	const auto f = run_angle(p, table, true);

	const float expected = static_cast<float>(SLOPE * reference_angle(p) / 3.0);
	for (size_t i = 0; i < 3; ++i) {
		INFO("particle " << i);
		REQUIRE(f[i].t == Approx(expected).epsilon(1e-3));
	}
}

TEST_CASE("Tabulated angle skips non-tabulated forms", "[force][bonded][angle]") {
	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const auto p = generic_angle_config();
	const auto f = run_angle(p, table, true, InteractionForm::Analytical);

	for (size_t i = 0; i < 3; ++i) {
		INFO("particle " << i);
		REQUIRE(f[i].length2() == Approx(0.0f).margin(1e-12));
		REQUIRE(f[i].t == Approx(0.0f).margin(1e-12));
	}
}

TEST_CASE("Dihedral angle follows the IUPAC sign convention", "[force][bonded][dihedral]") {
	const PeriodicBox box;
	const ARBD::int4 idx{0, 1, 2, 3};

	std::array<Vector3, 4> p{Vector3(1.0f, 1.0f, 0.0f),
							 Vector3(1.0f, 0.0f, 0.0f),
							 Vector3(0.0f, 0.0f, 0.0f),
							 Vector3(0.0f, 0.0f, 1.0f)};

	REQUIRE(reference_dihedral(p) == Approx(-constants::PI / 2.0).margin(1e-6));

	DihedralGeometry geom = DihedralGeometry::compute(p.data(), idx, &box);
	REQUIRE(geom.dihedral_angle == Approx(-constants::PI / 2.0).margin(1e-5));

	p[3] = Vector3(0.0f, 0.0f, -1.0f);
	geom = DihedralGeometry::compute(p.data(), idx, &box);
	REQUIRE(geom.dihedral_angle == Approx(constants::PI / 2.0).margin(1e-5));
}

TEST_CASE("Dihedral geometry agrees with an independent formula over a sweep",
		  "[force][bonded][dihedral]") {
	const PeriodicBox box;
	const ARBD::int4 idx{0, 1, 2, 3};

	for (int deg = -170; deg <= 170; deg += 10) {
		const double phi = deg * constants::PI / 180.0;
		std::array<Vector3, 4> p{
			Vector3(1.0f, 1.0f, 0.0f),
			Vector3(1.0f, 0.0f, 0.0f),
			Vector3(0.0f, 0.0f, 0.0f),
			Vector3(0.0f, static_cast<float>(std::cos(phi)), static_cast<float>(std::sin(phi)))};

		INFO("deg = " << deg);
		DihedralGeometry geom = DihedralGeometry::compute(p.data(), idx, &box);
		REQUIRE(geom.dihedral_angle == Approx(reference_dihedral(p)).margin(1e-4));
	}
}

TEST_CASE("Tabulated dihedral force matches finite differences", "[force][bonded][dihedral]") {
	const RampTable table(-constants::PI, constants::PI, TABLE_N, SLOPE, true);
	const auto p = generic_dihedral_config();
	REQUIRE(std::abs(reference_dihedral(p)) < 2.0);

	const auto got = run_dihedral(p, table, false);
	const auto want = reference_forces<4>(
		p, [](const std::array<Vector3, 4>& q) { return reference_dihedral(q); });

	for (size_t i = 0; i < 4; ++i) {
		INFO("particle " << i);
		REQUIRE(got[i].x == Approx(want[i].x).margin(2e-3));
		REQUIRE(got[i].y == Approx(want[i].y).margin(2e-3));
		REQUIRE(got[i].z == Approx(want[i].z).margin(2e-3));
	}
}

TEST_CASE("Tabulated dihedral conserves momentum and angular momentum",
		  "[force][bonded][dihedral]") {
	const RampTable table(-constants::PI, constants::PI, TABLE_N, SLOPE, true);
	const auto p = generic_dihedral_config();
	const auto f = run_dihedral(p, table, false);

	const Vector3 net = sum_force<4>(f);
	REQUIRE(net.x == Approx(0.0f).margin(1e-4));
	REQUIRE(net.y == Approx(0.0f).margin(1e-4));
	REQUIRE(net.z == Approx(0.0f).margin(1e-4));

	const Vector3 torque = sum_torque<4>(p, f);
	REQUIRE(torque.x == Approx(0.0f).margin(1e-4));
	REQUIRE(torque.y == Approx(0.0f).margin(1e-4));
	REQUIRE(torque.z == Approx(0.0f).margin(1e-4));
}

TEST_CASE("Tabulated dihedral energy is the potential split four ways",
		  "[force][bonded][dihedral][energy]") {
	const RampTable table(-constants::PI, constants::PI, TABLE_N, SLOPE, true);
	const auto p = generic_dihedral_config();
	const auto f = run_dihedral(p, table, true);

	const float expected =
		static_cast<float>(SLOPE * reference_dihedral(p) / 4.0);
	for (size_t i = 0; i < 4; ++i) {
		INFO("particle " << i);
		REQUIRE(f[i].t == Approx(expected).epsilon(1e-3));
	}
}

TEST_CASE("Tabulated dihedral stays exact on a near-collinear triple",
		  "[force][bonded][dihedral]") {
	// i-j-k is 3 degrees from collinear, well inside the band legacy zeroes.
	// The force is large but it is the correct large force, so it still tracks
	// finite differences. See BondGeometry.md.
	const RampTable table(-constants::PI, constants::PI, TABLE_N, SLOPE, true);
	const std::array<Vector3, 4> p{Vector3(2.0f, 0.05f, 0.0f),
								   Vector3(1.0f, 0.0f, 0.0f),
								   Vector3(0.0f, 0.0f, 0.0f),
								   Vector3(-0.5f, 0.8f, 0.3f)};

	const auto got = run_dihedral(p, table, false);
	const auto want = reference_forces<4>(
		p, [](const std::array<Vector3, 4>& q) { return reference_dihedral(q); });

	REQUIRE(got[0].length() > 10.0f);
	for (size_t i = 0; i < 4; ++i) {
		INFO("particle " << i);
		REQUIRE(got[i].x == Approx(want[i].x).epsilon(1e-2).margin(2e-3));
		REQUIRE(got[i].y == Approx(want[i].y).epsilon(1e-2).margin(2e-3));
		REQUIRE(got[i].z == Approx(want[i].z).epsilon(1e-2).margin(2e-3));
	}
}

TEST_CASE("Tabulated dihedral skips non-tabulated forms", "[force][bonded][dihedral]") {
	const RampTable table(-constants::PI, constants::PI, TABLE_N, SLOPE, true);
	const auto p = generic_dihedral_config();
	const auto f = run_dihedral(p, table, true, InteractionForm::Analytical);

	for (size_t i = 0; i < 4; ++i) {
		INFO("particle " << i);
		REQUIRE(f[i].length2() == Approx(0.0f).margin(1e-12));
		REQUIRE(f[i].t == Approx(0.0f).margin(1e-12));
	}
}

TEST_CASE("Device angle kernel reproduces the host result",
		  "[force][bonded][angle][device]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const RampTable table(0.0, constants::PI, TABLE_N, SLOPE, false);
	const auto p = generic_angle_config();
	const auto expected = run_angle(p, table, true);

	const PeriodicBox box_host;
	DeviceBuffer<PeriodicBox> box(1, res);
	box.copy_from_host(&box_host, 1);

	DeviceBuffer<arbd_real> pot(table.pot.size(), res);
	pot.copy_from_host(table.pot.data(), table.pot.size());

	TabulatedPotential desc = table.descriptor();
	desc.pot = pot.data();
	DeviceBuffer<TabulatedPotential> tables(1, res);
	tables.copy_from_host(&desc, 1);

	DeviceBuffer<Vector3> positions(3, res);
	positions.copy_from_host(p.data(), 3);
	const std::array<Vector3, 3> zeroed{};
	DeviceBuffer<Vector3> forces(3, res);
	forces.copy_from_host(zeroed.data(), 3);

	const ARBD::int3 idx_host{0, 1, 2};
	DeviceBuffer<ARBD::int3> indices(1, res);
	indices.copy_from_host(&idx_host, 1);

	const int zero = 0;
	DeviceBuffer<int> table_indices(1, res);
	table_indices.copy_from_host(&zero, 1);
	const int form_host = static_cast<int>(InteractionForm::Tabulated);
	DeviceBuffer<int> forms(1, res);
	forms.copy_from_host(&form_host, 1);

	launch_tabulated_angles(res,
							indices.data(),
							positions.data(),
							forces.data(),
							tables.data(),
							table_indices.data(),
							forms.data(),
							box.data(),
							true,
							1)
		.wait();

	std::array<Vector3, 3> got{};
	forces.copy_to_host(got.data(), 3);
	for (size_t i = 0; i < 3; ++i) {
		INFO("particle " << i);
		REQUIRE(got[i].x == Approx(expected[i].x).margin(1e-4));
		REQUIRE(got[i].y == Approx(expected[i].y).margin(1e-4));
		REQUIRE(got[i].z == Approx(expected[i].z).margin(1e-4));
		REQUIRE(got[i].t == Approx(expected[i].t).margin(1e-4));
	}
}

TEST_CASE("Device dihedral kernel reproduces the host result",
		  "[force][bonded][dihedral][device]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const RampTable table(-constants::PI, constants::PI, TABLE_N, SLOPE, true);
	const auto p = generic_dihedral_config();
	const auto expected = run_dihedral(p, table, true);

	const PeriodicBox box_host;
	DeviceBuffer<PeriodicBox> box(1, res);
	box.copy_from_host(&box_host, 1);

	DeviceBuffer<arbd_real> pot(table.pot.size(), res);
	pot.copy_from_host(table.pot.data(), table.pot.size());

	TabulatedPotential desc = table.descriptor();
	desc.pot = pot.data();
	DeviceBuffer<TabulatedPotential> tables(1, res);
	tables.copy_from_host(&desc, 1);

	DeviceBuffer<Vector3> positions(4, res);
	positions.copy_from_host(p.data(), 4);
	const std::array<Vector3, 4> zeroed{};
	DeviceBuffer<Vector3> forces(4, res);
	forces.copy_from_host(zeroed.data(), 4);

	const ARBD::int4 idx_host{0, 1, 2, 3};
	DeviceBuffer<ARBD::int4> indices(1, res);
	indices.copy_from_host(&idx_host, 1);

	const int zero = 0;
	DeviceBuffer<int> table_indices(1, res);
	table_indices.copy_from_host(&zero, 1);
	const int form_host = static_cast<int>(InteractionForm::Tabulated);
	DeviceBuffer<int> forms(1, res);
	forms.copy_from_host(&form_host, 1);

	launch_tabulated_dihedrals(res,
							   indices.data(),
							   positions.data(),
							   forces.data(),
							   tables.data(),
							   table_indices.data(),
							   forms.data(),
							   box.data(),
							   true,
							   1)
		.wait();

	std::array<Vector3, 4> got{};
	forces.copy_to_host(got.data(), 4);
	for (size_t i = 0; i < 4; ++i) {
		INFO("particle " << i);
		REQUIRE(got[i].x == Approx(expected[i].x).margin(1e-4));
		REQUIRE(got[i].y == Approx(expected[i].y).margin(1e-4));
		REQUIRE(got[i].z == Approx(expected[i].z).margin(1e-4));
		REQUIRE(got[i].t == Approx(expected[i].t).margin(1e-4));
	}
}
