#include "../catch_boiler.h"

#include "Types/Matrix3.h"
#include "Types/Vector3.h"

#include <cmath>
#include <numbers>

using Catch::Approx;
using namespace ARBD;

namespace {

using M3 = Matrix3_t<double>;
using V3 = Vector3_t<double>;

/// Element (row, col). Matrix3_t stores columns, so this is col(c)[r].
double at(const M3& m, int row, int col) {
	const V3& c = (col == 0) ? m.ex() : (col == 1) ? m.ey() : m.ez();
	return (row == 0) ? c.x : (row == 1) ? c.y : c.z;
}

/// The rotation angle a Cayley parameter t encodes.
double angle_of(double t) {
	return 2.0 * std::atan(0.5 * t);
}

void check_close(const V3& got, const V3& want, double eps = 1e-12) {
	CHECK(got.x == Approx(want.x).margin(eps));
	CHECK(got.y == Approx(want.y).margin(eps));
	CHECK(got.z == Approx(want.z).margin(eps));
}

void check_identity(const M3& m, double eps = 1e-12) {
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			CHECK(at(m, r, c) == Approx(r == c ? 1.0 : 0.0).margin(eps));
		}
	}
}

constexpr double kPi = std::numbers::pi_v<double>;

} // namespace

TEST_CASE("Matrix3 constructor takes columns, not rows", "[matrix3]") {
	// Everything below reads elements through at(), which assumes this.
	M3 m(V3(1, 2, 3), V3(4, 5, 6), V3(7, 8, 9));
	CHECK(at(m, 0, 0) == Approx(1.0));
	CHECK(at(m, 1, 0) == Approx(2.0));
	CHECK(at(m, 2, 0) == Approx(3.0));
	CHECK(at(m, 0, 1) == Approx(4.0));
	CHECK(at(m, 0, 2) == Approx(7.0));

	// transform() is a column combination: M * e_1 is the second column.
	check_close(m * V3(0, 1, 0), V3(4, 5, 6));
}

TEST_CASE("Rotation matrices are right-handed", "[matrix3][rotation]") {
	// Right-handed means each axis carries the next one cyclically forward:
	// x -> y -> z -> x. The parameter encodes 2*atan(t/2), so a quarter turn
	// is t = 2*tan(pi/4) = 2.
	const double quarter = 2.0 * std::tan(0.25 * kPi);

	SECTION("z carries x_hat toward +y") {
		check_close(rotation_matrix_z(quarter) * V3(1, 0, 0), V3(0, 1, 0), 1e-12);
	}
	SECTION("x carries y_hat toward +z") {
		check_close(rotation_matrix_x(quarter) * V3(0, 1, 0), V3(0, 0, 1), 1e-12);
	}
	SECTION("y carries z_hat toward +x") {
		check_close(rotation_matrix_y(quarter) * V3(0, 0, 1), V3(1, 0, 0), 1e-12);
	}

	SECTION("small positive angle tips x_hat to +y, not -y") {
		// Guards the sign independently of the quarter-turn algebra above.
		const V3 v = rotation_matrix_z(1e-3) * V3(1, 0, 0);
		CHECK(v.y > 0.0);
		CHECK(v.z == Approx(0.0).margin(1e-15));
	}
}

TEST_CASE("Rotation matrices are proper rotations", "[matrix3][rotation]") {
	// Sweep well past small angles: the Cayley form c = (1-t^2/4)/(1+t^2/4),
	// s = t/(1+t^2/4) satisfies c^2 + s^2 == 1 identically, so orthonormality
	// must hold for every t, not just tiny ones.
	for (double t : {-8.0, -1.0, -1e-4, 0.0, 1e-4, 0.3, 1.0, 8.0}) {
		CAPTURE(t);
		for (const M3& r : {rotation_matrix_x(t), rotation_matrix_y(t), rotation_matrix_z(t)}) {
			CHECK(r.det() == Approx(1.0).margin(1e-12));
			check_identity(r * r.transpose());
			CHECK(r.ex().length() == Approx(1.0).margin(1e-12));
			CHECK(r.ey().length() == Approx(1.0).margin(1e-12));
			CHECK(r.ez().length() == Approx(1.0).margin(1e-12));
		}
	}
}

TEST_CASE("R(t) * R(-t) is the identity", "[matrix3][rotation]") {
	for (double t : {-3.0, -0.5, 1e-3, 0.75, 5.0}) {
		CAPTURE(t);
		check_identity(rotation_matrix_x(t) * rotation_matrix_x(-t));
		check_identity(rotation_matrix_y(t) * rotation_matrix_y(-t));
		check_identity(rotation_matrix_z(t) * rotation_matrix_z(-t));
	}
	// Negating t is the same as transposing, for a rotation.
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			CHECK(at(rotation_matrix_y(-1.25), r, c) ==
				  Approx(at(rotation_matrix_y(1.25).transpose(), r, c)).margin(1e-12));
		}
	}
}

TEST_CASE("Cayley parameter encodes 2*atan(t/2), not t", "[matrix3][rotation]") {
	// Legacy calls this an approximation good for small angles, but it is exact
	// for the angle 2*atan(t/2). Replacing it with plain cos(t)/sin(t) would
	// change every DLM trajectory, so pin the identity.
	const double t = 0.9;
	const double a = angle_of(t);
	const M3 r = rotation_matrix_z(t);
	CHECK(at(r, 0, 0) == Approx(std::cos(a)).margin(1e-12));
	CHECK(at(r, 1, 0) == Approx(std::sin(a)).margin(1e-12));
	CHECK(a < t); // the encoded angle lags the parameter
	CHECK(angle_of(1e-6) == Approx(1e-6).margin(1e-15)); // ...but agrees as t -> 0
}

TEST_CASE("Rotation matrices match legacy RigidBody::Rx/Ry/Rz", "[matrix3][rotation][legacy]") {
	// Legacy builds these from nine row-major scalars (RigidBody.cu:528-557),
	// arbd2 from three columns. The two conventions must agree element for
	// element or every ported DLM trajectory diverges.
	const double t = 0.6;
	const double qt = 0.25 * t * t;
	const double c = (1.0 - qt) / (1.0 + qt);
	const double s = t / (1.0 + qt);

	const double rx[3][3] = {{1, 0, 0}, {0, c, -s}, {0, s, c}};
	const double ry[3][3] = {{c, 0, s}, {0, 1, 0}, {-s, 0, c}};
	const double rz[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};

	const M3 got_x = rotation_matrix_x(t);
	const M3 got_y = rotation_matrix_y(t);
	const M3 got_z = rotation_matrix_z(t);

	for (int r = 0; r < 3; ++r) {
		for (int col = 0; col < 3; ++col) {
			CAPTURE(r, col);
			CHECK(at(got_x, r, col) == Approx(rx[r][col]).margin(1e-12));
			CHECK(at(got_y, r, col) == Approx(ry[r][col]).margin(1e-12));
			CHECK(at(got_z, r, col) == Approx(rz[r][col]).margin(1e-12));
		}
	}
}

TEST_CASE("normalize_orientation returns a right-handed orthonormal frame", "[matrix3]") {
	SECTION("no-op on a rotation") {
		const M3 r = rotation_matrix_y(0.4) * rotation_matrix_z(-1.1);
		const M3 n = normalize_orientation(r);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				CAPTURE(i, j);
				CHECK(at(n, i, j) == Approx(at(r, i, j)).margin(1e-12));
			}
		}
	}

	SECTION("repairs a skewed, unnormalized frame") {
		M3 skewed(V3(2.0, 0.0, 0.0), V3(0.3, 1.7, 0.0), V3(0.0, 0.1, 0.9));
		const M3 n = normalize_orientation(skewed);
		CHECK(n.det() == Approx(1.0).margin(1e-12));
		check_identity(n * n.transpose());
		// Gram-Schmidt keeps the first column's direction and rebuilds the rest.
		check_close(n.ex(), V3(1, 0, 0));
	}
}

TEST_CASE("Matrix3 float rotations stay orthonormal", "[matrix3][rotation]") {
	// The DLM kernel runs in whatever arbd_real is; make sure the float
	// instantiation is not quietly worse than the double one.
	using M3f = Matrix3_t<float>;
	const M3f r = rotation_matrix_z(0.25f) * rotation_matrix_x(-0.75f);
	CHECK(r.det() == Approx(1.0f).margin(1e-6));
	CHECK(r.ex().length() == Approx(1.0f).margin(1e-6));
	CHECK(r.ey().length() == Approx(1.0f).margin(1e-6));
	CHECK(r.ez().length() == Approx(1.0f).margin(1e-6));
}
