#include "../catch_boiler.h"

#include "ARBDException.h"
#include "Backend/Buffer.h"
#include "Types/Matrix3.h"
#include "Types/Vector3.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

using Catch::Approx;
using namespace ARBD;
// Static traits --------------------------------------------------------------
static_assert(sizeof(Vector3_t<float>) == 4 * sizeof(float),
			  "Vector3_t<float> must occupy four scalars");
static_assert(alignof(Vector3_t<float>) == 4 * sizeof(float),
			  "Vector3_t<float> must be 16-byte aligned for float");
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
static_assert(
	std::is_same_v<std::common_type_t<Vector3_t<float>, Vector3_t<double>>, Vector3_t<double>>,
	"common_type should promote to double vector");
#endif

#ifdef USE_SYCL
static_assert(sycl::is_device_copyable<Vector3_t<float>>::value,
			  "Vector3_t must be device copyable for SYCL");
#endif

TEST_CASE("Vector3 constructors and indexing", "[vector3]") {
	Vector3_t<float> zero;
	CHECK(zero.x == 0.0f);
	CHECK(zero.y == 0.0f);
	CHECK(zero.z == 0.0f);
	CHECK(zero.t == 0.0f);

	Vector3_t<float> scalar(2.5f);
	CHECK(scalar.x == Approx(2.5f));
	CHECK(scalar.y == Approx(2.5f));
	CHECK(scalar.z == Approx(2.5f));

	Vector3_t<float> xyz(1.f, 2.f, 3.f);
	CHECK(xyz.t == Approx(0.f));

	Vector3_t<float> xyzw(1.f, 2.f, 3.f, 4.f);
	CHECK(xyzw.t == Approx(4.f));

	Vector3_t<float> copy = xyz;
	CHECK(copy == xyz);

	Vector3_t<float> moved = std::move(copy);
	CHECK(moved.x == Approx(1.f));
	CHECK(moved.y == Approx(2.f));
	CHECK(moved.z == Approx(3.f));

	moved[0] = -1.f;
	moved[1] = -2.f;
	moved[2] = -3.f;
	moved[3] = -4.f;
	CHECK(moved.x == Approx(-1.f));
	CHECK(moved.y == Approx(-2.f));
	CHECK(moved.z == Approx(-3.f));
	CHECK(moved.t == Approx(-4.f));

#ifdef HOST_GUARD
	Vector3_t<float> idx_vec;
	CHECK_THROWS_AS(static_cast<void>(idx_vec[4]), ARBD::Exception);
#endif
}

TEST_CASE("Vector3 arithmetic", "[vector3]") {
	Vector3_t<float> a(1.f, 2.f, 3.f);
	Vector3_t<float> b(4.f, 5.f, 6.f);

	auto sum = a + b;
	CHECK(sum.x == Approx(5.f));
	CHECK(sum.y == Approx(7.f));
	CHECK(sum.z == Approx(9.f));

	auto diff = b - a;
	CHECK(diff.x == Approx(3.f));
	CHECK(diff.y == Approx(3.f));
	CHECK(diff.z == Approx(3.f));

	auto scaled = a * 2.0f;
	CHECK(scaled.x == Approx(2.f));
	CHECK(scaled.y == Approx(4.f));
	CHECK(scaled.z == Approx(6.f));

	auto divided = b / 2.0f;
	CHECK(divided.x == Approx(2.f));
	CHECK(divided.y == Approx(2.5f));
	CHECK(divided.z == Approx(3.f));

	Vector3_t<float> accum = a;
	accum += b;
	CHECK(accum == sum);

	accum -= b;
	CHECK(accum == a);

	accum *= 3.f;
	CHECK(accum.x == Approx(3.f));
	CHECK(accum.y == Approx(6.f));
	CHECK(accum.z == Approx(9.f));

	accum /= 3.f;
	CHECK(accum == a);

	auto neg = -a;
	CHECK(neg.x == Approx(-1.f));
	CHECK(neg.y == Approx(-2.f));
	CHECK(neg.z == Approx(-3.f));
}

TEST_CASE("Vector3 geometric operations", "[vector3]") {
	Vector3_t<float> a(1.f, 0.f, 0.f);
	Vector3_t<float> b(0.f, 1.f, 0.f);
	Vector3_t<float> c(1.f, 2.f, 3.f);

	CHECK(a.dot(b) == Approx(0.f));
	CHECK(c.dot(c) == Approx(14.f));

	auto cross = a.cross(b);
	CHECK(cross.x == Approx(0.f));
	CHECK(cross.y == Approx(0.f));
	CHECK(cross.z == Approx(1.f));

	CHECK(c.length2() == Approx(14.f));
	CHECK(c.length() == Approx(std::sqrt(14.f)));
	CHECK(c.rLength2() == Approx(1.f / 14.f));
	CHECK(c.rLength() == Approx(1.f / std::sqrt(14.f)));

	Vector3_t<float> zero(0.f, 0.f, 0.f);
	CHECK(zero.rLength() == Approx(0.f));
	CHECK(zero.rLength2() == Approx(0.f));

	auto floor_vec = Vector3_t<float>(1.9f, -2.1f, 0.5f).element_floor();
	CHECK(floor_vec.x == Approx(1.f));
	CHECK(floor_vec.y == Approx(-3.f));
	CHECK(floor_vec.z == Approx(0.f));

	auto sqrt_vec = Vector3_t<float>::element_sqrt(Vector3_t<float>(4.f, 9.f, 16.f));
	CHECK(sqrt_vec.x == Approx(2.f));
	CHECK(sqrt_vec.y == Approx(3.f));
	CHECK(sqrt_vec.z == Approx(4.f));

	auto elem = c.element_mult(Vector3_t<float>(2.f, 3.f, 4.f));
	CHECK(elem.x == Approx(2.f));
	CHECK(elem.y == Approx(6.f));
	CHECK(elem.z == Approx(12.f));

	const float arr[3] = {2.f, 3.f, 4.f};
	auto elem_arr = c.element_mult(arr);
	CHECK(elem_arr.x == Approx(2.f));
	CHECK(elem_arr.y == Approx(6.f));
	CHECK(elem_arr.z == Approx(12.f));

	auto elem_static = Vector3_t<float>::element_mult(c, Vector3_t<float>(1.f, 0.5f, -1.f));
	CHECK(elem_static.x == Approx(1.f));
	CHECK(elem_static.y == Approx(1.f));
	CHECK(elem_static.z == Approx(-3.f));

	auto angle = a.angle_between(b);
	CHECK(angle == Approx(std::numbers::pi_v<float> / 2.f).margin(1e-5f));
}

TEST_CASE("Vector3 comparisons", "[vector3]") {
	Vector3_t<int> a(1, 2, 3);
	Vector3_t<int> b(1, 2, 3);
	Vector3_t<int> c(3, 2, 1);

	CHECK(a == b);
	CHECK(a != c);
}

TEST_CASE("Vector3 numeric limits", "[vector3]") {
	CHECK(Vector3_t<float>::highest() == std::numeric_limits<float>::max());
	CHECK(Vector3_t<float>::lowest() == std::numeric_limits<float>::lowest());
}

TEST_CASE("Vector3 DeviceBuffer roundtrip", "[vector3][device]") {
	initialize_backend_once();

	Resource res(Global::single_resource_id);
	Vector3_t<float> host_data[2] = {Vector3_t<float>(1.f, 2.f, 3.f),
									 Vector3_t<float>(4.f, 5.f, 6.f)};

	DeviceBuffer<Vector3_t<float>> device(2, res);
	device.copy_from_host(host_data, 2, true);

	Vector3_t<float> back[2];
	device.copy_to_host(back, 2, true);

	CHECK(back[0].x == Approx(1.f));
	CHECK(back[0].y == Approx(2.f));
	CHECK(back[0].z == Approx(3.f));
	CHECK(back[1].x == Approx(4.f));
	CHECK(back[1].y == Approx(5.f));
	CHECK(back[1].z == Approx(6.f));
}

TEST_CASE("Vector3 DeviceBuffer device-to-device copy", "[vector3][device]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	Vector3_t<float> host_data[2] = {Vector3_t<float>(-1.f, -2.f, -3.f),
									 Vector3_t<float>(7.f, 8.f, 9.f)};

	DeviceBuffer<Vector3_t<float>> src(2, res);
	DeviceBuffer<Vector3_t<float>> dst(2, res);

	src.copy_from_host(host_data, 2, true);
	dst.copy_device_to_device(src, 2, true);

	Vector3_t<float> back[2];
	dst.copy_to_host(back, 2, true);

	CHECK(back[0].x == Approx(-1.f));
	CHECK(back[0].y == Approx(-2.f));
	CHECK(back[0].z == Approx(-3.f));
	CHECK(back[1].x == Approx(7.f));
	CHECK(back[1].y == Approx(8.f));
	CHECK(back[1].z == Approx(9.f));
}
