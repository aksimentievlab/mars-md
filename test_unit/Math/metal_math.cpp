#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"

// -----------------------------------------------------------------------------
// Metal Vector3 and Matrix3 Kernel Tests
// -----------------------------------------------------------------------------

#include "Types/Vector3.h"
#include "Types/Matrix3.h"

using namespace ARBD;

// Helper: Check if Metal is available, else skip
#define REQUIRE_METAL_OR_SKIP() \
    ARBD::METAL::Manager::load_info(); \
    if (!ARBD::METAL::Manager::get_library()) { \
        SKIP("Metal library not loaded - skipping Metal kernel test"); \
    }

// -----------------------------
// Vector3 Metal Kernel Tests
// -----------------------------

TEST_CASE("Metal Vector3 Basic Arithmetic Kernels", "[metal][vector3][kernels]") {
    REQUIRE_METAL_OR_SKIP();

    ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
    constexpr size_t n = 16;

    std::vector<Vector3_t<float>> host_a(n), host_b(n), host_out(n);
    for (size_t i = 0; i < n; ++i) {
        host_a[i] = Vector3_t<float>(float(i), float(i+1), float(i+2));
        host_b[i] = Vector3_t<float>(float(2*i), float(2*i+1), float(2*i+2));
    }
    const float host_c=2.0f;
    // Create buffers using the correct constructor (size only)
    DeviceBuffer<Vector3_t<float>> buf_a(n);
    DeviceBuffer<Vector3_t<float>> buf_b(n);
    DeviceBuffer<Vector3_t<float>> buf_out(n);
    DeviceBuffer<float> buf_scalar(1);

    buf_a.copy_from_host(host_a.data(), n);
    buf_b.copy_from_host(host_b.data(), n);
    buf_scalar.copy_from_host(&host_c, 1);

    ARBD::KernelConfig config;
    config.async = false;
    config.grid_size = {n, 1, 1};  // Use grid_size instead of global_size

    // Launch vector_operations_kernel using the correct API
    ARBD::Event event = ARBD::launch_metal_kernel(
        metal_res,
        n,
        config,
        "vector_operations_kernel",
        buf_a,
        buf_b,
        buf_scalar,
        buf_out
    );

    event.wait();
    buf_out.copy_to_host(host_out.data(), n);

    for (size_t i = 0; i < n; ++i) {
        Vector3_t<float> expected = (host_a[i] + host_b[i]) * 2.0f;
        float expected_dot = host_a[i].dot(host_b[i]);
        REQUIRE(host_out[i].x == Catch::Approx(expected.x));
        REQUIRE(host_out[i].y == Catch::Approx(expected.y));
        REQUIRE(host_out[i].z == Catch::Approx(expected.z));
        REQUIRE(host_out[i].w == Catch::Approx(expected_dot));
    }
}

TEST_CASE("Metal Vector3 Cross Product Kernel", "[metal][vector3][cross][kernels]") {
    REQUIRE_METAL_OR_SKIP();

    ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
    constexpr size_t n = 8;

    std::vector<Vector3_t<float>> host_a(n), host_b(n), host_out(n);
    for (size_t i = 0; i < n; ++i) {
        host_a[i] = Vector3_t<float>(float(i+1), float(i+2), float(i+3));
        host_b[i] = Vector3_t<float>(float(2*i+1), float(2*i+2), float(2*i+3));
    }

    DeviceBuffer<Vector3_t<float>> buf_a(n);
    DeviceBuffer<Vector3_t<float>> buf_b(n);
    DeviceBuffer<Vector3_t<float>> buf_out(n);

    buf_a.copy_from_host(host_a.data(), n);
    buf_b.copy_from_host(host_b.data(), n);

    ARBD::KernelConfig config;
    config.async = false;
    config.grid_size = {n, 1, 1};

    ARBD::Event event = ARBD::launch_metal_kernel(
        metal_res,
        n,
        config,
        "cross_product_kernel",
        buf_a,
        buf_b,
        buf_out
    );

    event.wait();
    buf_out.copy_to_host(host_out.data(), n);

    for (size_t i = 0; i < n; ++i) {
        Vector3_t<float> expected = host_a[i].cross(host_b[i]);
        REQUIRE(host_out[i].x == Catch::Approx(expected.x));
        REQUIRE(host_out[i].y == Catch::Approx(expected.y));
        REQUIRE(host_out[i].z == Catch::Approx(expected.z));
    }
}

TEST_CASE("Metal Vector3 Length and Normalization Kernel", "[metal][vector3][length][kernels]") {
    REQUIRE_METAL_OR_SKIP();

    ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
    constexpr size_t n = 10;

    std::vector<Vector3_t<float>> host_in(n), host_out(n);
    std::vector<float> host_lengths(n);
    for (size_t i = 0; i < n; ++i) {
        host_in[i] = Vector3_t<float>(float(i+1), float(i+2), float(i+3));
    }

    DeviceBuffer<Vector3_t<float>> buf_in(n);
    DeviceBuffer<Vector3_t<float>> buf_out(n);
    DeviceBuffer<float> buf_lengths(n);

    buf_in.copy_from_host(host_in.data(), n);

    ARBD::KernelConfig config;
    config.async = false;
    config.grid_size = {n, 1, 1};

    ARBD::Event event = ARBD::launch_metal_kernel(
        metal_res,
        n,
        config,
        "length_operations_kernel",
        buf_in,
        buf_out,
        buf_lengths
    );

    event.wait();
    buf_out.copy_to_host(host_out.data(), n);
    buf_lengths.copy_to_host(host_lengths.data(), n);

    for (size_t i = 0; i < n; ++i) {
        float expected_len = host_in[i].length();
        REQUIRE(host_lengths[i] == Catch::Approx(expected_len));
        if (expected_len > 0.0f) {
            Vector3_t<float> expected_norm = host_in[i] * (1.0f / expected_len);
            REQUIRE(host_out[i].x == Catch::Approx(expected_norm.x));
            REQUIRE(host_out[i].y == Catch::Approx(expected_norm.y));
            REQUIRE(host_out[i].z == Catch::Approx(expected_norm.z));
        } else {
            REQUIRE(host_out[i].x == 0.0f);
            REQUIRE(host_out[i].y == 0.0f);
            REQUIRE(host_out[i].z == 0.0f);
        }
    }
}

// -----------------------------
// Matrix3 Metal Kernel Tests
// -----------------------------
TEST_CASE("Metal Matrix3 Elementwise Multiplication Kernel", "[metal][matrix3][kernels]") {
    REQUIRE_METAL_OR_SKIP();

    ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
    constexpr size_t n = 4;

    std::vector<Matrix3_t<float>> host_a(n), host_b(n), host_out(n);
    for (size_t i = 0; i < n; ++i) {
        Matrix3_t<float> m1, m2;
        // Access matrix elements through column vectors
        m1.ex().x = float(i + 1); m1.ex().y = float(i + 2); m1.ex().z = float(i + 3);
        m1.ey().x = float(i + 4); m1.ey().y = float(i + 5); m1.ey().z = float(i + 6);
        m1.ez().x = float(i + 7); m1.ez().y = float(i + 8); m1.ez().z = float(i + 9);
        
        m2.ex().x = float(2 * (i + 1)); m2.ex().y = float(2 * (i + 2)); m2.ex().z = float(2 * (i + 3));
        m2.ey().x = float(2 * (i + 4)); m2.ey().y = float(2 * (i + 5)); m2.ey().z = float(2 * (i + 6));
        m2.ez().x = float(2 * (i + 7)); m2.ez().y = float(2 * (i + 8)); m2.ez().z = float(2 * (i + 9));
        
        host_a[i] = m1;
        host_b[i] = m2;
    }

    DeviceBuffer<Matrix3_t<float>> buf_a(n);
    DeviceBuffer<Matrix3_t<float>> buf_b(n);
    DeviceBuffer<Matrix3_t<float>> buf_out(n);

    buf_a.copy_from_host(host_a.data(), n);
    buf_b.copy_from_host(host_b.data(), n);

    ARBD::KernelConfig config;
    config.async = false;
    config.grid_size = {n, 1, 1};

    // Launch matrix3_mult_kernel (must be implemented in matrix3.metal)
    ARBD::Event event = ARBD::launch_metal_kernel(
        metal_res,
        n,
        config,
        "matrix3_mult_kernel",
        buf_a,
        buf_b,
        buf_out
    );

    event.wait();
    buf_out.copy_to_host(host_out.data(), n);

    for (size_t i = 0; i < n; ++i) {
        Matrix3_t<float> expected;
        // Element-wise multiplication using column vectors
        expected.ex().x = host_a[i].ex().x * host_b[i].ex().x;
        expected.ex().y = host_a[i].ex().y * host_b[i].ex().y;
        expected.ex().z = host_a[i].ex().z * host_b[i].ex().z;
        expected.ey().x = host_a[i].ey().x * host_b[i].ey().x;
        expected.ey().y = host_a[i].ey().y * host_b[i].ey().y;
        expected.ey().z = host_a[i].ey().z * host_b[i].ey().z;
        expected.ez().x = host_a[i].ez().x * host_b[i].ez().x;
        expected.ez().y = host_a[i].ez().y * host_b[i].ez().y;
        expected.ez().z = host_a[i].ez().z * host_b[i].ez().z;
        
        // Verify results using column vectors
        REQUIRE(host_out[i].ex().x == Catch::Approx(expected.ex().x));
        REQUIRE(host_out[i].ex().y == Catch::Approx(expected.ex().y));
        REQUIRE(host_out[i].ex().z == Catch::Approx(expected.ex().z));
        REQUIRE(host_out[i].ey().x == Catch::Approx(expected.ey().x));
        REQUIRE(host_out[i].ey().y == Catch::Approx(expected.ey().y));
        REQUIRE(host_out[i].ey().z == Catch::Approx(expected.ey().z));
        REQUIRE(host_out[i].ez().x == Catch::Approx(expected.ez().x));
        REQUIRE(host_out[i].ez().y == Catch::Approx(expected.ez().y));
        REQUIRE(host_out[i].ez().z == Catch::Approx(expected.ez().z));
    }
}