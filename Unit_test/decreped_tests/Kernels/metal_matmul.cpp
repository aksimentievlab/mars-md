#ifdef USE_METAL
#include <unistd.h>
#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"
#include "Backend/Kernels.h"
#include "Backend/KernelConfig.h"
#include "Types/Matrix3.h"
#include "Types/Vector3.h"

TEST_CASE("Metal Matrix multiply kernel", "[metal][kernels][matmul]") {
	MARS::METAL::Manager::init();
	MARS::METAL::Manager::load_info();

	MARS::Resource metal_res(MARS::ResourceType::METAL, 0);

	const uint M = 3;
	const uint N = 3;
	const uint K = 3;

	// Define matrices A and B using MARS types (column-major internally)
	MARS::Matrix3f A(
		MARS::Vector3_t<float>(1.f, 2.f, 3.f),
		MARS::Vector3_t<float>(4.f, 5.f, 6.f),
		MARS::Vector3_t<float>(7.f, 8.f, 9.f));
	MARS::Matrix3f B(
		MARS::Vector3_t<float>(9.f, 8.f, 7.f),
		MARS::Vector3_t<float>(6.f, 5.f, 4.f),
		MARS::Vector3_t<float>(3.f, 2.f, 1.f));

	// Helper to access (row, col) from column-major Matrix3f
	auto get_rc = [](const MARS::Matrix3f& Mx, uint row, uint col) -> float {
		const MARS::Vector3_t<float>& c = (col == 0 ? Mx.ex() : (col == 1 ? Mx.ey() : Mx.ez()));
		return (row == 0 ? c.x : (row == 1 ? c.y : c.z));
	};

	// Flatten A (MxK) and B (KxN) to row-major float arrays expected by the kernel
	float hA[9];
	float hB[9];
	for (uint i = 0; i < M; ++i) {
		for (uint j = 0; j < K; ++j) {
			hA[i * K + j] = get_rc(A, i, j);
		}
	}
	for (uint i = 0; i < K; ++i) {
		for (uint j = 0; j < N; ++j) {
			hB[i * N + j] = get_rc(B, i, j);
		}
	}

	// CPU reference C = A * B
	float hC_expected[9];
	for (uint i = 0; i < M; ++i) {
		for (uint j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (uint k = 0; k < K; ++k) {
				sum += hA[i * K + k] * hB[k * N + j];
			}
			hC_expected[i * N + j] = sum;
		}
	}

	MARS::DeviceBuffer<float> dA(M * K);
	MARS::DeviceBuffer<float> dB(K * N);
	MARS::DeviceBuffer<float> dC(M * N);
	dA.copy_from_host(hA, M * K);
	dB.copy_from_host(hB, K * N);

	MARS::KernelConfig config;
	config.async = false;  // Force synchronous execution for testing
	// Use the new 2D auto-configure for efficient matrix multiplication
	config.auto_configure_2d(N, M, metal_res);  // width=cols, height=rows

	// Launch precompiled kernel: matmul_kernel
	MARS::Event evt = MARS::launch_metal_kernel(
		metal_res,
		static_cast<idx_t>(M * N),
		config,
		"matmul_kernel",
		dA,
		dB,
		dC,
		K);
	evt.wait();

	float hC[9];
	dC.copy_to_host(hC, M * N);

	for (size_t i = 0; i < M * N; ++i) {
		REQUIRE(hC[i] == Catch::Approx(hC_expected[i]).margin(1e-5f));
	}
}
#endif // USE_METAL