#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Types/BaseGridDevice.h"
#include "basegrid_device.h"

#include <vector>
using Catch::Approx;
using namespace MARS;

TEST_CASE("BaseGrid device interpolate and nearest", "[basegrid][device]") {
	initialize_backend_once();

	auto res = Resource{Global::single_resource_id};

	using T = float;
	using Params = Test::Params<T>;
	Params h_params;
	h_params.origin = Vector3_t<T>(0, 0, 0);
	h_params.basis_inv = Matrix3_t<T>(T(1)).inverse();
	h_params.dims = Vector3_t<idx_t>(2, 2, 2);

	// Grid values laid out with linear index = k + j*nz + i*ny*nz
	// For 2x2x2, indices 0..7
	std::vector<T> h_values = {0, 1, 2, 3, 4, 5, 6, 7};

	DeviceBuffer<T> d_values(h_values.size(), res);
	d_values.copy_from_host(h_values.data(), static_cast<idx_t>(h_values.size()), true);

	// Positions to sample
	std::vector<Vector3_t<T>> h_pos = {
		Vector3_t<T>(0, 0, 0),			  // exactly v000 -> 0
		Vector3_t<T>(0.5f, 0, 0),		  // between (0,0,0) and (1,0,0)
		Vector3_t<T>(0.25f, 0.25f, 0.25f) // inside first cell
	};

	DeviceBuffer<Vector3_t<T>> d_pos(h_pos.size(), res);
	d_pos.copy_from_host(h_pos.data(), static_cast<idx_t>(h_pos.size()), true);

	DeviceBuffer<T> d_out_interp(h_pos.size(), res);
	DeviceBuffer<T> d_out_nearest(h_pos.size(), res);

	// Copy params to device buffer
	DeviceBuffer<Params> d_params(1, res);
	d_params.copy_from_host(&h_params, 1, true);

	Event evt = Test::launch_grid_query<T>(res,
										   d_pos,
										   d_values,
										   d_out_interp,
										   d_out_nearest,
										   d_params,
										   static_cast<idx_t>(h_pos.size()));
	evt.wait();

	std::vector<T> out_interp(h_pos.size());
	std::vector<T> out_nearest(h_pos.size());
	d_out_interp.copy_to_host(out_interp.data(), static_cast<idx_t>(out_interp.size()), true);
	d_out_nearest.copy_to_host(out_nearest.data(), static_cast<idx_t>(out_nearest.size()), true);

	// Nearest samples should match linear index for exact grid points
	REQUIRE(out_nearest[0] == Approx(0.0f));

	// Interpolation at exact grid point should equal v000 = 0
	REQUIRE(out_interp[0] == Approx(0.0f));

	// For (0.5,0,0): interpolate between v000=0 and v100=4 -> 2
	REQUIRE(out_interp[1] == Approx(2.0f));

	// Nearest at (0.5,0,0) should round to ix=1 -> index of (1,0,0) = 4
	REQUIRE(out_nearest[1] == Approx(4.0f));

	// Inside first cell: value between 0..7; check it's within bounds and positive
	REQUIRE(out_interp[2] >= 0.0f);
	REQUIRE(out_interp[2] <= 7.0f);
}
