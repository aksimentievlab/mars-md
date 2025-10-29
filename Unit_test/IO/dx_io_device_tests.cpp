#include "../catch_boiler.h"

#include "Backend/Buffer.h"
#include "Header.h"
#include "IO/DxIO.h"

#include <cstring>
#include <filesystem>
using Catch::Approx;
using namespace ARBD;

namespace {
// Default external path; can be overridden with ARBD_DX_PATH env var
constexpr const char* kDefaultDxPath = "/data/server5/pinyili2/harmonic_z.dx";

std::string get_dx_path() {
	if (const char* envp = std::getenv("ARBD_DX_PATH"))
		return std::string(envp);
	return std::string(kDefaultDxPath);
}
} // namespace

TEST_CASE("DX IO: read and roundtrip to device", "[io][device][dx]") {
	// Locate DX file
	const std::string dx_path = get_dx_path();
	if (!std::filesystem::exists(dx_path)) {
		WARN("DX file not found: " << dx_path << "; set ARBD_DX_PATH to override. Skipping.");
		return;
	}

	// Read BaseGrid from DX
	BaseGrid<float> grid = DXReader::read_from_file<float>(dx_path);
	REQUIRE(grid.size() > 0);

	// Initialize backend
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	if (!manager.isInitialized()) {
		WARN("Backend not initialized; skipping device roundtrip for DX grid");
		return;
	}

	auto& res = manager.get_resource();

#ifdef HOST_GUARD
	// Copy grid values to host buffer using host-only accessors
	std::vector<float> host_values(grid.size());
	std::memcpy(host_values.data(), grid.data(), grid.size() * sizeof(float));

	// Upload to device and perform device-to-device copy
	DeviceBuffer<float> d_src(grid.size(), res);
	DeviceBuffer<float> d_dst(grid.size(), res);

	d_src.copy_from_host(host_values.data(), host_values.size(), true);
	d_dst.copy_device_to_device(d_src, d_src.size(), true);

	// Download back and compare a few samples (avoid O(n) checks on huge grids)
	std::vector<float> back(grid.size());
	d_dst.copy_to_host(back.data(), back.size(), true);

	// Validate endpoints and a mid element
	CHECK(back.front() == Approx(host_values.front()));
	CHECK(back.back() == Approx(host_values.back()));
	CHECK(back[back.size() / 2] == Approx(host_values[host_values.size() / 2]));
#else
	WARN("HOST_GUARD not enabled; skipping host extraction for DX grid");
#endif
}
