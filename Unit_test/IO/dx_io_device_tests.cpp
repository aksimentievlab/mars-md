#include "../catch_boiler.h"

#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "IO/DxIO.h"
#include "Types/BaseGrid.h"

#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
using Catch::Approx;

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
	auto file_exists = [](const std::string& p) -> bool {
		struct stat sb {};
		return ::stat(p.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
	};
	if (!file_exists(dx_path)) {
		WARN("DX file not found: " << dx_path << "; set ARBD_DX_PATH to override. Skipping.");
		return;
	}

	// Read BaseGrid from DX
	ARBD::BaseGrid<float> grid = ARBD::DXReader::read_from_file<float>(dx_path);
	REQUIRE(grid.size() > 0);

	// Initialize backend
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	if (!manager.isInitialized()) {
		WARN("Backend not initialized; skipping device roundtrip for DX grid");
		return;
	}

	auto& res = manager.get_resource();

	// Copy grid values to host buffer using host-only accessors
	std::vector<float> host_values(grid.size());
	std::memcpy(host_values.data(), grid.data(), grid.size() * sizeof(float));

	// Upload to device
	ARBD::DeviceBuffer<float> d_src(grid.size(), res);
	ARBD::DeviceBuffer<float> d_scaled(grid.size(), res);
	d_src.copy_from_host(host_values.data(), static_cast<idx_t>(host_values.size()), true);

	// Scale values on device using a simple kernel
	constexpr float scale_factor = 0.5f;
	auto scale_kernel = [](idx_t idx, const float* src, float* dst, float scale) {
		dst[idx] = src[idx] * scale;
	};

	ARBD::KernelConfig kernel_cfg =
		ARBD::KernelConfig::for_1d(static_cast<idx_t>(host_values.size()), res);
	kernel_cfg.sync = true;
	ARBD::launch_kernel(res, kernel_cfg, scale_kernel, d_src, d_scaled, scale_factor);

	// Copy back to host and validate a few samples
	std::vector<float> back(grid.size());
	d_scaled.copy_to_host(back.data(), static_cast<idx_t>(back.size()), true);

	CHECK(back.front() == Approx(host_values.front() * scale_factor));
	CHECK(back.back() == Approx(host_values.back() * scale_factor));
	CHECK(back[back.size() / 2] == Approx(host_values[host_values.size() / 2] * scale_factor));
}
