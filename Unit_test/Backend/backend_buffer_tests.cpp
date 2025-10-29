#include "../catch_boiler.h"

using namespace ARBD;

TEST_CASE("DeviceBuffer copy round-trip", "[device]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const size_t N = 16;
	std::vector<int> host_in(N), host_out(N, 0);
	for (size_t i = 0; i < N; ++i)
		host_in[i] = static_cast<int>(i * 3 + 7);

	DeviceBuffer<int> buf(N, manager.get_resource());
	buf.copy_from_host(host_in.data(), N, true);
	buf.copy_to_host(host_out.data(), N, true);

	for (size_t i = 0; i < N; ++i) {
		CHECK(host_out[i] == host_in[i]);
	}
}

TEST_CASE("DeviceBuffer device-to-device copy", "[device]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const size_t N = 32;
	std::vector<float> host_in(N), host_out(N, 0.f);
	for (size_t i = 0; i < N; ++i)
		host_in[i] = 0.25f * static_cast<float>(i);

	DeviceBuffer<float> a(N, manager.get_resource());
	DeviceBuffer<float> b(N, manager.get_resource());
	a.copy_from_host(host_in.data(), N, true);
	b.copy_device_to_device(a, N, true);
	b.copy_to_host(host_out.data(), N, true);

	for (size_t i = 0; i < N; ++i) {
		CHECK(host_out[i] == host_in[i]);
	}
}
