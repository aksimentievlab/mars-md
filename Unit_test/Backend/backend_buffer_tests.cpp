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

TEST_CASE("DeviceBuffer event-based async transfers", "[device][async]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const size_t N = 64;
	std::vector<float> host_in(N), host_out(N, 0.0f);
	for (size_t i = 0; i < N; ++i)
		host_in[i] = static_cast<float>(i) + 1.0f;

	DeviceBuffer<float> buf(N, manager.get_resource());
	Event e1 = buf.copy_from_host_event(host_in.data(), N);
	if (e1.is_valid())
		e1.wait();

	Event e2 = buf.copy_to_host_event(host_out.data(), N);
	if (e2.is_valid())
		e2.wait();

	for (size_t i = 0; i < N; ++i) {
		CHECK(host_out[i] == host_in[i]);
	}
}

TEST_CASE("PinnedBuffer upload/download", "[pinned]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const size_t N = 128;
	std::vector<float> host_in(N), host_out(N, 0.0f);
	for (size_t i = 0; i < N; ++i)
		host_in[i] = 10.0f + static_cast<float>(i);

	PinnedBuffer<float> p(N, manager.get_resource());
	p.upload_to_device(host_in.data(), N);
	p.download_from_device(host_out.data(), N);

	for (size_t i = 0; i < N; ++i) {
		CHECK(host_out[i] == host_in[i]);
	}
}

TEST_CASE("UnifiedBuffer host copies", "[unified]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const size_t N = 96;
	std::vector<int> host_in(N, 123), host_out;

	UnifiedBuffer<int> u(N, manager.get_resource());
	u.copy_from_host(host_in);
	u.copy_to_host(host_out);

	REQUIRE(host_out.size() == host_in.size());
	for (size_t i = 0; i < N; ++i) {
		CHECK(host_out[i] == host_in[i]);
	}
}

TEST_CASE("DeviceBuffer fill", "[device][fill]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const size_t N = 128;
	const int value = 42;
	DeviceBuffer<int> buf(N, manager.get_resource());

	buf.fill(value, buf.get_queue(), true);

	std::vector<int> host_out(N, 0);
	buf.copy_to_host(host_out.data(), N, true);

	for (size_t i = 0; i < N; ++i) {
		CHECK(host_out[i] == value);
	}
}

TEST_CASE("Zero-size buffers", "[edge]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	DeviceBuffer<float> d0(0, manager.get_resource());
	PinnedBuffer<float> p0(0, manager.get_resource());
	UnifiedBuffer<float> u0(0, manager.get_resource());

	CHECK(d0.empty());
	CHECK(u0.empty());
}
