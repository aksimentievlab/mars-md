// Unit tests for Z-order canonical particle reordering (Stage 5 of the reorder plan).
//
// Two levels of coverage:
//   * remap_indices (ungated primitive)      - runs in any zorder-tests build
//   * DeviceParticle::permute + un-permute    - needs -DENABLE_ZORDER_REORDER
//
// The heavier energy/trajectory parity check (npc_6enl_beads, reorder ON vs OFF)
// lives outside the unit tests.

#include "../catch_boiler.h"
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/ZOrderKernels/ZOrderSort.h"
#include "Types/Vector3.h"
#include <algorithm>
#include <vector>

using namespace ARBD;

namespace {

// A scattered, non-trivial layout so the Morton sort is a real permutation.
std::vector<Vector3> make_scattered_positions(size_t n) {
	std::vector<Vector3> pos(n);
	for (size_t i = 0; i < n; ++i) {
		// Cheap deterministic hash spread over a ~10 A box.
		const float x = static_cast<float>((i * 2654435761u) % 997) / 997.0f;
		const float y = static_cast<float>((i * 40503u + 7u) % 991) / 991.0f;
		const float z = static_cast<float>((i * 2246822519u) % 983) / 983.0f;
		pos[i] = Vector3(10.0f * x, 10.0f * y, 10.0f * z);
	}
	return pos;
}

void padded_box(const std::vector<Vector3>& pos, Vector3& box_min, Vector3& box_max) {
	box_min = Vector3(1e30f, 1e30f, 1e30f);
	box_max = Vector3(-1e30f, -1e30f, -1e30f);
	for (const auto& p : pos) {
		box_min.x = std::min(box_min.x, p.x);
		box_min.y = std::min(box_min.y, p.y);
		box_min.z = std::min(box_min.z, p.z);
		box_max.x = std::max(box_max.x, p.x);
		box_max.y = std::max(box_max.y, p.y);
		box_max.z = std::max(box_max.z, p.z);
	}
	const Vector3 pad(0.5f, 0.5f, 0.5f);
	box_min = box_min - pad;
	box_max = box_max + pad;
}

} // namespace


TEST_CASE("remap_indices maps slots through the inverse permutation", "[reorder][remap]") {
	initialize_backend_once();
	Resource device(Global::single_resource_id);
	const size_t n = 128;

	auto pos = make_scattered_positions(n);
	Vector3 box_min, box_max;
	padded_box(pos, box_min, box_max);

	DeviceBuffer<Vector3> d_pos(n, device);
	d_pos.copy_from_host(pos.data(), n);

	ZOrderSort sorter(device, n, ZOrderOptimizationMode::System);
	sorter.sort_particles(d_pos, n, box_min, box_max);

	std::vector<uint32_t> inv(n);
	sorter.get_inverse_indices().copy_to_host(inv.data(), n);

	SECTION("flat int buffer: result[i] == inv[i]") {
		std::vector<int> idx(n);
		for (size_t i = 0; i < n; ++i) {
			idx[i] = static_cast<int>(i);
		}
		DeviceBuffer<int> d_idx(n, device);
		d_idx.copy_from_host(idx.data(), n);

		sorter.remap_indices(d_idx.data(), n);

		std::vector<int> out(n);
		d_idx.copy_to_host(out.data(), n);
		for (size_t i = 0; i < n; ++i) {
			REQUIRE(out[i] == static_cast<int>(inv[i]));
		}
	}

	SECTION("int2 reinterpret (the bonded-edge path): endpoints remap independently") {
		// Mirror exactly how DeviceBondedInteractions remaps bond_indices_.
		// Qualify ARBD::int2 (Vec2<int>) - CUDA's builtin ::int2 is also in scope.
		std::vector<ARBD::int2> edges(n);
		for (size_t i = 0; i < n; ++i) {
			edges[i] = ARBD::int2{static_cast<int>(i), static_cast<int>((i + 1) % n)};
		}
		DeviceBuffer<ARBD::int2> d_edges(n, device);
		d_edges.copy_from_host(edges.data(), n);

		sorter.remap_indices(reinterpret_cast<int*>(d_edges.data()), n * 2);

		std::vector<ARBD::int2> out(n);
		d_edges.copy_to_host(out.data(), n);
		for (size_t i = 0; i < n; ++i) {
			REQUIRE(out[i].x == static_cast<int>(inv[i]));
			REQUIRE(out[i].y == static_cast<int>(inv[(i + 1) % n]));
		}
	}

	SECTION("negative sentinel passes through untouched") {
		std::vector<int> idx = {-1, 0, -1, static_cast<int>(n - 1)};
		DeviceBuffer<int> d_idx(idx.size(), device);
		d_idx.copy_from_host(idx.data(), idx.size());

		sorter.remap_indices(d_idx.data(), idx.size());

		std::vector<int> out(idx.size());
		d_idx.copy_to_host(out.data(), out.size());
		REQUIRE(out[0] == -1);
		REQUIRE(out[1] == static_cast<int>(inv[0]));
		REQUIRE(out[2] == -1);
		REQUIRE(out[3] == static_cast<int>(inv[n - 1]));
	}
}

#ifdef ENABLE_ZORDER_REORDER

TEST_CASE("DeviceParticle::permute reorders the SoA and un-permutes output", "[reorder][permute]") {
	initialize_backend_once();
	Resource device(Global::single_resource_id);
	const idx_t n = 96;

	auto pos = make_scattered_positions(n);
	Vector3 box_min, box_max;
	padded_box(pos, box_min, box_max);

	// Distinct, checkable payloads per particle.
	HostParticleData host;
	host.resize(n);
	for (idx_t i = 0; i < n; ++i) {
		host.global_id[i] = static_cast<int>(i);
		host.type_id[i] = static_cast<int>(i % 4);
		host.pos[i] = pos[i];
		host.mom[i] = Vector3(static_cast<float>(i), -static_cast<float>(i), 2.0f * i);
		host.orient[i] = Vector3(0.0f, 0.0f, 0.0f);
		host.flags[i] = static_cast<uint32_t>(i * 7u + 1u);
	}

	DeviceParticle dp(n, device);
	dp.copy_from_host(host, n);

	ZOrderSort sorter(device, n, ZOrderOptimizationMode::System);
	sorter.sort_particles(dp.pos(), n, box_min, box_max);

	std::vector<uint32_t> sorted(n);
	sorter.get_sorted_indices().copy_to_host(sorted.data(), n);

	dp.permute(sorter);
	REQUIRE(dp.is_reordered());

	SECTION("in-memory slot i holds original particle sorted[i]") {
		std::vector<int> id_slot(n);
		std::vector<Vector3> pos_slot(n);
		dp.id().copy_to_host(id_slot.data(), n);
		dp.pos().copy_to_host(pos_slot.data(), n);
		for (idx_t i = 0; i < n; ++i) {
			const uint32_t orig = sorted[i];
			REQUIRE(id_slot[i] == static_cast<int>(orig)); // global_id was iota
			REQUIRE(pos_slot[i].x == Catch::Approx(pos[orig].x));
			REQUIRE(pos_slot[i].y == Catch::Approx(pos[orig].y));
			REQUIRE(pos_slot[i].z == Catch::Approx(pos[orig].z));
		}
	}

	SECTION("copy_to_host emits in original global_id order (round-trip identity)") {
		HostParticleData out;
		out.resize(n);
		dp.copy_to_host(out, n);
		for (idx_t g = 0; g < n; ++g) {
			REQUIRE(out.global_id[g] == static_cast<int>(g));
			REQUIRE(out.type_id[g] == static_cast<int>(g % 4));
			REQUIRE(out.flags[g] == static_cast<uint32_t>(g * 7u + 1u));
			REQUIRE(out.pos[g].x == Catch::Approx(pos[g].x));
			REQUIRE(out.pos[g].y == Catch::Approx(pos[g].y));
			REQUIRE(out.pos[g].z == Catch::Approx(pos[g].z));
			REQUIRE(out.mom[g].x == Catch::Approx(static_cast<float>(g)));
			REQUIRE(out.mom[g].y == Catch::Approx(-static_cast<float>(g)));
			REQUIRE(out.mom[g].z == Catch::Approx(2.0f * g));
		}
	}
}

#endif // ENABLE_ZORDER_REORDER