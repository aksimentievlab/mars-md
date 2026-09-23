#include "../catch_boiler.h"
#include "PatchOperation/Random/Random.h"
#include "PatchOperation/ZOrderKernels/DeviceRadix.h"
#if defined(PROJECT_USES_SYCL_ICPX) && SYCL_DEVICE_TYPE == 2
#include "PatchOperation/ZOrderKernels/oneapiSort.h"
#define MARS_HAVE_ONEDPL_SORT 1
#endif
#include <numeric>
#include <vector>

#ifndef MARS_SORT_BENCH_REPS
#define MARS_SORT_BENCH_REPS 5
#endif

short single_resource_id = Global::single_resource_id;

using namespace MARS;
using namespace Tests;
Resource single_resource(single_resource_id);
void generate_random_data_drs(const Resource& device,
							  std::vector<uint32_t>& data,
							  uint32_t seed,
							  float entropy = 1.0f) {
	Random<Resource> rng(device, 128);
	rng.init(seed, 0);
	DeviceBuffer<uint32_t> d_data(data.size(), device.id());
	uint32_t min_val = 0;
	uint32_t max_val;
	if (entropy >= 0.99f) {
		max_val = UINT32_MAX;
	} else if (entropy >= 0.8f) {
		max_val = (1u << 26) - 1;
	} else if (entropy >= 0.5f) {
		max_val = (1u << 18) - 1;
	} else {
		max_val = (1u << 7) - 1;
	}
	Event gen_event = rng.generate_uniform(d_data, min_val, max_val);
	gen_event.wait();
	d_data.copy_to_host(data.data(), data.size());
}

#ifdef USE_SYCL
void report_sort_throughput(const Resource& device,
							const char* label,
							size_t size,
							std::vector<long long> samples_ms,
							uint32_t radix_passes) {
	std::sort(samples_ms.begin(), samples_ms.end());
	const long long ms = samples_ms[samples_ms.size() / 2];
	const long long ms_min = samples_ms.front();
	const long long ms_max = samples_ms.back();

	sycl::device dev = device.get_sycl_queue(0).get_device();
	const auto affinity = dev.get_info<sycl::info::device::partition_type_affinity_domain>();
	const bool is_tile = affinity != sycl::info::partition_affinity_domain::not_applicable;
	const uint32_t max_sub = dev.get_info<sycl::info::device::partition_max_sub_devices>();

	const double seconds = double(ms) / 1000.0;
	const double gkeys = double(size) / seconds / 1e9;
	const double gbytes = double(radix_passes) * double(size) * 16.0 / seconds / 1e9;

	std::cout << "\n[" << label << "] Resource " << device.id() << " -> "
			  << dev.get_info<sycl::info::device::name>() << "\n"
			  << "  platform          : "
			  << dev.get_platform().get_info<sycl::info::platform::name>() << "\n"
			  << "  sub-device (tile) : " << (is_tile ? "yes" : "no")
			  << "  (partition_max_sub_devices=" << max_sub << ")\n"
			  << "  compute units     : " << dev.get_info<sycl::info::device::max_compute_units>()
			  << "\n"
			  << "  global memory     : "
			  << dev.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024 * 1024)
			  << " GiB\n"
			  << "  sorted            : " << size << " keys, median of " << samples_ms.size()
			  << " reps\n"
			  << "  time              : " << ms << " ms  (min " << ms_min << ", max " << ms_max
			  << ", spread " << (100.0 * double(ms_max - ms_min) / double(ms)) << "%)\n"
			  << "  throughput        : " << gkeys << " Gkeys/s\n"
			  << "  effective bw      : " << gbytes << " GB/s (" << radix_passes
			  << " passes x r+w of key+payload)" << std::endl;
}
#endif

TEST_CASE("DeviceRadixSort Key-Value Pairs - Small", "[deviceradix][sort][pairs][small]") {

	initialize_backend_once();
	Resource device = single_resource;
	const uint32_t size = 1024;

	SECTION("Sort 1K elements") {

		std::vector<uint32_t> h_keys(size);
		std::vector<uint32_t> h_payloads(size);
		generate_random_data_drs(device, h_keys, 12345);
		std::iota(h_payloads.begin(), h_payloads.end(), 0);

		DeviceBuffer<uint32_t> d_keys(size, device.id());
		DeviceBuffer<uint32_t> d_payloads(size, device.id());
		d_keys.copy_from_host(h_keys.data(), size);
		d_payloads.copy_from_host(h_payloads.data(), size);

		DeviceBuffer<uint32_t> d_alt_keys(size, device.id());
		DeviceBuffer<uint32_t> d_alt_payloads(size, device.id());
#ifdef USE_CUDA
		device_radix_sort_pairs_cub(device.id(),
									d_keys.data(),
									d_payloads.data(),
									d_alt_keys.data(),
									d_alt_payloads.data(),
									size);
#elif defined(USE_SYCL)
		DeviceBuffer<uint32_t> d_globalHistogram(DRS_RADIX * 4, device.id());
		const uint32_t threadBlocks = (size + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
		DeviceBuffer<uint32_t> d_passHistogram(DRS_RADIX * threadBlocks, device.id());
		d_globalHistogram.fill(0, true);
		d_passHistogram.fill(0, true);
		device_radix_sort_pairs_usm(device,
									d_keys.data(),
									d_payloads.data(),
									d_alt_keys.data(),
									d_alt_payloads.data(),
									d_globalHistogram.data(),
									d_passHistogram.data(),
									size);
#else
		REQUIRE(false && "No backend available for radix sort test");
#endif

		std::vector<uint32_t> h_sorted_keys(size);
		std::vector<uint32_t> h_sorted_payloads(size);
		d_keys.copy_to_host(h_sorted_keys.data(), size);
		d_payloads.copy_to_host(h_sorted_payloads.data(), size);

		// Verify keys are sorted
		for (uint32_t i = 1; i < size; ++i) {
			REQUIRE(h_sorted_keys[i - 1] <= h_sorted_keys[i]);
		}

		// Verify stability: payloads should maintain relative order for equal keys
		std::vector<std::pair<uint32_t, uint32_t>> original(size);
		for (uint32_t i = 0; i < size; ++i) {
			original[i] = {h_keys[i], h_payloads[i]};
		}
		std::stable_sort(original.begin(), original.end(), [](const auto& a, const auto& b) {
			return a.first < b.first;
		});

		for (uint32_t i = 0; i < size; ++i) {
			REQUIRE(h_sorted_keys[i] == original[i].first);
			REQUIRE(h_sorted_payloads[i] == original[i].second);
		}
	}
}

TEST_CASE("DeviceRadixSort Key-Value Pairs", "[deviceradix][sort][pairs][medium]") {
	initialize_backend_once();
	Resource device = single_resource;
	const size_t size = 1024 * 1024 * 1024; // 1G elements

	SECTION("Sort 1G elements") {
		std::vector<uint32_t> h_keys(size);
		std::vector<uint32_t> h_payloads(size);
		generate_random_data_drs(device, h_keys, 54321);
		std::iota(h_payloads.begin(), h_payloads.end(), 0);

		DeviceBuffer<uint32_t> d_keys(size, device.id());
		DeviceBuffer<uint32_t> d_payloads(size, device.id());
		d_keys.copy_from_host(h_keys.data(), size);
		d_payloads.copy_from_host(h_payloads.data(), size);

		DeviceBuffer<uint32_t> d_alt_keys(size, device.id());
		DeviceBuffer<uint32_t> d_alt_payloads(size, device.id());
#if defined(USE_SYCL) && !defined(USE_CUDA)
		DeviceBuffer<uint32_t> d_globalHistogram(DRS_RADIX * 4, device.id());
		const uint32_t threadBlocks = (size + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
		DeviceBuffer<uint32_t> d_passHistogram(DRS_RADIX * threadBlocks, device.id());
#endif
		std::vector<long long> samples;
		for (int rep = 0; rep < MARS_SORT_BENCH_REPS + 1; ++rep) {
			d_keys.copy_from_host(h_keys.data(), size);
			d_payloads.copy_from_host(h_payloads.data(), size);
			auto start = std::chrono::high_resolution_clock::now();
#ifdef USE_CUDA
			device_radix_sort_pairs_cub(device.id(),
										d_keys.data(),
										d_payloads.data(),
										d_alt_keys.data(),
										d_alt_payloads.data(),
										size);
#elif defined(USE_SYCL)
			d_globalHistogram.fill(0, true);
			d_passHistogram.fill(0, true);
			device_radix_sort_pairs_usm(device,
										d_keys.data(),
										d_payloads.data(),
										d_alt_keys.data(),
										d_alt_payloads.data(),
										d_globalHistogram.data(),
										d_passHistogram.data(),
										size);
#else
			REQUIRE(false && "No backend available for radix sort test");
#endif
			auto end = std::chrono::high_resolution_clock::now();
			if (rep > 0) {
				samples.push_back(
					std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
			}
		}

#ifdef USE_SYCL
		report_sort_throughput(device, "device_radix_sort_pairs_usm", size, samples, 4);
#else
		std::cout << "Sorted " << size / (1024 * 1024 * 1024) << "G elements, median "
				  << samples[samples.size() / 2] << " ms on Device " << device.id() << std::endl;
#endif

		std::vector<uint32_t> h_sorted_keys(size);
		d_keys.copy_to_host(h_sorted_keys.data(), size);

		// Verify keys are sorted
		for (uint32_t i = 1; i < size; ++i) {
			REQUIRE(h_sorted_keys[i - 1] <= h_sorted_keys[i]);
		}
	}
}

#ifdef MARS_HAVE_ONEDPL_SORT
TEST_CASE("oneapi sort Key-Value Pairs", "[deviceradix][sort][pairs][intel]") {
	initialize_backend_once();
	Resource device = single_resource;
	const size_t size = 1024 * 1024 * 1024; // 1G elements

	SECTION("Sort 1G elements") {
		std::vector<uint32_t> h_keys(size);
		std::vector<uint32_t> h_payloads(size);
		generate_random_data_drs(device, h_keys, 54321);
		std::iota(h_payloads.begin(), h_payloads.end(), 0);

		DeviceBuffer<uint32_t> d_keys(size, device.id());
		DeviceBuffer<uint32_t> d_payloads(size, device.id());
		d_keys.copy_from_host(h_keys.data(), size);
		d_payloads.copy_from_host(h_payloads.data(), size);

		std::vector<long long> samples;
		for (int rep = 0; rep < MARS_SORT_BENCH_REPS + 1; ++rep) {
			d_keys.copy_from_host(h_keys.data(), size);
			d_payloads.copy_from_host(h_payloads.data(), size);
			auto start = std::chrono::high_resolution_clock::now();

			sort_morton_codes_oneapi(device, d_keys.data(), d_payloads.data(), size);

			auto end = std::chrono::high_resolution_clock::now();
			if (rep > 0) {
				samples.push_back(
					std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
			}
		}

		report_sort_throughput(device, "oneDPL stable_sort_by_key", size, samples, 8);

		std::vector<uint32_t> h_sorted_keys(size);
		d_keys.copy_to_host(h_sorted_keys.data(), size);

		// Verify keys are sorted
		for (uint32_t i = 1; i < size; ++i) {
			REQUIRE(h_sorted_keys[i - 1] <= h_sorted_keys[i]);
		}
	}
}
#endif