#ifdef USE_SYCL
#include "SYCLManager.h"
#include "ARBDLogger.h"
#include <algorithm>
#include <span>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace ARBD {
namespace SYCL {
// Static member initialization
std::vector<SYCLManager::Device> SYCLManager::all_devices_;
std::vector<SYCLManager::Device> SYCLManager::devices_;
int SYCLManager::current_device_{0};
sycl::info::device_type SYCLManager::preferred_type_{sycl::info::device_type::cpu};

// Device class implementation
SYCLManager::Device::Device(const sycl::device& dev, unsigned int id)
	: id_(id), device_(dev), queues_(create_queues(dev, id)) {

	// Query device properties after construction
	query_device_properties();

	LOGDEBUG("Device {} initialized: {} ({})", id_, name_.c_str(), vendor_.c_str());
	LOGDEBUG("  Compute units: {}, Global memory: {:.1f}GB, Max work group: {}",
			max_compute_units_,
			static_cast<float>(global_mem_size_) / (1024.0f * 1024.0f * 1024.0f),
			max_work_group_size_);
}

// Helper function to create all queues for a device with proper RAII
std::array<ARBD::SYCL::Queue, SYCLManager::NUM_QUEUES>
SYCLManager::Device::create_queues(const sycl::device& dev, unsigned int id) {
	try {
		// Test if device can create a basic queue first with explicit single-device context
		sycl::queue test_queue(sycl::context({dev}), dev);

		// If successful, create all our wrapped queues
		// Note: We need to construct each queue individually since Queue() is deleted
		return std::array<Queue, NUM_QUEUES>{{Queue(dev),
											  Queue(dev),
											  Queue(dev),
											  Queue(dev),
											  Queue(dev),
											  Queue(dev),
											  Queue(dev),
											  Queue(dev)}};

	} catch (const sycl::exception& e) {
		LOGERROR("SYCL exception creating queues for device {}: {}", id, e.what());

		// Try fallback with empty properties
		try {
			sycl::property_list empty_props;
			return std::array<Queue, NUM_QUEUES>{{Queue(dev, empty_props),
												  Queue(dev, empty_props),
												  Queue(dev, empty_props),
												  Queue(dev, empty_props),
												  Queue(dev, empty_props),
												  Queue(dev, empty_props),
												  Queue(dev, empty_props),
												  Queue(dev, empty_props)}};
		} catch (const sycl::exception& e2) {
			LOGERROR("Failed to create fallback queues for device {}: {}", id, e2.what());
			throw; // Re-throw if we can't create any queues
		}
	}
}

void SYCLManager::Device::query_device_properties() {
	try {
		// Set default values first
		name_ = "Unknown Device";
		vendor_ = "Unknown Vendor";
		version_ = "Unknown Version";
		max_work_group_size_ = 1;
		max_compute_units_ = 1;
		global_mem_size_ = 0;
		local_mem_size_ = 0;
		is_cpu_ = false;
		is_gpu_ = false;
		is_accelerator_ = false;

		// Try to get device type first (most critical)
		try {
			auto device_type = device_.get_info<sycl::info::device::device_type>();
			is_cpu_ = (device_type == sycl::info::device_type::cpu);
			is_gpu_ = (device_type == sycl::info::device_type::gpu);
			is_accelerator_ = (device_type == sycl::info::device_type::accelerator);
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query device type for device {}: {}", id_, e.what());
		}

		// Try to get basic device info
		try {
			name_ = device_.get_info<sycl::info::device::name>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query device name for device {}: {}", id_, e.what());
		}

		try {
			vendor_ = device_.get_info<sycl::info::device::vendor>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query device vendor for device {}: {}", id_, e.what());
		}

		try {
			version_ = device_.get_info<sycl::info::device::version>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query device version for device {}: {}", id_, e.what());
		}

		// Try to get performance characteristics
		try {
			max_work_group_size_ = device_.get_info<sycl::info::device::max_work_group_size>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query max work group size for device {}: {}", id_, e.what());
		}

		try {
			max_compute_units_ = device_.get_info<sycl::info::device::max_compute_units>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query max compute units for device {}: {}", id_, e.what());
		}

		try {
			global_mem_size_ = device_.get_info<sycl::info::device::global_mem_size>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query global memory size for device {}: {}", id_, e.what());
		}

		try {
			local_mem_size_ = device_.get_info<sycl::info::device::local_mem_size>();
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query local memory size for device {}: {}", id_, e.what());
		}

	} catch (const sycl::exception& e) {
		std::cerr << "!!! query_device_properties caught sycl::exception: " << e.what()
				  << " for device (name might be uninit): " << name_ << std::endl;
		LOGERROR("Critical error querying device properties for device {}: {}", id_, e.what());
		// Keep default values set above
	}
}

void SYCLManager::Device::synchronize_all_queues() {
	for (auto& queue : queues_) {
		try {
			queue.synchronize();
		} catch (const sycl::exception& e) {
			// Log SYCL errors but continue with other queues
			std::cerr << "Warning: SYCL error during queue synchronization: " << e.what()
					  << std::endl;
		} catch (const std::exception& e) {
			// Log other errors but continue with other queues
			std::cerr << "Warning: Error during queue synchronization: " << e.what() << std::endl;
		}
	}
}

// SYCLManager static methods implementation
void SYCLManager::init() {
	LOGDEBUG("Initializing SYCL Manager...");

	all_devices_.clear();
	devices_.clear();
	current_device_ = 0;

	discover_devices();

	if (all_devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No SYCL devices found");
	}

	LOGINFO("Found {} SYCL device(s)", all_devices_.size());
}

void SYCLManager::discover_devices() {
	try {
		// Get all platforms
		auto platforms = sycl::platform::get_platforms();

		// First pass: collect valid devices information without constructing Device objects
		struct DeviceInfo {
			sycl::device device;
			unsigned int id;
			sycl::info::device_type type;
		};

		std::vector<DeviceInfo> potential_device_infos;
		unsigned int device_id = 0;

		for (const auto& platform : platforms) {
			LOGDEBUG("Platform: {} ({})",
					 platform.get_info<sycl::info::platform::name>().c_str(),
					 platform.get_info<sycl::info::platform::vendor>().c_str());

			// Get all devices for this platform
			auto platform_devices = platform.get_devices();

			for (const auto& device : platform_devices) {
				try {
					// Test sycl::device copy construction explicitly
					sycl::device temp_device_copy(device);
					auto dev_type = temp_device_copy.get_info<sycl::info::device::device_type>();
					LOGDEBUG("Successfully test-copied sycl::device: {}",
							 temp_device_copy.get_info<sycl::info::device::name>().c_str());

					// Store device info for later construction
					potential_device_infos.push_back(
						{std::move(temp_device_copy), device_id, dev_type});
					device_id++;

				} catch (const sycl::exception& e) {
					LOGWARN("SYCL exception during device discovery for device id {}: {}",
							device_id,
							e.what());
				} catch (const ARBD::Exception& e) {
					LOGWARN("ARBD::Exception during device discovery for device id {}: {}",
							device_id,
							e.what());
				} catch (const std::exception& e) {
					LOGWARN("Std::exception during device discovery for device id {}: {}",
							device_id,
							e.what());
				}
			}
		}

		// Filter devices based on preference
		std::vector<DeviceInfo> selected_device_infos;

		if (preferred_type_ != sycl::info::device_type::all) {
			// First, try to find devices matching the preferred type
			for (const auto& device_info : potential_device_infos) {
				if (device_info.type == preferred_type_) {
					selected_device_infos.push_back(device_info);
				}
			}

			// If no preferred devices found, fall back to any available devices
			if (selected_device_infos.empty()) {
				LOGWARN("No devices of preferred type {} found, using all available devices",
						static_cast<int>(preferred_type_));
				selected_device_infos = std::move(potential_device_infos);
			} else {
				LOGINFO("Found {} device(s) of preferred type {}",
						selected_device_infos.size(),
						static_cast<int>(preferred_type_));
			}
		} else {
			// Use all devices if preference is 'all'
			selected_device_infos = std::move(potential_device_infos);
		}

		// Sort device infos by preference
		std::stable_sort(selected_device_infos.begin(),
						 selected_device_infos.end(),
						 [](const DeviceInfo& a, const DeviceInfo& b) {
							 if (a.type != b.type) {
								 return a.type == SYCLManager::preferred_type_;
							 }
							 return a.id < b.id;
						 });

		// Now construct Device objects in place
		all_devices_.clear();
		all_devices_.reserve(selected_device_infos.size());

		for (size_t i = 0; i < selected_device_infos.size(); ++i) {
			const auto& device_info = selected_device_infos[i];
			all_devices_.emplace_back(device_info.device, static_cast<unsigned int>(i));
		}

	} catch (const sycl::exception& e) {
		check_sycl_error(e, __FILE__, __LINE__);
	}
}

void SYCLManager::load_info() {
	init();

	// For single device case (Mac), explicitly select only the first device
	// to avoid multi-device context issues
	if (all_devices_.size() == 1) {
		LOGINFO("Single device detected, selecting device 0 for single-device operation");
		select_devices(std::span<const unsigned int>{&all_devices_[0].id_, 1});
	} else {
		// Multi-device case - use all discovered devices
		// Copy all_devices_ to devices_ instead of moving to preserve all_devices_ for
		// select_devices()
		devices_.clear();
		devices_.reserve(all_devices_.size());
		for (size_t i = 0; i < all_devices_.size(); ++i) {
			devices_.emplace_back(all_devices_[i].get_device(), static_cast<unsigned int>(i));
		}
		init_devices();
	}
}

void SYCLManager::init_devices() {
	LOGINFO("Initializing SYCL devices...");
	std::string msg;

	for (size_t i = 0; i < devices_.size(); i++) {
		if (i > 0) {
			if (i == devices_.size() - 1) {
				msg += " and ";
			} else {
				msg += ", ";
			}
		}
		msg += std::to_string(devices_[i].id());

		// Devices are already initialized in constructor
		// Just log that they're ready
		LOGDEBUG("Device {} ready: {}", devices_[i].id(), devices_[i].name().c_str());
	}

	LOGINFO("Initialized SYCL devices: {}", msg.c_str());
	current_device_ = 0;
}

void SYCLManager::select_devices(std::span<const unsigned int> device_ids) {
	devices_.clear();
	devices_.reserve(device_ids.size()); // Reserve space to avoid reallocations

	for (unsigned int id : device_ids) {
		if (id >= all_devices_.size()) {
			ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}", id);
		}
		// Create a new Device by copying the sycl::device and id
		devices_.emplace_back(all_devices_[id].get_device(), id);
	}
	init_devices();
}

void SYCLManager::use(int device_id) {
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No devices selected");
	}
	current_device_ = device_id % static_cast<int>(devices_.size());
}

void SYCLManager::sync(int device_id) {
	if (device_id >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}", device_id);
	}
	devices_[device_id].synchronize_all_queues();
}

void SYCLManager::sync() {
	for (auto& device : devices_) {
		try {
			device.synchronize_all_queues();
		} catch (const std::exception& e) {
			// Log but continue with other devices
			std::cerr << "Warning: Error synchronizing device " << device.id() << ": " << e.what()
					  << std::endl;
		}
	}
}

void SYCLManager::finalize() {
	try {
		// First, synchronize all devices to ensure all operations complete
		if (!devices_.empty()) {
			sync();
		}

		// Clear current device reference first to prevent access during cleanup
		current_device_ = 0;

		// Clear devices explicitly which will call Device destructors
		// in a controlled manner
		devices_.clear();

		// Now clear all_devices_
		all_devices_.clear();

		// Reset to default state
		preferred_type_ = sycl::info::device_type::cpu;

	} catch (const std::exception& e) {
		// Log but don't throw during finalization to prevent cascading errors
		std::cerr << "Warning: Error during SYCL finalization: " << e.what() << std::endl;
	} catch (...) {
		// Catch any other exceptions during cleanup
		std::cerr << "Warning: Unknown error during SYCL finalization" << std::endl;
	}
}

int SYCLManager::current() {
	return current_device_;
}

void SYCLManager::prefer_device_type(sycl::info::device_type type) {
	preferred_type_ = type;

	if (!all_devices_.empty()) {
		std::sort(all_devices_.begin(),
				  all_devices_.end(),
				  [type](const Device& a, const Device& b) {
					  auto a_type = a.get_device().get_info<sycl::info::device::device_type>();
					  auto b_type = b.get_device().get_info<sycl::info::device::device_type>();

					  if ((a_type == type) != (b_type == type)) {
						  return a_type == type;
					  }
					  return a.id() < b.id();
				  });

		// Reassign IDs after sorting
		for (size_t i = 0; i < all_devices_.size(); ++i) {
			const_cast<unsigned int&>(all_devices_[i].id_) = static_cast<unsigned int>(i);
		}
	}
}

std::vector<unsigned int> SYCLManager::get_gpu_device_ids() {
	std::vector<unsigned int> gpu_ids;
	for (const auto& device : all_devices_) {
		if (device.is_gpu()) {
			gpu_ids.push_back(device.id());
		}
	}
	return gpu_ids;
}

std::vector<unsigned int> SYCLManager::get_cpu_device_ids() {
	std::vector<unsigned int> cpu_ids;
	for (const auto& device : all_devices_) {
		if (device.is_cpu()) {
			cpu_ids.push_back(device.id());
		}
	}
	return cpu_ids;
}

std::vector<unsigned int> SYCLManager::get_accelerator_device_ids() {
	std::vector<unsigned int> accel_ids;
	for (const auto& device : all_devices_) {
		if (device.is_accelerator()) {
			accel_ids.push_back(device.id());
		}
	}
	return accel_ids;
}

} // namespace SYCL
} // namespace ARBD

#endif // PROJECT_USES_SYCL
