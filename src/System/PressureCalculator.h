#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <memory>
#include <vector>

namespace ARBD {

/**
 * @brief Pressure calculation functionality for ARBD2
 *
 * This class provides pressure calculation capabilities including:
 * - Block-based pressure calculation for GPU efficiency
 * - Pressure output and logging
 * - Integration with the simulation system
 */
class PressureCalculator {
  public:
	/**
	 * @brief Constructor
	 * @param num_particles Number of particles in the system
	 * @param num_replicas Number of replicas
	 * @param resources Backend resources for GPU operations
	 */
	PressureCalculator(int num_particles, int num_replicas, std::shared_ptr<Resource> resources)
		: num_particles_(num_particles), num_replicas_(num_replicas), resources_(resources),
		  pressure_output_period_(100.0f) {

		// Calculate number of blocks needed
		const int NUM_THREADS = 32; // Standard thread block size
		num_blocks_ = (num_particles * num_replicas) / NUM_THREADS +
					  ((num_particles * num_replicas) % NUM_THREADS == 0 ? 0 : 1);

		// Allocate GPU memory for pressure calculation
		allocateGPUMemory(resources_->id());

		LOGINFO("PressureCalculator: Initialized for %d particles, %d replicas, %d blocks",
				num_particles_,
				num_replicas_,
				num_blocks_);
	}

	/**
	 * @brief Destructor
	 */
	~PressureCalculator() = default;

	// Non-copyable but movable
	PressureCalculator(const PressureCalculator&) = delete;
	PressureCalculator& operator=(const PressureCalculator&) = delete;

	PressureCalculator(PressureCalculator&&) = default;
	PressureCalculator& operator=(PressureCalculator&&) = default;

	/**
	 * @brief Calculate pressure for current system state
	 * @param positions Current particle positions
	 * @param forces Current forces on particles
	 * @param system_box System box dimensions
	 * @param timestep Current timestep
	 * @return Calculated pressure value
	 */
	float calculatePressure(const Vector3* positions,
							const Vector3* forces,
							const PeriodicBox& system_box,
							int timestep) {
		if (!enabled_)
			return 0.0f;

		// Launch pressure calculation kernel
		launchPressureKernel(positions, forces, system_box, resources_->id());

		// Copy results back to host
		copyPressureResults();

		// Calculate total pressure
		float total_pressure = 0.0f;
		for (int i = 0; i < num_blocks_; ++i) {
			total_pressure += block_pressure_[i];
		}

		// Output pressure if needed
		if (timestep % static_cast<int>(pressure_output_period_) == 0) {
			LOGINFO("Pressure at timestep %d: %f", timestep, total_pressure);
		}

		return total_pressure;
	}

	/**
	 * @brief Enable or disable pressure calculation
	 * @param enable Whether to enable pressure calculation
	 */
	void setEnabled(bool enable) {
		enabled_ = enable;
	}

	/**
	 * @brief Check if pressure calculation is enabled
	 * @return true if enabled
	 */
	bool isEnabled() const {
		return enabled_;
	}

	/**
	 * @brief Set pressure output period
	 * @param period Number of timesteps between pressure outputs
	 */
	void setOutputPeriod(float period) {
		pressure_output_period_ = period;
	}

	/**
	 * @brief Get pressure output period
	 * @return Output period in timesteps
	 */
	float getOutputPeriod() const {
		return pressure_output_period_;
	}

  private:
	int num_particles_;
	int num_replicas_;
	int num_blocks_;
	std::shared_ptr<Resource> resources_;
	bool enabled_{false};
	float pressure_output_period_;

	// GPU memory for pressure calculation
	std::unique_ptr<DeviceBuffer<arbd_real>> block_pressure_d_;
	std::vector<float> block_pressure_;

	/**
	 * @brief Allocate GPU memory for pressure calculation
	 */
	void allocateGPUMemory(short resource_id) {
		// Allocate GPU memory for block pressure results
		block_pressure_d_ = std::make_unique<DeviceBuffer<arbd_real>>(num_blocks_, resource_id);
	};
	/**
	 * @brief Launch pressure calculation kernel
	 * @param positions Current particle positions
	 * @param forces Current forces on particles
	 * @param system_box System box dimensions
	 */
	void launchPressureKernel(const Vector3* positions,
							  const Vector3* forces,
							  const PeriodicBox& system_box,
							  short resource_id) {
		LOGDEBUG("PressureCalculator: Launching pressure kernel");
		// TODO: Implement actual kernel launch based on backend
		// This would involve calling the appropriate compute function
		// from the backend (CUDA, SYCL, or Metal)
		// block_pressure_d_->launch_kernel(resource_id, num_blocks_, [this](int i) {
		// 	block_pressure_[i] = 0.0f;
		// });
	}

	/**
	 * @brief Copy pressure results from GPU to host
	 */
	void copyPressureResults() {
		try {
			// Copy results from GPU to host
			block_pressure_d_->copy_to_host(block_pressure_.data(), num_blocks_);
		} catch (const std::exception& e) {
			ARBD_Exception(ExceptionType::RuntimeError,
						   "Failed to copy pressure results from GPU: {}",
						   e.what());
		}
	};

}; // namespace ARBD
} // namespace ARBD
