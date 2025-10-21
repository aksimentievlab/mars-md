#pragma once

/*********************************************************************
 * @file  ZOrderDecomposer.h
 *
 * @brief Z-order based patch decomposition for load balancing
 *
 * This class implements patch decomposition using Morton codes to
 * achieve natural load balancing across GPU patches while maintaining
 * spatial locality within each patch.
 *********************************************************************/

#include "System/Decomposer.h"
#include "System/MortonCode.h"
#include "System/ZOrderSort.h"
#include "Backend/Buffer.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Z-order based patch decomposition
 *
 * Uses Morton codes to:
 * 1. Sort all particles globally by spatial location
 * 2. Distribute consecutive Morton ranges to GPU patches
 * 3. Achieve automatic load balancing while preserving locality
 *
 * This approach is particularly effective for:
 * - Non-uniform particle distributions
 * - Dynamic load balancing
 * - Multi-device scaling
 */
class ZOrderDecomposer : public PatchDecomposer {
public:
    /**
     * @brief Constructor
     */
    ZOrderDecomposer();

    /**
     * @brief Perform Z-order based patch decomposition
     * @param sys The global system containing particle data
     * @param resources Collection of hardware resources (GPUs)
     */
    void decompose(SimSystem& sys, const ResourceCollection& resources) override;

    /**
     * @brief Get decomposer name
     */
    const std::string get_name() override {
        return "Z-Order Decomposer";
    }

    /**
     * @brief Configuration for Z-order decomposition
     */
    struct Config {
        bool auto_bounding_box = true;      ///< Compute bounding box automatically
        Vector3 manual_box_min = Vector3(0.0f); ///< Manual box minimum (if auto=false)
        Vector3 manual_box_max = Vector3(1.0f); ///< Manual box maximum (if auto=false)

        float load_balance_tolerance = 0.05f; ///< Tolerance for load imbalance (5%)
        bool enable_load_balancing = true;   ///< Enable dynamic load balancing

        size_t min_particles_per_patch = 100; ///< Minimum particles per patch
        bool preserve_patch_boundaries = false; ///< Keep existing patch boundaries
    };

    /**
     * @brief Set decomposer configuration
     */
    void set_config(const Config& config) { config_ = config; }

    /**
     * @brief Get current configuration
     */
    const Config& get_config() const { return config_; }

    /**
     * @brief Get statistics from last decomposition
     */
    struct Statistics {
        size_t total_particles;
        size_t num_patches;
        std::vector<size_t> particles_per_patch;
        float load_imbalance_factor;  ///< max_load / avg_load
        double decomposition_time_ms;
        Vector3 global_box_min;
        Vector3 global_box_max;
    };

    /**
     * @brief Get decomposition statistics
     */
    const Statistics& get_statistics() const { return stats_; }

private:
    Config config_;
    mutable Statistics stats_;

    // Temporary buffers for global operations
    std::unique_ptr<DeviceBuffer<Vector3>> global_positions_;
    std::unique_ptr<DeviceBuffer<MortonCode::morton_t>> global_morton_codes_;
    std::unique_ptr<DeviceBuffer<uint32_t>> global_indices_;
    std::unique_ptr<ZOrderSort> global_sorter_;

    /**
     * @brief Collect all particle positions from patches
     * @param sys System containing patch data
     * @param total_particles Output: total number of particles
     * @return Device buffer with all positions
     */
    void collect_global_positions(SimSystem& sys, size_t& total_particles);

    /**
     * @brief Compute global bounding box
     * @param positions All particle positions
     * @param num_particles Total number of particles
     * @param box_min Output: minimum bounds
     * @param box_max Output: maximum bounds
     */
    void compute_global_bounding_box(const DeviceBuffer<Vector3>& positions,
                                   size_t num_particles,
                                   Vector3& box_min,
                                   Vector3& box_max);

    /**
     * @brief Determine optimal patch boundaries based on Morton codes
     * @param num_particles Total number of particles
     * @param num_patches Number of patches (GPUs)
     * @return Vector of Morton code ranges for each patch
     */
    std::vector<std::pair<MortonCode::morton_t, MortonCode::morton_t>>
    compute_patch_boundaries(size_t num_particles, size_t num_patches);

    /**
     * @brief Redistribute particles to patches based on Morton ranges
     * @param sys System to update
     * @param resources Available compute resources
     * @param patch_boundaries Morton code ranges for each patch
     */
    void redistribute_particles(SimSystem& sys,
                              const ResourceCollection& resources,
                              const std::vector<std::pair<MortonCode::morton_t, MortonCode::morton_t>>& patch_boundaries);

    /**
     * @brief Update patch metadata after redistribution
     * @param sys System to update
     * @param resources Available compute resources
     */
    void update_patch_metadata(SimSystem& sys, const ResourceCollection& resources);

    /**
     * @brief Validate decomposition and compute statistics
     * @param sys System to validate
     * @param num_patches Number of patches
     */
    void validate_and_compute_stats(const SimSystem& sys, size_t num_patches);
};

} // namespace ARBD