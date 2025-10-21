#include "System/ZOrderDecomposer.h"
#include "SimSystem.h"
#include "System/PatchManager.h"
#include "ARBDException.h"
#include "ARBDLogger.h"

#include <chrono>
#include <algorithm>

namespace ARBD {

ZOrderDecomposer::ZOrderDecomposer() : PatchDecomposer() {
    type_ = DecomposerType::ZOrder;

    // Set default configuration
    config_ = Config{};

    LOGINFO("Created ZOrderDecomposer with default configuration");
}

void ZOrderDecomposer::decompose(SimSystem& sys, const ResourceCollection& resources) {
    auto start_time = std::chrono::high_resolution_clock::now();

    LOGINFO("Starting Z-order decomposition for {} resources", resources.size());

    if (resources.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "Cannot decompose with zero resources");
    }

    // Step 1: Collect all particle positions from existing patches
    size_t total_particles = 0;
    collect_global_positions(sys, total_particles);

    if (total_particles == 0) {
        LOGWARN("No particles found for decomposition");
        return;
    }

    LOGINFO("Collected {} particles for decomposition", total_particles);

    // Step 2: Compute global bounding box
    Vector3 global_box_min, global_box_max;
    if (config_.auto_bounding_box) {
        compute_global_bounding_box(*global_positions_, total_particles,
                                   global_box_min, global_box_max);
        LOGTRACE("Computed global bounding box: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}]",
                global_box_min.x, global_box_min.y, global_box_min.z,
                global_box_max.x, global_box_max.y, global_box_max.z);
    } else {
        global_box_min = config_.manual_box_min;
        global_box_max = config_.manual_box_max;
        LOGTRACE("Using manual bounding box: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}]",
                global_box_min.x, global_box_min.y, global_box_min.z,
                global_box_max.x, global_box_max.y, global_box_max.z);
    }

    // Step 3: Sort particles globally by Morton code
    global_sorter_->sort_particles(*global_positions_, total_particles,
                                  global_box_min, global_box_max);

    LOGTRACE("Sorted {} particles by Morton code", total_particles);

    // Step 4: Determine optimal patch boundaries
    auto patch_boundaries = compute_patch_boundaries(total_particles, resources.size());

    // Step 5: Redistribute particles to patches
    redistribute_particles(sys, resources, patch_boundaries);

    // Step 6: Update patch metadata
    update_patch_metadata(sys, resources);

    // Step 7: Validate and compute statistics
    validate_and_compute_stats(sys, resources.size());

    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.decomposition_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    stats_.global_box_min = global_box_min;
    stats_.global_box_max = global_box_max;

    LOGINFO("Z-order decomposition completed in {:.2f} ms, load imbalance factor: {:.3f}",
            stats_.decomposition_time_ms, stats_.load_imbalance_factor);
}

void ZOrderDecomposer::collect_global_positions(SimSystem& sys, size_t& total_particles) {
    // Get patch manager to access current patches
    auto& patch_manager = sys.get_patch_manager();

    // Count total particles across all patches
    total_particles = 0;
    for (const auto& patch : patch_manager.get_patches()) {
        total_particles += patch.get_num_particles();
    }

    if (total_particles == 0) {
        return;
    }

    // Initialize global buffers if needed
    if (!global_positions_ || global_positions_->size() < total_particles) {
        // Use the first available resource for global operations
        const auto& first_resource = sys.get_resource_collection()[0];

        global_positions_ = std::make_unique<DeviceBuffer<Vector3>>(total_particles, first_resource);
        global_morton_codes_ = std::make_unique<DeviceBuffer<MortonCode::morton_t>>(total_particles, first_resource);
        global_indices_ = std::make_unique<DeviceBuffer<uint32_t>>(total_particles, first_resource);
        global_sorter_ = std::make_unique<ZOrderSort>(first_resource, total_particles);
    }

    // Copy positions from all patches to global buffer
    size_t offset = 0;
    for (const auto& patch : patch_manager.get_patches()) {
        const auto& positions = patch.get_positions();
        size_t num_particles = patch.get_num_particles();

        if (num_particles > 0) {
            // Copy from patch to global buffer
            // Note: This is a simplified implementation
            // Production code would handle cross-device transfers more efficiently
            positions.copy_to_buffer(*global_positions_, num_particles, 0, offset);
            offset += num_particles;
        }
    }
}

void ZOrderDecomposer::compute_global_bounding_box(const DeviceBuffer<Vector3>& positions,
                                                  size_t num_particles,
                                                  Vector3& box_min,
                                                  Vector3& box_max) {
    // Initialize bounds
    box_min = Vector3(std::numeric_limits<float>::max());
    box_max = Vector3(std::numeric_limits<float>::lowest());

    // Use the same resource as the positions buffer
    DeviceBuffer<Vector3> temp_min(1, positions.get_resource());
    DeviceBuffer<Vector3> temp_max(1, positions.get_resource());

    temp_min.copy_from_host(&box_min, 1);
    temp_max.copy_from_host(&box_max, 1);

    BoundingBoxKernel kernel{
        positions.data(),
        temp_min.data(),
        temp_max.data(),
        num_particles
    };

    launch_kernel(positions.get_resource(), kernel, num_particles);

    temp_min.copy_to_host(&box_min, 1, true);
    temp_max.copy_to_host(&box_max, 1, true);

    // Add small margin to avoid boundary issues
    Vector3 margin = (box_max - box_min) * 0.001f;
    box_min -= margin;
    box_max += margin;
}

std::vector<std::pair<MortonCode::morton_t, MortonCode::morton_t>>
ZOrderDecomposer::compute_patch_boundaries(size_t num_particles, size_t num_patches) {
    std::vector<std::pair<MortonCode::morton_t, MortonCode::morton_t>> boundaries;
    boundaries.reserve(num_patches);

    if (num_patches == 1) {
        // Special case: all particles go to single patch
        boundaries.emplace_back(0, std::numeric_limits<MortonCode::morton_t>::max());
        return boundaries;
    }

    // Get sorted Morton codes
    const auto& morton_codes = global_sorter_->get_morton_codes();

    // Simple equal-size partitioning
    // More sophisticated approaches could consider load balancing
    size_t particles_per_patch = num_particles / num_patches;
    size_t remainder = num_particles % num_patches;

    size_t start_idx = 0;
    for (size_t patch_id = 0; patch_id < num_patches; ++patch_id) {
        size_t patch_size = particles_per_patch + (patch_id < remainder ? 1 : 0);

        if (patch_size < config_.min_particles_per_patch && num_patches > 1) {
            patch_size = config_.min_particles_per_patch;
        }

        size_t end_idx = std::min(start_idx + patch_size, num_particles);

        MortonCode::morton_t start_morton = 0;
        MortonCode::morton_t end_morton = std::numeric_limits<MortonCode::morton_t>::max();

        if (start_idx < num_particles) {
            // Get Morton code from device (simplified - production code would be more efficient)
            morton_codes.copy_to_host(&start_morton, 1, true, start_idx);
        }

        if (end_idx < num_particles) {
            morton_codes.copy_to_host(&end_morton, 1, true, end_idx);
        }

        boundaries.emplace_back(start_morton, end_morton);
        start_idx = end_idx;

        if (start_idx >= num_particles) {
            break;
        }
    }

    return boundaries;
}

void ZOrderDecomposer::redistribute_particles(SimSystem& sys,
                                             const ResourceCollection& resources,
                                             const std::vector<std::pair<MortonCode::morton_t, MortonCode::morton_t>>& patch_boundaries) {
    // This is a high-level outline of particle redistribution
    // Production implementation would need to:
    // 1. Determine which particles belong to which patches based on Morton ranges
    // 2. Perform efficient data transfers between devices
    // 3. Update patch particle counts and metadata
    // 4. Handle inter-device communication for halo particles

    LOGINFO("Redistributing particles across {} patches", patch_boundaries.size());

    // For now, log the intended distribution
    for (size_t i = 0; i < patch_boundaries.size(); ++i) {
        LOGTRACE("Patch {}: Morton range [{:016x}, {:016x})",
                i, patch_boundaries[i].first, patch_boundaries[i].second);
    }

    // TODO: Implement actual particle redistribution
    LOGWARN("Particle redistribution not yet implemented - placeholder only");
}

void ZOrderDecomposer::update_patch_metadata(SimSystem& sys, const ResourceCollection& resources) {
    // Update patch metadata after redistribution
    // This includes spatial bounds, neighbor information, etc.

    auto& patch_manager = sys.get_patch_manager();

    // TODO: Update patch metadata based on new particle distribution
    LOGTRACE("Updated patch metadata for {} patches", resources.size());
}

void ZOrderDecomposer::validate_and_compute_stats(const SimSystem& sys, size_t num_patches) {
    stats_.total_particles = 0;
    stats_.num_patches = num_patches;
    stats_.particles_per_patch.clear();
    stats_.particles_per_patch.reserve(num_patches);

    // Collect particle counts from patches
    const auto& patch_manager = sys.get_patch_manager();
    for (const auto& patch : patch_manager.get_patches()) {
        size_t count = patch.get_num_particles();
        stats_.particles_per_patch.push_back(count);
        stats_.total_particles += count;
    }

    // Compute load imbalance factor
    if (!stats_.particles_per_patch.empty()) {
        auto max_load = *std::max_element(stats_.particles_per_patch.begin(),
                                         stats_.particles_per_patch.end());
        auto min_load = *std::min_element(stats_.particles_per_patch.begin(),
                                         stats_.particles_per_patch.end());

        double avg_load = static_cast<double>(stats_.total_particles) / num_patches;
        stats_.load_imbalance_factor = max_load / avg_load;

        LOGTRACE("Load distribution: min={}, max={}, avg={:.1f}, imbalance={:.3f}",
                min_load, max_load, avg_load, stats_.load_imbalance_factor);
    } else {
        stats_.load_imbalance_factor = 1.0f;
    }
}

} // namespace ARBD