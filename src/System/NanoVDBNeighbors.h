#pragma once

#include "Types/NanoGridHandle.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Types/Vector3.h"
#include <nanovdb/math/Stencils.h>

namespace ARBD {
namespace Interactions {

/**
 * @brief NanoVDB-based neighbor finding for sparse grid structures
 */
template<typename ValueT = float>
class NanoVDBNeighborList {
private:
    NanoGridAdapter<> grid_adapter_;
    float cutoff_distance_;
    
public:
    explicit NanoVDBNeighborList(NanoGridAdapter<>&& adapter, float cutoff)
        : grid_adapter_(std::move(adapter)), cutoff_distance_(cutoff) {}
    
    /**
     * @brief Find neighbors within cutoff distance using VDB tree structure
     */
    template<typename PositionBufferT, typename NeighborBufferT>
    Event find_neighbors(const DeviceBuffer<PositionBufferT>& positions,
                        DeviceBuffer<NeighborBufferT>& neighbor_lists,
                        const KernelConfig& config = KernelConfig{}) {
        
        const auto& resource = grid_adapter_.resource();
        size_t num_positions = positions.size();
        
        // Use NanoVDB's tree structure for efficient neighbor queries
        const auto* grid = grid_adapter_.grid<ValueT>();
        auto accessor = grid->getAccessor();
        
#ifdef USE_CUDA
        if (resource.type == ResourceType::CUDA) {
            return launch_cuda_kernel(
                resource,
                num_positions,
                std::make_tuple(grid_adapter_.buffer(), positions),
                std::forward_as_tuple(neighbor_lists),
                config,
                [cutoff = cutoff_distance_](size_t idx, 
                                          DEVICE_PTR(nanovdb::NanoGrid<ValueT>) grid,
                                          DEVICE_PTR(PositionBufferT) pos_ptr,
                                          DEVICE_PTR(NeighborBufferT) neighbors_ptr) {
                    
                    auto pos = pos_ptr[idx];
                    auto coord = grid->worldToIndex(pos);
                    
                    // Use BoxStencil for 8-neighbor connectivity or extend for larger regions
                    nanovdb::math::BoxStencil<nanovdb::NanoGrid<ValueT>> stencil(*grid);
                    stencil.moveTo(coord);
                    
                    // Find neighbors within cutoff using stencil traversal
                    // Implementation depends on specific neighbor structure needed
                    // neighbors_ptr[idx] = find_neighbors_in_region(stencil, pos, cutoff);
                }
            );
        }
#endif
        
        return find_neighbors_cpu(positions, neighbor_lists);
    }
    
    /**
     * @brief Get neighbor count at given positions
     */
    template<typename PositionBufferT>
    Event count_neighbors(const DeviceBuffer<PositionBufferT>& positions,
                         DeviceBuffer<int>& counts,
                         const KernelConfig& config = KernelConfig{}) {
        
        const auto& resource = grid_adapter_.resource();
        size_t num_positions = positions.size();
        counts.resize(num_positions, resource);
        
        // Implementation similar to find_neighbors but only counting
        return find_neighbors_cpu_count_only(positions, counts);
    }
    
private:
    template<typename PositionBufferT, typename NeighborBufferT>
    Event find_neighbors_cpu(const DeviceBuffer<PositionBufferT>& positions,
                             DeviceBuffer<NeighborBufferT>& neighbor_lists) {
        // CPU implementation using NanoVDB host-side access
        std::vector<PositionBufferT> host_pos(positions.size());
        positions.copy_to_host(host_pos);
        
        const auto* grid = grid_adapter_.grid<ValueT>();
        auto accessor = grid->getAccessor();
        
        std::vector<NeighborBufferT> host_neighbors(positions.size());
        
        for (size_t i = 0; i < host_pos.size(); ++i) {
            auto coord = grid->worldToIndex(host_pos[i]);
            nanovdb::math::BoxStencil<nanovdb::NanoGrid<ValueT>> stencil(*grid);
            stencil.moveTo(coord);
            
            // Find neighbors using stencil and geometric proximity
            host_neighbors[i] = compute_neighbors_for_position(stencil, host_pos[i]);
        }
        
        neighbor_lists.copy_from_host(host_neighbors);
        return Event{};
    }
    
    template<typename StencilT, typename PosT, typename NeighborT>
    NeighborT find_neighbors_in_region(const StencilT& stencil, const PosT& pos, float cutoff) {
        //todo: Implement neighbor finding logic using stencil values
        NeighborT neighbors;
        // ... implementation details ...
        return neighbors;
    }
    
    template<typename StencilT, typename PosT, typename NeighborT>
    NeighborT compute_neighbors_for_position(const StencilT& stencil, const PosT& pos) {
        // CPU version of neighbor computation
        NeighborT neighbors;
        // ... implementation details ...
        return neighbors;
    }
};

} // namespace Interactions
} // namespace ARBD