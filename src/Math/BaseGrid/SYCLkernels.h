#pragma once

#include "../BaseGrid.h"

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#include "Backend/SYCL/SYCLManager.h"
#endif

#include "ARBDLogger.h"
#include "ARBDException.h"
#include "Backend/Resource.h"
#include "Backend/Events.h"
#include "Backend/Buffer.h"

namespace ARBD {
namespace BaseGridKernels {

#ifdef USE_SYCL

/*===============================*\
|       INTERPOLATION KERNELS    |
\===============================*/

// SYCL kernel for trilinear interpolation
template<BoundaryCondition BC>
struct TrilinearInterpolationKernel {
    const float* grid_data;
    int nx, ny, nz;
    Matrix3 basis_inv;
    Vector3 origin;
    Vector3 query_pos;
    float* result;
    
    void operator()(sycl::id<1> idx) const {
        // Convert world position to grid coordinates
        Vector3 local = basis_inv.transform(query_pos - origin);
        
        // Get integer grid indices and fractional parts
        int i0 = int(sycl::floor(local.x));
        int j0 = int(sycl::floor(local.y));
        int k0 = int(sycl::floor(local.z));
        
        float fx = local.x - float(i0);
        float fy = local.y - float(j0);
        float fz = local.z - float(k0);
        
        // Apply boundary conditions and get values
        auto get_value = [&](int i, int j, int k) -> float {
            Vector3_t<int> bounded_idx;
            
            if constexpr (BC == BoundaryCondition::periodic) {
                bounded_idx.x = ((i % nx) + nx) % nx;
                bounded_idx.y = ((j % ny) + ny) % ny;
                bounded_idx.z = ((k % nz) + nz) % nz;
            } else if constexpr (BC == BoundaryCondition::dirichlet) {
                bounded_idx.x = sycl::max(0, sycl::min(nx - 1, i));
                bounded_idx.y = sycl::max(0, sycl::min(ny - 1, j));
                bounded_idx.z = sycl::max(0, sycl::min(nz - 1, k));
            } else if constexpr (BC == BoundaryCondition::neumann) {
                bounded_idx.x = (i < 0) ? -i : ((i >= nx) ? 2 * nx - 1 - i : i);
                bounded_idx.y = (j < 0) ? -j : ((j >= ny) ? 2 * ny - 1 - j : j);
                bounded_idx.z = (k < 0) ? -k : ((k >= nz) ? 2 * nz - 1 - k : k);
            }
            
            size_t grid_idx = size_t(bounded_idx.x) * ny * nz + 
                             size_t(bounded_idx.y) * nz + 
                             size_t(bounded_idx.z);
            return grid_data[grid_idx];
        };
        
        // Get the 8 surrounding grid point values
        float v000 = get_value(i0, j0, k0);
        float v001 = get_value(i0, j0, k0 + 1);
        float v010 = get_value(i0, j0 + 1, k0);
        float v011 = get_value(i0, j0 + 1, k0 + 1);
        float v100 = get_value(i0 + 1, j0, k0);
        float v101 = get_value(i0 + 1, j0, k0 + 1);
        float v110 = get_value(i0 + 1, j0 + 1, k0);
        float v111 = get_value(i0 + 1, j0 + 1, k0 + 1);
        
        // Trilinear interpolation
        float v00 = v000 * (1.0f - fx) + v100 * fx;
        float v01 = v001 * (1.0f - fx) + v101 * fx;
        float v10 = v010 * (1.0f - fx) + v110 * fx;
        float v11 = v011 * (1.0f - fx) + v111 * fx;
        
        float v0 = v00 * (1.0f - fy) + v10 * fy;
        float v1 = v01 * (1.0f - fy) + v11 * fy;
        
        *result = v0 * (1.0f - fz) + v1 * fz;
    }
};

// SYCL kernel for computing gradient (force)
template<BoundaryCondition BC>
struct GradientKernel {
    const float* grid_data;
    int nx, ny, nz;
    Matrix3 basis_inv;
    Matrix3 basis;
    Vector3 origin;
    Vector3 query_pos;
    Vector3* result;
    
    void operator()(sycl::id<1> idx) const {
        // Convert world position to grid coordinates
        Vector3 local = basis_inv.transform(query_pos - origin);
        
        // Get integer grid indices
        int i0 = int(sycl::round(local.x));
        int j0 = int(sycl::round(local.y));
        int k0 = int(sycl::round(local.z));
        
        auto get_value = [&](int i, int j, int k) -> float {
            Vector3_t<int> bounded_idx;
            
            if constexpr (BC == BoundaryCondition::periodic) {
                bounded_idx.x = ((i % nx) + nx) % nx;
                bounded_idx.y = ((j % ny) + ny) % ny;
                bounded_idx.z = ((k % nz) + nz) % nz;
            } else if constexpr (BC == BoundaryCondition::dirichlet) {
                bounded_idx.x = sycl::max(0, sycl::min(nx - 1, i));
                bounded_idx.y = sycl::max(0, sycl::min(ny - 1, j));
                bounded_idx.z = sycl::max(0, sycl::min(nz - 1, k));
            } else if constexpr (BC == BoundaryCondition::neumann) {
                bounded_idx.x = (i < 0) ? -i : ((i >= nx) ? 2 * nx - 1 - i : i);
                bounded_idx.y = (j < 0) ? -j : ((j >= ny) ? 2 * ny - 1 - j : j);
                bounded_idx.z = (k < 0) ? -k : ((k >= nz) ? 2 * nz - 1 - k : k);
            }
            
            size_t grid_idx = size_t(bounded_idx.x) * ny * nz + 
                             size_t(bounded_idx.y) * nz + 
                             size_t(bounded_idx.z);
            return grid_data[grid_idx];
        };
        
        // Compute finite differences for gradient
        float dx_plus = get_value(i0 + 1, j0, k0);
        float dx_minus = get_value(i0 - 1, j0, k0);
        float dy_plus = get_value(i0, j0 + 1, k0);
        float dy_minus = get_value(i0, j0 - 1, k0);
        float dz_plus = get_value(i0, j0, k0 + 1);
        float dz_minus = get_value(i0, j0, k0 - 1);
        
        // Central differences in grid coordinates
        Vector3 grad_local;
        grad_local.x = (dx_plus - dx_minus) * 0.5f;
        grad_local.y = (dy_plus - dy_minus) * 0.5f;
        grad_local.z = (dz_plus - dz_minus) * 0.5f;
        
        // Transform gradient to world coordinates and negate for force
        Vector3 grad_world = basis_inv.transpose().transform(grad_local);
        *result = -grad_world;
    }
};

/*===============================*\
|       BULK OPERATION KERNELS   |
\===============================*/

// Kernel for adding scalar to all grid points
struct AddScalarKernel {
    float* data;
    float value;
    
    void operator()(sycl::id<1> idx) const {
        data[idx] += value;
    }
};

// Kernel for multiplying all grid points by scalar
struct MultiplyScalarKernel {
    float* data;
    float value;
    
    void operator()(sycl::id<1> idx) const {
        data[idx] *= value;
    }
};

// Kernel for element-wise addition of two grids
struct AddGridKernel {
    float* data;
    const float* other_data;
    
    void operator()(sycl::id<1> idx) const {
        data[idx] += other_data[idx];
    }
};

// Kernel for element-wise multiplication of two grids
struct MultiplyGridKernel {
    float* data;
    const float* other_data;
    
    void operator()(sycl::id<1> idx) const {
        data[idx] *= other_data[idx];
    }
};

/*===============================*\
|       UTILITY FUNCTIONS        |
\===============================*/

// Helper function to launch trilinear interpolation
template<BoundaryCondition BC>
BACKEND::Event launch_trilinear_interpolation(const BaseGrid& grid, 
                                             const Vector3& pos, 
                                             float& result,
                                             const Resource& resource) {
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue();
    
    // Allocate device memory for result
    float* d_result = sycl::malloc_device<float>(1, queue.get());
    
    // Launch kernel
    auto event = queue.get().parallel_for(sycl::range<1>(1),
        TrilinearInterpolationKernel<BC>{
            grid.data(),
            grid.nx(),
            grid.ny(), 
            grid.nz(),
            grid.basisInv(),
            grid.origin(),
            pos,
            d_result
        });
    
    // Copy result back to host
    queue.get().copy(d_result, &result, 1).wait();
    sycl::free(d_result, queue.get());
    
    return BACKEND::Event(event, resource);
}

// Helper function to launch gradient computation
template<BoundaryCondition BC>
BACKEND::Event launch_gradient_computation(const BaseGrid& grid, 
                                          const Vector3& pos, 
                                          Vector3& result,
                                          const Resource& resource) {
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue();
    
    Vector3* d_result = sycl::malloc_device<Vector3>(1, queue.get());
    
    auto event = queue.get().parallel_for(sycl::range<1>(1),
        GradientKernel<BC>{
            grid.data(),
            grid.nx(),
            grid.ny(),
            grid.nz(),
            grid.basisInv(),
            grid.basis(),
            grid.origin(),
            pos,
            d_result
        });
    
    queue.get().copy(d_result, &result, 1).wait();
    sycl::free(d_result, queue.get());
    
    return BACKEND::Event(event, resource);
}

#endif // USE_SYCL

} // namespace BaseGridKernels
} // namespace ARBD