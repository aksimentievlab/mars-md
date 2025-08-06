#include "../BaseGrid.h"
#include "ARBDLogger.h"
#include "ARBDException.h"
#include "Backend/Resource.h"
#include "Math/Matrix3.h"
#include "Math/Vector3.h"

namespace ARBD {

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

/*===============================*\
|       PRIVATE METHODS          |
\===============================*/

void BaseGrid::init() {
    basisInv_ = basis_.inverse();
    size_ = size_t(nx_) * size_t(ny_) * size_t(nz_);
    
    // Allocate device buffer using ARBD2 infrastructure
    val_buffer_ = std::make_unique<DeviceBuffer<float>>(size_, resource_);
}

/*===============================*\
|        CONSTRUCTORS            |
\===============================*/

BaseGrid::BaseGrid(const Resource& resource) 
    : nx_(1), ny_(1), nz_(1), resource_(resource) {
    basis_ = Matrix3(1.0f);
    origin_ = Vector3(0.0f);
    init();
    zero().wait();
}

BaseGrid::BaseGrid(const Matrix3& basis0, const Vector3& origin0, 
                   int nx0, int ny0, int nz0, const Resource& resource)
    : basis_(basis0), origin_(origin0), resource_(resource) {
    nx_ = std::abs(nx0);
    ny_ = std::abs(ny0);
    nz_ = std::abs(nz0);
    
    init();
    zero().wait();
}

BaseGrid::BaseGrid(const Vector3& box, float dx, const Resource& resource)
    : resource_(resource) {
    dx = std::abs(dx);
    Vector3 abs_box(std::abs(box.x), std::abs(box.y), std::abs(box.z));

    // Tile the grid into the system box
    // Grid spacing is always a bit smaller than dx
    nx_ = int(std::ceil(abs_box.x / dx));
    ny_ = int(std::ceil(abs_box.y / dx));
    nz_ = int(std::ceil(abs_box.z / dx));
    
    if (nx_ <= 0) nx_ = 1;
    if (ny_ <= 0) ny_ = 1;
    if (nz_ <= 0) nz_ = 1;
    
    basis_ = Matrix3(abs_box.x / nx_, abs_box.y / ny_, abs_box.z / nz_);
    origin_ = -0.5f * abs_box;

    init();
    zero().wait();
}

BaseGrid::BaseGrid(const Matrix3& box, int nx0, int ny0, int nz0, 
                   const Resource& resource)
    : nx_(nx0), ny_(ny0), nz_(nz0), resource_(resource) {
    
    if (nx_ <= 0) nx_ = 1;
    if (ny_ <= 0) ny_ = 1;
    if (nz_ <= 0) nz_ = 1;
    
    basis_ = Matrix3(box.ex() / nx_, box.ey() / ny_, box.ez() / nz_);
    origin_ = -0.5f * (box.ex() + box.ey() + box.ez());

    init();
    zero().wait();
}

BaseGrid::BaseGrid(const Matrix3& box, const Vector3& origin0, float dx,
                   const Resource& resource)
    : origin_(origin0), resource_(resource) {
    dx = std::abs(dx);
    
    // Grid spacing is always a bit larger than dx
    nx_ = int(std::floor(box.ex().length() / dx)) - 1;
    ny_ = int(std::floor(box.ey().length() / dx)) - 1;
    nz_ = int(std::floor(box.ez().length() / dx)) - 1;
    
    if (nx_ <= 0) nx_ = 1;
    if (ny_ <= 0) ny_ = 1;
    if (nz_ <= 0) nz_ = 1;

    basis_ = Matrix3(box.ex() / nx_, box.ey() / ny_, box.ez() / nz_);

    init();
    zero().wait();
}

BaseGrid::BaseGrid(const Matrix3& box, float dx, const Resource& resource)
    : resource_(resource) {
    dx = std::abs(dx);
    
    // Grid spacing is always a bit smaller than dx
    nx_ = int(std::ceil(box.ex().length() / dx));
    ny_ = int(std::ceil(box.ey().length() / dx));
    nz_ = int(std::ceil(box.ez().length() / dx));
    
    if (nx_ <= 0) nx_ = 1;
    if (ny_ <= 0) ny_ = 1;
    if (nz_ <= 0) nz_ = 1;

    basis_ = Matrix3(box.ex() / nx_, box.ey() / ny_, box.ez() / nz_);
    origin_ = -0.5f * (box.ex() + box.ey() + box.ez());

    init();
    zero().wait();
}

BaseGrid::BaseGrid(const BaseGrid& g) 
    : nx_(g.nx_), ny_(g.ny_), nz_(g.nz_),
      basis_(g.basis_), origin_(g.origin_), resource_(g.resource_) {
    
    init();
    copy_from(g).wait();
}

BaseGrid::BaseGrid(const BaseGrid& g, int nx0, int ny0, int nz0)
    : nx_(nx0), ny_(ny0), nz_(nz0), resource_(g.resource_) {
    
    if (nx_ <= 0) nx_ = 1;
    if (ny_ <= 0) ny_ = 1;
    if (nz_ <= 0) nz_ = 1;

    // Tile the grid into the box of the template grid
    Matrix3 box = g.getBox();
    basis_ = Matrix3(box.ex() / nx_, box.ey() / ny_, box.ez() / nz_);
    origin_ = g.origin_;
    
    init();

    // Do interpolation to obtain the values
    // This would be implemented as a SYCL kernel
    // For now, just zero the grid
    zero().wait();
    
    // TODO: Implement interpolation kernel
    /*
    for (size_t i = 0; i < size_; i++) {
        Vector3 r = getPosition(i);
        val[i] = g.interpolatePotential(r);
    }
    */
}

BaseGrid::BaseGrid(const char* fileName, const Resource& resource)
    : resource_(resource) {
    read(fileName);
}

BaseGrid::BaseGrid(BaseGrid&& other) noexcept 
    : nx_(other.nx_), ny_(other.ny_), nz_(other.nz_), size_(other.size_),
      basis_(other.basis_), basisInv_(other.basisInv_), origin_(other.origin_),
      val_buffer_(std::move(other.val_buffer_)), resource_(other.resource_) {
    
    // Reset other object to valid state
    other.nx_ = other.ny_ = other.nz_ = 1;
    other.size_ = 1;
    other.basis_ = Matrix3(1.0f);
    other.basisInv_ = Matrix3(1.0f);
    other.origin_ = Vector3(0.0f);
}

/*===============================*\
|          OPERATORS             |
\===============================*/

BaseGrid& BaseGrid::operator=(const BaseGrid& g) {
    if (this != &g) {
        nx_ = g.nx_;
        ny_ = g.ny_;
        nz_ = g.nz_;
        basis_ = g.basis_;
        origin_ = g.origin_;
        resource_ = g.resource_;
        
        init();
        copy_from(g).wait();
    }
    return *this;
}

BaseGrid& BaseGrid::operator=(BaseGrid&& other) noexcept {
    if (this != &other) {
        nx_ = other.nx_;
        ny_ = other.ny_;
        nz_ = other.nz_;
        size_ = other.size_;
        basis_ = other.basis_;
        basisInv_ = other.basisInv_;
        origin_ = other.origin_;
        val_buffer_ = std::move(other.val_buffer_);
        resource_ = other.resource_;
        
        // Reset other object to valid state
        other.nx_ = other.ny_ = other.nz_ = 1;
        other.size_ = 1;
        other.basis_ = Matrix3(1.0f);
        other.basisInv_ = Matrix3(1.0f);
        other.origin_ = Vector3(0.0f);
    }
    return *this;
}

BaseGrid& BaseGrid::mult(const BaseGrid& g) {
    multiply_grid(g).wait();
    return *this;
}

/*===============================*\
|       MEMORY OPERATIONS        |
\===============================*/

BACKEND::Event BaseGrid::zero() {
    return dispatch_zero();
}

BACKEND::Event BaseGrid::copy_from(const BaseGrid& other) {
    return dispatch_copy_from(other);
}

/*===============================*\
|       GRID OPERATIONS          |
\===============================*/

BACKEND::Event BaseGrid::add_scalar(float value) {
    return dispatch_add_scalar(value);
}

BACKEND::Event BaseGrid::multiply_scalar(float value) {
    return dispatch_multiply_scalar(value);
}

BACKEND::Event BaseGrid::add_grid(const BaseGrid& other) {
    return dispatch_add_grid(other);
}

BACKEND::Event BaseGrid::multiply_grid(const BaseGrid& other) {
    return dispatch_multiply_grid(other);
}

/*===============================*\
|       BACKEND DISPATCH         |
\===============================*/

BACKEND::Event BaseGrid::dispatch_zero() {
    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_zero_grid();
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_zero_grid();
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_zero_grid();
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback
            float* data = val_buffer_->data();
            std::fill(data, data + size_, 0.0f);
            return BACKEND::Event(nullptr, resource_);
    }
}

BACKEND::Event BaseGrid::dispatch_copy_from(const BaseGrid& other) {
    if (size_ != other.size_) {
        throw std::runtime_error("BaseGrid::dispatch_copy_from: Size mismatch");
    }

    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_copy_from(other);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_copy_from(other);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_copy_from(other);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback
            std::copy(other.val_buffer_->data(), 
                      other.val_buffer_->data() + size_, 
                      val_buffer_->data());
            return BACKEND::Event(nullptr, resource_);
    }
}

BACKEND::Event BaseGrid::dispatch_add_scalar(float value) {
    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_add_scalar(value);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_add_scalar(value);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_add_scalar(value);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback
            float* data = val_buffer_->data();
            for (size_t i = 0; i < size_; ++i) {
                data[i] += value;
            }
            return BACKEND::Event(nullptr, resource_);
    }
}

BACKEND::Event BaseGrid::dispatch_multiply_scalar(float value) {
    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_multiply_scalar(value);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_multiply_scalar(value);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_multiply_scalar(value);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback
            float* data = val_buffer_->data();
            for (size_t i = 0; i < size_; ++i) {
                data[i] *= value;
            }
            return BACKEND::Event(nullptr, resource_);
    }
}

BACKEND::Event BaseGrid::dispatch_add_grid(const BaseGrid& other) {
    if (size_ != other.size_) {
        throw std::runtime_error("BaseGrid::dispatch_add_grid: Size mismatch");
    }

    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_add_grid(other);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_add_grid(other);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_add_grid(other);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback
            float* data = val_buffer_->data();
            const float* other_data = other.val_buffer_->data();
            for (size_t i = 0; i < size_; ++i) {
                data[i] += other_data[i];
            }
            return BACKEND::Event(nullptr, resource_);
    }
}

BACKEND::Event BaseGrid::dispatch_multiply_grid(const BaseGrid& other) {
    if (size_ != other.size_) {
        throw std::runtime_error("BaseGrid::dispatch_multiply_grid: Size mismatch");
    }

    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_multiply_grid(other);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_multiply_grid(other);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_multiply_grid(other);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback
            float* data = val_buffer_->data();
            const float* other_data = other.val_buffer_->data();
            for (size_t i = 0; i < size_; ++i) {
                data[i] *= other_data[i];
            }
            return BACKEND::Event(nullptr, resource_);
    }
}

template<BoundaryCondition BC>
float BaseGrid::dispatch_interpolate_trilinear(const Vector3& pos) const {
    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_interpolate_trilinear<BC>(pos);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_interpolate_trilinear<BC>(pos);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_interpolate_trilinear<BC>(pos);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback implementation
            // This would contain the CPU version of trilinear interpolation
            return 0.0f; // Placeholder
    }
}

template<BoundaryCondition BC>
Vector3 BaseGrid::dispatch_interpolate_force(const Vector3& pos) const {
    switch (resource_.type) {
#ifdef USE_SYCL
        case ResourceType::SYCL:
            return sycl_interpolate_force<BC>(pos);
#endif
#ifdef USE_CUDA
        case ResourceType::CUDA:
            return cuda_interpolate_force<BC>(pos);
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
            return metal_interpolate_force<BC>(pos);
#endif
        case ResourceType::CPU:
        default:
            // CPU fallback implementation
            return Vector3(0.0f); // Placeholder
    }
}

/*===============================*\
|       SYCL IMPLEMENTATIONS     |
\===============================*/

#ifdef USE_SYCL

BACKEND::Event BaseGrid::sycl_zero_grid() {
    auto& device = SYCL::SYCLManager::get_device(resource_.id);
    auto& queue = device.get_next_queue();
    
    auto event = queue.get().fill(val_buffer_->data(), 0.0f, size_);
    return BACKEND::Event(event, resource_);
}

BACKEND::Event BaseGrid::sycl_copy_from(const BaseGrid& other) {
    auto& device = SYCL::SYCLManager::get_device(resource_.id);
    auto& queue = device.get_next_queue();
    
    auto event = queue.get().copy(other.val_buffer_->data(), 
                                 val_buffer_->data(), size_);
    return BACKEND::Event(event, resource_);
}

BACKEND::Event BaseGrid::sycl_add_scalar(float value) {
    auto& device = SYCL::SYCLManager::get_device(resource_.id);
    auto& queue = device.get_next_queue();
    
    float* data = val_buffer_->data();
    auto event = queue.get().parallel_for(sycl::range<1>(size_), 
        [=](sycl::id<1> idx) {
            data[idx] += value;
        });
    return BACKEND::Event(event, resource_);
}

BACKEND::Event BaseGrid::sycl_multiply_scalar(float value) {
    auto& device = SYCL::SYCLManager::get_device(resource_.id);
    auto& queue = device.get_next_queue();
    
    float* data = val_buffer_->data();
    auto event = queue.get().parallel_for(sycl::range<1>(size_), 
        [=](sycl::id<1> idx) {
            data[idx] *= value;
        });
    return BACKEND::Event(event, resource_);
}

BACKEND::Event BaseGrid::sycl_add_grid(const BaseGrid& other) {
    auto& device = SYCL::SYCLManager::get_device(resource_.id);
    auto& queue = device.get_next_queue();
    
    float* data = val_buffer_->data();
    const float* other_data = other.val_buffer_->data();
    auto event = queue.get().parallel_for(sycl::range<1>(size_), 
        [=](sycl::id<1> idx) {
            data[idx] += other_data[idx];
        });
    return BACKEND::Event(event, resource_);
}

BACKEND::Event BaseGrid::sycl_multiply_grid(const BaseGrid& other) {
    auto& device = SYCL::SYCLManager::get_device(resource_.id);
    auto& queue = device.get_next_queue();
    
    float* data = val_buffer_->data();
    const float* other_data = other.val_buffer_->data();
    auto event = queue.get().parallel_for(sycl::range<1>(size_), 
        [=](sycl::id<1> idx) {
            data[idx] *= other_data[idx];
        });
    return BACKEND::Event(event, resource_);
}

template<BoundaryCondition BC>
float BaseGrid::sycl_interpolate_trilinear(const Vector3& pos) const {
    // Implementation would use SYCL kernels from BaseGridKernels.h
    // For now, return placeholder
    return 0.0f;
}

template<BoundaryCondition BC>
Vector3 BaseGrid::sycl_interpolate_force(const Vector3& pos) const {
    // Implementation would use SYCL kernels from BaseGridKernels.h
    // For now, return placeholder
    return Vector3(0.0f);
}

#endif // USE_SYCL

/*===============================*\
|       CUDA IMPLEMENTATIONS     |
\===============================*/

#ifdef USE_CUDA

BACKEND::Event BaseGrid::cuda_zero_grid() {
    // TODO: Implement CUDA zero grid operation
    // Would use cudaMemset or custom CUDA kernel
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::cuda_copy_from(const BaseGrid& other) {
    // TODO: Implement CUDA copy operation
    // Would use cudaMemcpy for device-to-device copy
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::cuda_add_scalar(float value) {
    // TODO: Implement CUDA add scalar kernel
    // Would launch CUDA kernel similar to:
    // cuda_add_scalar_kernel<<<grid_size, block_size>>>(val_buffer_->data(), value, size_);
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::cuda_multiply_scalar(float value) {
    // TODO: Implement CUDA multiply scalar kernel
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::cuda_add_grid(const BaseGrid& other) {
    // TODO: Implement CUDA add grid kernel
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::cuda_multiply_grid(const BaseGrid& other) {
    // TODO: Implement CUDA multiply grid kernel
    return BACKEND::Event(nullptr, resource_);
}

template<BoundaryCondition BC>
float BaseGrid::cuda_interpolate_trilinear(const Vector3& pos) const {
    // TODO: Implement CUDA trilinear interpolation
    // Could use CUDA texture memory for optimization
    return 0.0f;
}

template<BoundaryCondition BC>
Vector3 BaseGrid::cuda_interpolate_force(const Vector3& pos) const {
    // TODO: Implement CUDA force interpolation
    return Vector3(0.0f);
}

void BaseGrid::cuda_bind_texture() {
    // TODO: Bind CUDA texture memory for optimized interpolation
}

void BaseGrid::cuda_unbind_texture() {
    // TODO: Unbind CUDA texture memory
}

#endif // USE_CUDA

/*===============================*\
|       METAL IMPLEMENTATIONS    |
\===============================*/

#ifdef USE_METAL

BACKEND::Event BaseGrid::metal_zero_grid() {
    // TODO: Implement Metal zero grid operation
    // Would use Metal compute shader
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::metal_copy_from(const BaseGrid& other) {
    // TODO: Implement Metal copy operation
    // Would use Metal blit command encoder
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::metal_add_scalar(float value) {
    // TODO: Implement Metal add scalar compute shader
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::metal_multiply_scalar(float value) {
    // TODO: Implement Metal multiply scalar compute shader
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::metal_add_grid(const BaseGrid& other) {
    // TODO: Implement Metal add grid compute shader
    return BACKEND::Event(nullptr, resource_);
}

BACKEND::Event BaseGrid::metal_multiply_grid(const BaseGrid& other) {
    // TODO: Implement Metal multiply grid compute shader
    return BACKEND::Event(nullptr, resource_);
}

template<BoundaryCondition BC>
float BaseGrid::metal_interpolate_trilinear(const Vector3& pos) const {
    // TODO: Implement Metal trilinear interpolation
    return 0.0f;
}

template<BoundaryCondition BC>
Vector3 BaseGrid::metal_interpolate_force(const Vector3& pos) const {
    // TODO: Implement Metal force interpolation
    return Vector3(0.0f);
}

void BaseGrid::metal_sync_buffers() {
    // TODO: Synchronize Metal buffers if needed
}

#endif // USE_METAL

/*===============================*\
|        INTERPOLATION           |
\===============================*/

float BaseGrid::interpolatePotential(const Vector3& pos) const {
    // Simple trilinear interpolation with periodic boundary conditions
    return dispatch_interpolate_trilinear<BoundaryCondition::periodic>(pos);
}

/*===============================*\
|          FILE I/O              |
\===============================*/

void BaseGrid::write(const char* fileName) const {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("BaseGrid::write: Cannot open file ") + fileName);
    }
    
    // Write header information
    file << "# BaseGrid data file\n";
    file << "# nx ny nz: " << nx_ << " " << ny_ << " " << nz_ << "\n";
    file << "# origin: " << origin_.x << " " << origin_.y << " " << origin_.z << "\n";
    file << "# basis vectors:\n";
    file << "# ex: " << basis_.ex().x << " " << basis_.ex().y << " " << basis_.ex().z << "\n";
    file << "# ey: " << basis_.ey().x << " " << basis_.ey().y << " " << basis_.ey().z << "\n";
    file << "# ez: " << basis_.ez().x << " " << basis_.ez().y << " " << basis_.ez().z << "\n";
    
    // For device data, we'd need to copy to host first
    // This is a simplified version - in practice would need host buffer
    const float* data = val_buffer_->data();
    
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0 && i % 10 == 0) file << "\n";
        file << data[i] << " ";
    }
    file << "\n";
    file.close();
}

void BaseGrid::read(const char* fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("BaseGrid::read: Cannot open file ") + fileName);
    }
    
    // Initialize default values
    nx_ = ny_ = nz_ = 0;
    size_ = 0;
    basis_ = Matrix3(1.0f);
    origin_ = Vector3(0.0f);
    
    std::string line;
    std::vector<float> values;
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Parse header information if present
        if (line.find("nx ny nz:") != std::string::npos) {
            std::istringstream iss(line.substr(line.find(':') + 1));
            iss >> nx_ >> ny_ >> nz_;
            continue;
        }
        
        if (line.find("origin:") != std::string::npos) {
            std::istringstream iss(line.substr(line.find(':') + 1));
            iss >> origin_.x >> origin_.y >> origin_.z;
            continue;
        }
        
        // Parse data values
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            values.push_back(value);
        }
    }
    
    // If grid dimensions weren't specified, try to infer them
    if (nx_ == 0 || ny_ == 0 || nz_ == 0) {
        // Simple cube assumption - this should be improved for real use
        size_t total = values.size();
        nx_ = ny_ = nz_ = int(std::round(std::cbrt(float(total))));
        if (size_t(nx_) * size_t(ny_) * size_t(nz_) != total) {
            throw std::runtime_error("BaseGrid::read: Cannot determine grid dimensions");
        }
    }
    
    // Initialize the grid
    init();
    
    // Copy values to device buffer
    if (values.size() != size_) {
        throw std::runtime_error("BaseGrid::read: Data size mismatch");
    }
    
    // Copy data to device buffer
    float* data = val_buffer_->data();
    std::copy(values.begin(), values.end(), data);
    
    file.close();
}

/*===============================*\
|       HELPER FUNCTIONS         |
\===============================*/

HOST DEVICE float BaseGrid::getValue(int i, int j, int k) const {
    if (!isInBounds(i, j, k)) return 0.0f;
    size_t idx = getIndex(i, j, k);
    return val_buffer_->data()[idx];
}

HOST DEVICE float BaseGrid::getValueSafe(int i, int j, int k, BoundaryCondition bc) const {
    Vector3_t<int> idx;
    
    switch (bc) {
        case BoundaryCondition::periodic:
            idx = applyBoundaryConditions<BoundaryCondition::periodic>(i, j, k);
            break;
        case BoundaryCondition::dirichlet:
            idx = applyBoundaryConditions<BoundaryCondition::dirichlet>(i, j, k);
            break;
        case BoundaryCondition::neumann:
            idx = applyBoundaryConditions<BoundaryCondition::neumann>(i, j, k);
            break;
    }
    
    return getValue(idx.x, idx.y, idx.z);
}

} // namespace ARBD