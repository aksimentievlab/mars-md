/*********************************************************************
 * @file  BaseGrid.cpp
 *
 * @brief BaseGrid V2 implementation using launch_kernel system
 *********************************************************************/

 #include "Math/BaseGrid.h"
 #include "Math/BaseGridKernels.h"  // Separate kernel definitions
 #include "ARBDLogger.h"
 #include "ARBDException.h"
 
 #ifdef USE_SYCL
 #include "Backend/SYCL/SYCLManager.h"
 #endif
 
 #ifdef USE_CUDA
 #include "Backend/CUDA/CUDAManager.h"
 #endif
 
 namespace ARBD {
 
 /*===============================*\
 |       PRIVATE METHODS          |
 \===============================*/
 
 void BaseGrid::init() {
     basisInv_ = basis_.inverse();
     size_ = size_t(nx_) * size_t(ny_) * size_t(nz_);
     
     // V2: Use DeviceBuffer with RAII
     grid_buffer = std::make_unique<DeviceBuffer<float>>(size_);
     
     LOGDEBUG("BaseGrid initialized: {}x{}x{} = {} elements", nx_, ny_, nz_, size_);
 }
 
 /*===============================*\
 |        CONSTRUCTORS            |
 \===============================*/
 
 BaseGrid::BaseGrid(const Resource& resource) 
     : nx_(1), ny_(1), nz_(1), resource_(resource) {
     basis_ = Matrix3(1.0f);
     origin_ = Vector3(0.0f);
     init();
     zero().wait();  // V2: Event-driven
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
 
     // Tile the grid into the system box (grid spacing smaller than dx)
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
 
 // Copy constructor - V2 version
 BaseGrid::BaseGrid(const BaseGrid& g) 
     : nx_(g.nx_), ny_(g.ny_), nz_(g.nz_), 
       basis_(g.basis_), origin_(g.origin_), resource_(g.resource_) {
     init();
     copy_from(g).wait();  // V2: Event-driven copy
 }
 
 // Copy constructor with different resolution
 BaseGrid::BaseGrid(const BaseGrid& g, int nx0, int ny0, int nz0)
     : resource_(g.resource_) {
     nx_ = std::abs(nx0);
     ny_ = std::abs(ny0);
     nz_ = std::abs(nz0);
     
     if (nx_ <= 0) nx_ = 1;
     if (ny_ <= 0) ny_ = 1;
     if (nz_ <= 0) nz_ = 1;
 
     // Tile the grid into the box of the template grid
     Matrix3 box = g.getBox();
     basis_ = Matrix3(box.ex() / nx_, box.ey() / ny_, box.ez() / nz_);
     origin_ = g.origin_;
     
     init();
     
     // V2: Perform interpolation using dispatch system
     // This will be implemented in backend-specific files
     //batch_interpolate<BoundaryCondition::periodic>(g).wait();
 }
 
 /*===============================*\
 |         OPERATORS              |
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
 
 BaseGrid& BaseGrid::mult(const BaseGrid& g) {
     multiply_grid(g).wait();
     return *this;
 }
 
 /*===============================*\
 |      MEMORY OPERATIONS         |
 \===============================*/
 
 BACKEND::Event BaseGrid::zero() {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, BaseGridKernels::ZeroKernel{});
 }
 
 BACKEND::Event BaseGrid::copy_from(const BaseGrid& other) {
     if (size_ != other.size_) {
         throw_value_error("BaseGrid::copy_from: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *other.grid_buffer, *grid_buffer, BaseGridKernels::CopyKernel{});
 }
 
 BACKEND::Event BaseGrid::add_scalar(float value) {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, BaseGridKernels::AddScalarKernel{value});
 }
 
 BACKEND::Event BaseGrid::multiply_scalar(float value) {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, BaseGridKernels::MultiplyScalarKernel{value});
 }
 
 BACKEND::Event BaseGrid::add_grid(const BaseGrid& other) {
     if (size_ != other.size_) {
         throw_value_error("BaseGrid::add_grid: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, *other.grid_buffer, BaseGridKernels::AddGridKernel{});
 }
 
 BACKEND::Event BaseGrid::multiply_grid(const BaseGrid& other) {
     if (size_ != other.size_) {
         throw_value_error("BaseGrid::multiply_grid: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, *other.grid_buffer, BaseGridKernels::MultiplyGridKernel{});
 }
 
 BACKEND::Event BaseGrid::subtract_grid(const BaseGrid& other) {
     if (size_ != other.size_) {
         throw_value_error("BaseGrid::subtract_grid: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, *other.grid_buffer, BaseGridKernels::SubtractGridKernel{});
 }
 
 BACKEND::Event BaseGrid::divide_grid(const BaseGrid& other, float epsilon) {
     if (size_ != other.size_) {
         throw_value_error("BaseGrid::divide_grid: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, *other.grid_buffer, BaseGridKernels::DivideGridKernel{epsilon});
 }
 
 BACKEND::Event BaseGrid::set_constant(float value) {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, BaseGridKernels::SetConstantKernel{value});
 }
 
 BACKEND::Event BaseGrid::abs() {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, AbsKernel{});
 }
 
 BACKEND::Event BaseGrid::clamp(float min_val, float max_val) {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, ClampKernel{min_val, max_val});
 }
 
 BACKEND::Event BaseGrid::power(float exponent) {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, PowerKernel{exponent});
 }
 
 BACKEND::Event BaseGrid::exp() {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, ExpKernel{});
 }
 
 BACKEND::Event BaseGrid::log(float epsilon) {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, LogKernel{epsilon});
 }
 
 BACKEND::Event BaseGrid::square() {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, SquareKernel{});
 }
 
 BACKEND::Event BaseGrid::sqrt() {
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, SqrtKernel{});
 }Event BaseGrid::add_grid(const BaseGrid& other) {
     if (size_ != other.size_) {
         throw ARBDException("BaseGrid::add_grid: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, *other.grid_buffer, AddGridKernel{});
 }
 
 BACKEND::Event BaseGrid::multiply_grid(const BaseGrid& other) {
     if (size_ != other.size_) {
         throw ARBDException("BaseGrid::multiply_grid: Size mismatch");
     }
     return launch_kernel(resource_, size_, KernelConfig{}, 
                         *grid_buffer, *other.grid_buffer, MultiplyGridKernel{});
 }
 
 /*===============================*\
 |      BACKEND DISPATCH          |
 \===============================*/
 
 BACKEND::Event BaseGrid::dispatch_zero() {
     switch (resource_.type()) {
 #ifdef USE_CUDA
         case ResourceType::CUDA:
             return cuda_zero();
 #endif
 #ifdef USE_SYCL
         case ResourceType::SYCL:
             return sycl_zero();
 #endif
         case ResourceType::CPU:
         default:
             // CPU fallback: zero using memset
             std::memset(grid_buffer->data(), 0, size_ * sizeof(float));
             return BACKEND::Event::immediate();
     }
 }
 
 BACKEND::Event BaseGrid::dispatch_copy_from(const BaseGrid& other) {
     switch (resource_.type()) {
 #ifdef USE_CUDA
         case ResourceType::CUDA:
             return cuda_copy_from(other);
 #endif
 #ifdef USE_SYCL
         case ResourceType::SYCL:
             return sycl_copy_from(other);
 #endif
         case ResourceType::CPU:
         default:
             // CPU fallback
             std::memcpy(grid_buffer->data(), other.grid_buffer->data(), 
                        size_ * sizeof(float));
             return BACKEND::Event::immediate();
     }
 }
 
 /*===============================*\
 |      INTERPOLATION METHODS     |
 \===============================*/
 
 // V1 compatibility methods
 float BaseGrid::interpolatePotential(const Vector3& pos) const {
     return interpolate_trilinear<BoundaryCondition::periodic>(pos);
 }
 
 Vector3 BaseGrid::interpolateForce(const Vector3& pos) const {
     return interpolate_force<BoundaryCondition::periodic>(pos);
 }
 
 template<BoundaryCondition BC>
 float BaseGrid::interpolate_trilinear(const Vector3& pos) const {
     // For single-point interpolation, use CPU implementation
     // since it's typically called per-particle and not amenable to batching
     return cpu_interpolate_trilinear<BC>(pos, grid_buffer->data());
 }
 
 template<BoundaryCondition BC>
 Vector3 BaseGrid::interpolate_force(const Vector3& pos) const {
     // For single-point force interpolation, use CPU implementation
     return cpu_interpolate_force<BC>(pos, grid_buffer->data());
 }
 
 template<BoundaryCondition BC>
 BACKEND::Event BaseGrid::batch_interpolate(const DeviceBuffer<Vector3>& positions,
                                           DeviceBuffer<float>& results) const {
     // For batch interpolation, use kernels for better performance
     TrilinearInterpolationKernel<BC> kernel{
         nx_, ny_, nz_, origin_, basisInv_
     };
     
     return launch_kernel(resource_, positions.size(), KernelConfig{},
                         positions, *grid_buffer, results, kernel);
 }
 
 /*===============================*\
 |      CPU IMPLEMENTATIONS       |
 \===============================*/
 
 template<BoundaryCondition BC>
 float BaseGrid::cpu_interpolate_trilinear(const Vector3& pos, const float* data) const {
     // Convert world position to grid coordinates
     Vector3 grid_pos = basisInv_.transform(pos - origin_);
     
     // Get integer and fractional parts
     int i0 = int(std::floor(grid_pos.x));
     int j0 = int(std::floor(grid_pos.y));
     int k0 = int(std::floor(grid_pos.z));
     
     float fx = grid_pos.x - i0;
     float fy = grid_pos.y - j0;
     float fz = grid_pos.z - k0;
     
     // Apply boundary conditions and sample 8 corner points
     float c000, c001, c010, c011, c100, c101, c110, c111;
     
     auto idx000 = applyBoundaryConditions<BC>(i0, j0, k0);
     auto idx001 = applyBoundaryConditions<BC>(i0, j0, k0+1);
     auto idx010 = applyBoundaryConditions<BC>(i0, j0+1, k0);
     auto idx011 = applyBoundaryConditions<BC>(i0, j0+1, k0+1);
     auto idx100 = applyBoundaryConditions<BC>(i0+1, j0, k0);
     auto idx101 = applyBoundaryConditions<BC>(i0+1, j0, k0+1);
     auto idx110 = applyBoundaryConditions<BC>(i0+1, j0+1, k0);
     auto idx111 = applyBoundaryConditions<BC>(i0+1, j0+1, k0+1);
     
     c000 = data[getIndex(idx000.x, idx000.y, idx000.z)];
     c001 = data[getIndex(idx001.x, idx001.y, idx001.z)];
     c010 = data[getIndex(idx010.x, idx010.y, idx010.z)];
     c011 = data[getIndex(idx011.x, idx011.y, idx011.z)];
     c100 = data[getIndex(idx100.x, idx100.y, idx100.z)];
     c101 = data[getIndex(idx101.x, idx101.y, idx101.z)];
     c110 = data[getIndex(idx110.x, idx110.y, idx110.z)];
     c111 = data[getIndex(idx111.x, idx111.y, idx111.z)];
     
     // Trilinear interpolation
     float c00 = c000 * (1.0f - fx) + c100 * fx;
     float c01 = c001 * (1.0f - fx) + c101 * fx;
     float c10 = c010 * (1.0f - fx) + c110 * fx;
     float c11 = c011 * (1.0f - fx) + c111 * fx;
     
     float c0 = c00 * (1.0f - fy) + c10 * fy;
     float c1 = c01 * (1.0f - fy) + c11 * fy;
     
     return c0 * (1.0f - fz) + c1 * fz;
 }
 
 template<BoundaryCondition BC>
 Vector3 BaseGrid::cpu_interpolate_force(const Vector3& pos, const float* data) const {
     // Force is negative gradient of potential
     // Use finite differences to compute gradient
     
     const float h = 0.001f; // Small step size
     Vector3 force;
     
     // Central differences for gradient
     force.x = -(cpu_interpolate_trilinear<BC>(pos + Vector3(h, 0, 0), data) - 
                 cpu_interpolate_trilinear<BC>(pos - Vector3(h, 0, 0), data)) / (2.0f * h);
     
     force.y = -(cpu_interpolate_trilinear<BC>(pos + Vector3(0, h, 0), data) - 
                 cpu_interpolate_trilinear<BC>(pos - Vector3(0, h, 0), data)) / (2.0f * h);
     
     force.z = -(cpu_interpolate_trilinear<BC>(pos + Vector3(0, 0, h), data) - 
                 cpu_interpolate_trilinear<BC>(pos - Vector3(0, 0, h), data)) / (2.0f * h);
     
     return force;
 }
 
 // Explicit template instantiations for common boundary conditions
 template float BaseGrid::interpolate_trilinear<BoundaryCondition::periodic>(const Vector3& pos) const;
 template float BaseGrid::interpolate_trilinear<BoundaryCondition::dirichlet>(const Vector3& pos) const;
 template float BaseGrid::interpolate_trilinear<BoundaryCondition::neumann>(const Vector3& pos) const;
 
 template Vector3 BaseGrid::interpolate_force<BoundaryCondition::periodic>(const Vector3& pos) const;
 template Vector3 BaseGrid::interpolate_force<BoundaryCondition::dirichlet>(const Vector3& pos) const;
 template Vector3 BaseGrid::interpolate_force<BoundaryCondition::neumann>(const Vector3& pos) const;
 
 template BACKEND::Event BaseGrid::batch_interpolate<BoundaryCondition::periodic>(
     const DeviceBuffer<Vector3>& positions, DeviceBuffer<float>& results) const;
 template BACKEND::Event BaseGrid::batch_interpolate<BoundaryCondition::dirichlet>(
     const DeviceBuffer<Vector3>& positions, DeviceBuffer<float>& results) const;
 template BACKEND::Event BaseGrid::batch_interpolate<BoundaryCondition::neumann>(
     const DeviceBuffer<Vector3>& positions, DeviceBuffer<float>& results) const;
 
 } // namespace ARBD