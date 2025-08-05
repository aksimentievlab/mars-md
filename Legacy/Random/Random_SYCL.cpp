#ifdef USE_SYCL
#include "Random.h"
#include "Backend/SYCL/SYCLManager.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "openrand/philox.h"
#include <sycl/sycl.hpp>
#include "ARBDLogger.h"
#include "ARBDException.h"

namespace ARBD {

// SYCL Random Number Generation Kernels
namespace RandomSYCLKernels {

// Create Philox generator for thread
inline openrand::Philox create_philox(unsigned long seed, unsigned int thread_id, unsigned int offset) {
    return openrand::Philox(seed, thread_id + offset);
}

// Box-Muller transform for Gaussian from uniform
inline sycl::float2 box_muller(float u1, float u2) {
    float r = sycl::sqrt(-2.0f * sycl::log(u1));
    float theta = 2.0f * M_PI * u2;
    return sycl::float2{r * sycl::cos(theta), r * sycl::sin(theta)};
}

} // namespace RandomSYCLKernels

// RandomSYCL Implementation
template<size_t num_states>
RandomSYCL<num_states>::RandomSYCL(const Resource& resource) 
    : RandomDevice<num_states>(resource) {
    if (resource.type != ResourceType::SYCL) {
        throw_value_error("RandomSYCL requires SYCL resource");
    }
    LOGINFO("Created RandomSYCL with {} states", num_states);
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_uniform(DeviceBuffer<float>& output, float min, float max) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            float* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              if (tid >= count) return;
                              
                              auto rng = RandomSYCLKernels::create_philox(seed, tid, offset);
                              auto result = rng.draw_float4();
                              
                              out_ptr[tid] = min + result.x * (max - min);
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in uniform float generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_gaussian(DeviceBuffer<float>& output, float mean, float stddev) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            float* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              size_t stride = item.get_global_range(0);
                              
                              // Process pairs of outputs for Box-Muller
                              for (size_t i = tid * 2; i < count; i += stride * 2) {
                                  auto rng = RandomSYCLKernels::create_philox(seed, i, offset);
                                  auto uniform_vals = rng.draw_float4();
                                  
                                  float u1 = sycl::max(uniform_vals.x, 1e-7f);
                                  float u2 = uniform_vals.y;
                                  
                                  auto gauss_pair = RandomSYCLKernels::box_muller(u1, u2);
                                  
                                  if (i < count) {
                                      out_ptr[i] = mean + stddev * gauss_pair.x();
                                  }
                                  if (i + 1 < count) {
                                      out_ptr[i + 1] = mean + stddev * gauss_pair.y();
                                  }
                              }
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in Gaussian float generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_gaussian_vector3(DeviceBuffer<Vector3>& output, float mean, float stddev) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            Vector3* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              if (tid >= count) return;
                              
                              auto rng = RandomSYCLKernels::create_philox(seed, tid, offset);
                              auto uniform_vals1 = rng.draw_float4();
                              auto uniform_vals2 = rng.draw_float4();
                              
                              float u1 = sycl::max(uniform_vals1.x, 1e-7f);
                              float u2 = uniform_vals1.y;
                              float u3 = sycl::max(uniform_vals1.z, 1e-7f);
                              float u4 = uniform_vals1.w;
                              
                              auto gauss_pair1 = RandomSYCLKernels::box_muller(u1, u2);
                              auto gauss_pair2 = RandomSYCLKernels::box_muller(u3, u4);
                              
                              out_ptr[tid] = Vector3(
                                  mean + stddev * gauss_pair1.x(),
                                  mean + stddev * gauss_pair1.y(),
                                  mean + stddev * gauss_pair2.x()
                              );
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in Gaussian Vector3 generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_uniform(DeviceBuffer<double>& output, double min, double max) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            double* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              if (tid >= count) return;
                              
                              auto rng = RandomSYCLKernels::create_philox(seed, tid, offset);
                              uint64_t u64 = rng.template draw<uint64_t>();
                              
                              double u = u64 * 0x1.0p-64;  // Convert to [0,1)
                              out_ptr[tid] = min + u * (max - min);
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in uniform double generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_gaussian(DeviceBuffer<double>& output, double mean, double stddev) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            double* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              size_t stride = item.get_global_range(0);
                              
                              for (size_t i = tid * 2; i < count; i += stride * 2) {
                                  auto rng = RandomSYCLKernels::create_philox(seed, i, offset);
                                  
                                  uint64_t u64_1 = rng.template draw<uint64_t>();
                                  uint64_t u64_2 = rng.template draw<uint64_t>();
                                  
                                  double u1 = sycl::max(u64_1 * 0x1.0p-64, 1e-15);
                                  double u2 = u64_2 * 0x1.0p-64;
                                  
                                  double r = sycl::sqrt(-2.0 * sycl::log(u1));
                                  double theta = 2.0 * M_PI * u2;
                                  
                                  if (i < count) {
                                      out_ptr[i] = mean + stddev * r * sycl::cos(theta);
                                  }
                                  if (i + 1 < count) {
                                      out_ptr[i + 1] = mean + stddev * r * sycl::sin(theta);
                                  }
                              }
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in Gaussian double generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_uniform_int(DeviceBuffer<int>& output, int min, int max) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            int* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              if (tid >= count) return;
                              
                              auto rng = RandomSYCLKernels::create_philox(seed, tid, offset);
                              uint32_t u32 = rng.template draw<uint32_t>();
                              
                              unsigned int range = static_cast<unsigned int>(max - min + 1);
                              if (range == 0) {
                                  out_ptr[tid] = min;
                                  return;
                              }
                              
                              // Avoid modulo bias for more uniform distribution
                              unsigned int limit = UINT_MAX - (UINT_MAX % range);
                              while (u32 >= limit) {
                                  u32 = rng.template draw<uint32_t>();
                              }
                              
                              out_ptr[tid] = min + static_cast<int>(u32 % range);
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in uniform int generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomSYCL<num_states>::generate_uniform_uint(DeviceBuffer<unsigned int>& output, unsigned int min, unsigned int max) {
    if (!this->initialized_) {
        throw_sycl_error("RandomSYCL not initialized - call init() first");
    }
    
    auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
    auto& queue = sycl_device.get_next_queue();
    
    const size_t count = output.size();
    const size_t work_group_size = 256;
    const size_t num_work_groups = (count + work_group_size - 1) / work_group_size;
    
    try {
        auto sycl_event = queue.get().submit([&](sycl::handler& h) {
            unsigned int* out_ptr = output.data();
            const unsigned long seed = this->seed_;
            const size_t offset = this->offset_;
            
            h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
                          [=](sycl::nd_item<1> item) {
                              size_t tid = item.get_global_id(0);
                              if (tid >= count) return;
                              
                              auto rng = RandomSYCLKernels::create_philox(seed, tid, offset);
                              uint32_t u32 = rng.template draw<uint32_t>();
                              
                              unsigned int range = max - min + 1;
                              if (range == 0) {
                                  out_ptr[tid] = min;
                                  return;
                              }
                              
                              unsigned int limit = UINT_MAX - (UINT_MAX % range);
                              while (u32 >= limit) {
                                  u32 = rng.template draw<uint32_t>();
                              }
                              
                              out_ptr[tid] = min + (u32 % range);
                          });
        });
        
        return Event(sycl_event, this->resource_);
        
    } catch (const sycl::exception& e) {
        throw_sycl_error("SYCL error in uniform uint generation: {}", e.what());
    }
}

// Explicit template instantiations for common sizes
template class RandomSYCL<32>;
template class RandomSYCL<64>;
template class RandomSYCL<128>;
template class RandomSYCL<256>;
template class RandomSYCL<512>;
template class RandomSYCL<1024>;

} // namespace ARBD

#endif // USE_SYCL