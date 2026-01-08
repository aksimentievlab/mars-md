# Backend folder CMakeLists.txt
# Handles backend-specific source files and headers

list(APPEND HOST_ONLY_HEADERS
    Backend/Resource.h
    Backend/MPIManager.h
    Backend/CUDA/CUDAManager.h
    Backend/SYCL/SYCLManager.h
    Backend/SYCL/SYCLQueues.h
    Backend/SYCL/ONECCLManager.h
    Backend/METAL/METALManager.h
    Backend/Profiler.h
)

list(APPEND HOST_DEVICE_HEADERS
    Backend/Buffer.h
    Backend/Events.h
    Backend/KernelConfig.h
    Backend/Kernels.h
    Backend/Collective.h
    Backend/CUDA/CUDABuffer.h
    Backend/CUDA/CUDAStreams.h
    Backend/CUDA/KernelHelper.cuh
    Backend/CUDA/MemoryKernels.cuh
    Backend/CUDA/NCCLManager.h
    Backend/SYCL/SYCLBuffer.h
    Backend/METAL/METALBuffer.h
    Backend/METAL/METALKernels.h
)

# SYCL device-only headers (contain device code)
list(APPEND DEVICE_ONLY_HEADERS
    Backend/SYCL/SYCLKernels.h
)


list(APPEND LIB_SOURCES
    Backend/Resource.cpp
)

# Conditionally add MPI specific source files to the library
if(USE_MPI)
    list(APPEND LIB_SOURCES
        Backend/MPIManager.cpp
    )
    message(STATUS "Backend: Adding MPI specific sources to lib${PROJECT_NAME}")
endif()

# Conditionally add SYCL specific source files to the library
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "Backend: Adding SYCL specific sources to lib${PROJECT_NAME}")
    # SYCL host-only sources (no device kernels)
    list(APPEND SYCL_HOST_SOURCES
        Backend/SYCL/SYCLManager.cpp
        Backend/SYCL/ONECCLManager.cpp
    )
    list(APPEND LIB_SOURCES
        Backend/SYCL/SYCLManager.cpp
        Backend/SYCL/ONECCLManager.cpp
    )
endif()

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)
    message(STATUS "Backend: Adding CUDA specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/CUDA/CUDAManager.cpp
        Backend/CUDA/KernelHelper.cu
        Backend/CUDA/NCCLManager.cpp
    )

    # Set CUDA properties for .cu files
    set_source_files_properties(
        Backend/CUDA/KernelHelper.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "Backend: CUDA is enabled for lib${PROJECT_NAME}")
endif()

# Conditionally add Metal specific source files to the library
if(USE_METAL)
    message(STATUS "Backend: Adding Metal specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/METAL/METALManager.cpp
    )
    message(STATUS "Backend: Metal is enabled for lib${PROJECT_NAME}")
endif()

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(DEVICE_ONLY_HEADERS ${DEVICE_ONLY_HEADERS} PARENT_SCOPE)
if(USE_SYCL_ICPX OR USE_SYCL_ACPP)
    set(SYCL_HOST_SOURCES ${SYCL_HOST_SOURCES} PARENT_SCOPE)
endif()

