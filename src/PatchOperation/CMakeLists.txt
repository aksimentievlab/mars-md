# PatchOperation folder CMakeLists.txt
# Handles patch operations, pairlists, integrators, and related kernels

# Common source files
list(APPEND LIB_SOURCES
    PatchOperation/Patch.cpp
    PatchOperation/Pairlist.cpp
    #PatchOperation/PairListKernels/CellListPairlist.cpp
    PatchOperation/PairListKernels/ZOrderPairlist.cpp
    PatchOperation/ZOrderKernels/ZOrderSort.cpp
)

# Common headers
list(APPEND HOST_DEVICE_HEADERS
    PatchOperation/Patch.h
    PatchOperation/Pairlist.h
    PatchOperation/Integrator.h
    PatchOperation/Colvars.h
    PatchOperation/ComputeEnergy.h
    PatchOperation/ZOrderKernels.h
    # PairListKernels headers
    PatchOperation/PairListKernels/CellListPairlist.h
    PatchOperation/PairListKernels/VerletCompute.h
    PatchOperation/PairListKernels/ClusterCompute.h
    PatchOperation/PairListKernels/ZOrderNeighbor.h
    PatchOperation/PairListKernels/ZOrderPairlist.h
    PatchOperation/PairListKernels/AdaptiveZOrderNeighbor.h
    PatchOperation/PairListKernels/DisplacementTracker.h
    PatchOperation/PairListKernels/DecomposeKernels.h
    # Integrator headers
    PatchOperation/Integrator/BD.h
    PatchOperation/Integrator/BAOAB.h
    #PatchOperation/Integrator/BBK.h
    # Random headers
    PatchOperation/Random/Random.h
    PatchOperation/Random/RandomKernels.h
    PatchOperation/Random/philox.h
    PatchOperation/Random/base_state.h
    PatchOperation/Random/util.h
    # ZOrderKernels headers
    PatchOperation/ZOrderKernels/AdaptiveKernels.h
    PatchOperation/ZOrderKernels/ZMetal.h
    PatchOperation/ZOrderKernels/ZOrderSort.h
    PatchOperation/ZOrderKernels/MortonCode.h
)

# CUDA specific source files
if(USE_CUDA)
    list(APPEND LIB_SOURCES
        PatchOperation/ZOrderKernels/CUDASort.cu
        PatchOperation/ZOrderKernels/ZOrderKernels.cu
        PatchOperation/PairListKernels/ZOrderPairlist.cu
        PatchOperation/Random/Random.cu
        PatchOperation/Integrator/IntegratorInstantiations.cu
        #PatchOperation/PairListKernels/VerletCompute.cu
        #PatchOperation/PairListKernels/ClusterCompute.cu
    )

    # Set CUDA properties for .cu files
    set_source_files_properties(
        PatchOperation/ZOrderKernels/ZOrderKernels.cu
        PatchOperation/PairListKernels/ZOrderPairlist.cu
        PatchOperation/Random/Random.cu
        PatchOperation/Integrator/IntegratorInstantiations.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "PatchOperation: Adding CUDA specific sources to lib${PROJECT_NAME}")
endif()

# SYCL specific source files
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    list(APPEND SYCL_DEVICE_SOURCES
        PatchOperation/ZOrderKernels/DeviceRadixSortUSM.cpp
        # Kernels used by Z-order sorting and pairlists
        PatchOperation/ZOrderKernels/ZOrderSort.cpp
        PatchOperation/PairListKernels/ZOrderPairlist.cpp
        PatchOperation/PairListKernels/CellListPairlist.cpp
        # Instantiates the integrator kernels (launch_BAOAB / launch_BD) via
        # templates in Integrator.h, so it needs device compilation too.
        PatchOperation/Patch.cpp
    )
    list(APPEND DEVICE_ONLY_HEADERS
        PatchOperation/ZOrderKernels/DeviceRadix.h
        #PatchOperation/ZOrderKernels/oneapiSort.h
    )
    list(APPEND LIB_SOURCES
        PatchOperation/ZOrderKernels/DeviceRadixSortUSM.cpp
    )
    message(STATUS "PatchOperation: Adding SYCL device sources to lib${PROJECT_NAME}")
endif()

# Metal specific source files
if(USE_METAL)
    message(STATUS "PatchOperation: Metal shaders handled separately")
endif()

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(DEVICE_ONLY_HEADERS ${DEVICE_ONLY_HEADERS} PARENT_SCOPE)
if(USE_SYCL_ICPX OR USE_SYCL_ACPP)
    set(SYCL_DEVICE_SOURCES ${SYCL_DEVICE_SOURCES} PARENT_SCOPE)
endif()
