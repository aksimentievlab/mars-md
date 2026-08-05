# Objects folder CMakeLists.txt
# Handles object-related headers (mostly header-only)

list(APPEND HOST_DEVICE_HEADERS
    Objects/DeviceParticle.h
    Objects/DeviceRigidBody.h
)

list(APPEND HOST_ONLY_HEADERS
    Objects/Grid.h
    Objects/Tables.h
    Objects/ParticleProperties.h
    Objects/RigidBodyProperties.h
    #Objects/GroupSites.h
)

# Propagate to parent scope
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)
    list(APPEND LIB_SOURCES
        Objects/RigidBodyCosmeticsKernel.cu
    )

    # Set CUDA properties for .cu files
    set_source_files_properties(
        Objects/RigidBodyCosmeticsKernel.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "Objects: Adding CUDA specific sources to lib${PROJECT_NAME}")
endif()

# Must be propagated *after* the CUDA block above, or the .cu appended there is
# dropped and its explicit instantiations never make it into the library.
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
