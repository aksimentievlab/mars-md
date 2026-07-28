# Interactions folder CMakeLists.txt
# Handles interaction-related source files and headers

list(APPEND LIB_SOURCES
    Interactions/BondedInteraction.cpp
    Interactions/NonBondedInteraction.cpp
)

list(APPEND HOST_DEVICE_HEADERS
    Interactions/Bonded/BondGeometry.h
    Interactions/Bonded/BondComputer.h
    Interactions/Bonded/Analytical.h
    Interactions/Bonded/ProductPotentialMigration.h
    Interactions/Nonbonded/PmfKernels.h
    Interactions/Nonbonded/GridGridKernels.h
    Interactions/Nonbonded/Pairwise.h
    Interactions/Nonbonded/Columb.h
    Interactions/TabulatedPotential.h
    Interactions/Interactions.h
    Interactions/Exclusion.h
    Interactions/SMDGrid.h
)

list(APPEND HOST_ONLY_HEADERS
    Interactions/BondedInteraction.h
    Interactions/NonBondedInteraction.h
    Interactions/DevicePairNonBondedInteraction.h
    Interactions/Improper.h
)

# Conditionally add SYCL specific source files to the library
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "Interactions: Adding SYCL specific sources to lib${PROJECT_NAME}")
    # Add SYCL-specific interaction sources here if needed
endif()

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)
    list(APPEND LIB_SOURCES
        Interactions/Bonded/BondedInstantiations.cu
        Interactions/Nonbonded/NonbondedInstantiations.cu
    )

    # Set CUDA properties for .cu files
    set_source_files_properties(
        Interactions/Bonded/BondedInstantiations.cu
        Interactions/Nonbonded/NonbondedInstantiations.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "Interactions: Adding CUDA specific sources to lib${PROJECT_NAME}")
endif()

# Conditionally add Metal specific source files to the library
if(USE_METAL)
    message(STATUS "Interactions: Adding Metal specific sources to lib${PROJECT_NAME}")
    # Add Metal-specific interaction sources here if needed
endif()

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
