# Unit_test/CMakeLists.txt
# Minimal backend-focused tests (no references to test_unit.old)

cmake_minimum_required(VERSION 3.20)

include(CTest)
enable_testing()

# Single unified test binary using Catch2 amalgamated
add_executable(arbd_unit_tests
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Backend/backend_buffer_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Types/vector3_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Types/matrix3_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Types/basegrid_device_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Types/grid_boundary_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Types/grid_arithmetic_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Objects/Particles.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Objects/RigidBodies.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Objects/RigidBodyForcePairs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Objects/RigidBodyManager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Objects/RigidBodyExternalForces.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Objects/TestParticle.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Compute/BDKernels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Compute/StochasticIntegrators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/Forces.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/TabulatedBonded.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/AnalyticalNB.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/NonbondedEnergy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/GridGrid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/RigidBodyGridBatch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/RigidBodyParticleGridBatch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/Pmf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/Convolution.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Interactions/GenPotGolden.cpp
    ${CMAKE_CURRENT_LIST_DIR}/IO/RigidBodyConfigParsing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/System/SingleGPU.cpp
    ${CMAKE_CURRENT_LIST_DIR}/System/SinglePatchSimulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/System/SinglePatchRigidBodySimulation.cpp
)
# Includes
target_include_directories(arbd_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}
)

# Link against project library and optional backends
target_link_libraries(arbd_unit_tests PRIVATE lib${PROJECT_NAME})

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(arbd_unit_tests PRIVATE CUDA::cudart)

    # basegrid_device_tests.cpp launches a named GridQueryKernel functor (see
    # Types/basegrid_device.h) rather than an ad hoc lambda, so its
    # launch_cuda_kernel instantiation can be forced into a real CUDA
    # compilation unit here instead of getting the host stub.
    target_sources(arbd_unit_tests PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/Types/basegrid_device_instantiations.cu
    )
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/Types/basegrid_device_instantiations.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(arbd_unit_tests PRIVATE MPI::MPI_CXX)
endif()

# ZOrder tests executable
add_executable(arbd_zorder_tests
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    ${CMAKE_CURRENT_LIST_DIR}/System/zorder_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/System/reorder_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Compute/Zorder_pairlist2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Compute/zorder_pairlist_tests.cpp
)

# Includes for ZOrder tests
target_include_directories(arbd_zorder_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}
)

# Link against project library and optional backends for ZOrder tests
target_link_libraries(arbd_zorder_tests PRIVATE lib${PROJECT_NAME})

if(USE_CUDA)
	target_link_libraries(arbd_zorder_tests PRIVATE CUDA::cudart)
endif()

if(USE_MPI)
	target_link_libraries(arbd_zorder_tests PRIVATE MPI::MPI_CXX)
endif()

# Ensure SYCL tests are compiled with device targets so kernels are available
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    separate_arguments(SYCL_DEVICE_EXTRA_FLAGS_LIST UNIX_COMMAND "${SYCL_DEVICE_EXTRA_FLAGS}")
    target_compile_options(arbd_unit_tests PRIVATE ${SYCL_DEVICE_EXTRA_FLAGS_LIST} -DSYCL_DEVICE_SOURCE)
    target_compile_options(arbd_zorder_tests PRIVATE ${SYCL_DEVICE_EXTRA_FLAGS_LIST} -DSYCL_DEVICE_SOURCE)
endif()


if(USE_MPI)
    add_test(NAME unit_mpi_2ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:arbd_unit_tests> [mpi] ${MPIEXEC_POSTFLAGS})
    set_tests_properties(unit_mpi_2ranks PROPERTIES LABELS "mpi")
endif()

# Handle both user comma lists, space lists, and CMake native semicolon lists safely
string(REPLACE ";" "," UNIT_TEST_DEVICE_ARRAY_PP "${UNIT_TEST_DEVICE_ARRAY}")
string(REPLACE " " "," UNIT_TEST_DEVICE_ARRAY_PP "${UNIT_TEST_DEVICE_ARRAY_PP}")

if(NOT "${UNIT_TEST_DEVICE_ID}" STREQUAL "")
    set(UNIT_TEST_DEVICE_ID_DEFINE "#define UNIT_TEST_DEVICE_ID ${UNIT_TEST_DEVICE_ID}")
else()
    set(UNIT_TEST_DEVICE_ID_DEFINE "// UNIT_TEST_DEVICE_ID unset: catch_boiler.h uses device_ids[0]")
endif()

# Generated header instead of -D: nvcc splits -D values on commas, so a
# command-line list expands to a bogus second macro name.
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/unit_test_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/unit_test_config.h
    @ONLY
)

add_library(unit_test_config INTERFACE)
target_include_directories(unit_test_config INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# The rest of your target links remain the same
target_link_libraries(arbd_unit_tests PRIVATE unit_test_config)
target_link_libraries(arbd_zorder_tests PRIVATE unit_test_config)
message(STATUS "Unit tests: available resources={${UNIT_TEST_DEVICE_ARRAY_PP}}")
