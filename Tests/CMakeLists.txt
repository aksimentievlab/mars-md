cmake_minimum_required(VERSION 3.25.5 FATAL_ERROR)

# Interactive and Integration Testing Suite for ARBD
# This provides comprehensive testing beyond unit tests

message(STATUS "Configuring ARBD Interactive/Integration Tests")

# Test configuration options
option(BUILD_INTERACTIVE_TESTS "Build interactive testing with PyBind" ON)
option(BUILD_INTEGRATION_TESTS "Build integration tests" ON)
option(BUILD_PERFORMANCE_TESTS "Build performance tests" ON)

# Find required packages for testing
if(BUILD_INTERACTIVE_TESTS AND USE_PYBIND)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "PyBind11 found - interactive Python tests will be available")
        set(PYBIND_AVAILABLE TRUE)
    else()
        message(WARNING "PyBind11 not found - interactive tests will be C++ only")
        set(PYBIND_AVAILABLE FALSE)
    endif()
else()
    set(PYBIND_AVAILABLE FALSE)
endif()

# Common test settings
set(COMMON_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extern
)

set(COMMON_TEST_LIBS "lib${PROJECT_NAME}")

# Add backend-specific libraries if enabled
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND COMMON_TEST_LIBS
        CUDA::cudart
        CUDA::cuda_driver
        curand)
endif()

if(USE_NCCL AND USE_CUDA)
    list(APPEND COMMON_TEST_LIBS nccl)
endif()

if(USE_METAL)
    list(APPEND COMMON_TEST_LIBS
        METAL_CPP
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK})
endif()

if(USE_MPI)
    if(NOT CMAKE_CXX_COMPILER MATCHES "mpi")
        list(APPEND COMMON_TEST_LIBS MPI::MPI_CXX)
    endif()
endif()

# Custom test targets for organization
add_custom_target(interactive_tests
    COMMENT "Building interactive testing suite")
add_custom_target(integration_tests
    COMMENT "Building integration testing suite")
add_custom_target(performance_tests
    COMMENT "Building performance testing suite")
add_custom_target(all_comprehensive_tests
    COMMENT "Building all comprehensive tests")

# Add test subdirectories
if(BUILD_INTERACTIVE_TESTS)
    add_subdirectory(interactive)
    add_dependencies(all_comprehensive_tests interactive_tests)
    message(STATUS "Interactive tests enabled")
endif()

if(BUILD_INTEGRATION_TESTS)
    add_subdirectory(private_test)
    add_dependencies(all_comprehensive_tests integration_tests)
    message(STATUS "Integration tests enabled")
endif()

if(BUILD_PERFORMANCE_TESTS)
    add_subdirectory(performance)
    add_dependencies(all_comprehensive_tests performance_tests)
    message(STATUS "Performance tests enabled")
endif()

# Test discovery and CTest integration
enable_testing()

# Global test properties
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # Set SYCL-specific test environment
    set_property(TEST PROPERTY
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1")
endif()

if(USE_MPI)
    # Set MPI-specific test environment for OpenMPI
    set_property(TEST PROPERTY
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
endif()

# Summary
message(STATUS "Tests Configuration Summary:")
message(STATUS "  Interactive tests: ${BUILD_INTERACTIVE_TESTS}")
message(STATUS "  Integration tests: ${BUILD_INTEGRATION_TESTS}")
message(STATUS "  Performance tests: ${BUILD_PERFORMANCE_TESTS}")
message(STATUS "  PyBind available: ${PYBIND_AVAILABLE}")
message(STATUS "  Backend: ${CMAKE_CXX_COMPILER_ID}")
if(USE_CUDA)
    message(STATUS "  CUDA support: ON")
endif()
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "  SYCL support: ON")
endif()
if(USE_METAL)
    message(STATUS "  Metal support: ON")
endif()
if(USE_MPI)
    message(STATUS "  MPI support: ON")
endif()
