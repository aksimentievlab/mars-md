# Interactive Testing Module
# Provides both C++ and Python interfaces for particle/system testing

message(STATUS "Configuring interactive particle/system tests")

# Interactive particle testing executable (C++)
add_executable(arbd_interactive_tests
    particle_tests.cpp
)

target_include_directories(arbd_interactive_tests PRIVATE ${COMMON_TEST_INCLUDE_DIRS})
target_link_libraries(arbd_interactive_tests PRIVATE ${COMMON_TEST_LIBS})

# Backend-specific configuration
if(USE_CUDA)
    set_property(TARGET arbd_interactive_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(arbd_interactive_tests PRIVATE USE_CUDA)
endif()

if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    target_compile_definitions(arbd_interactive_tests PRIVATE USE_SYCL)
    if(USE_SYCL_ACPP)
        target_compile_definitions(arbd_interactive_tests PRIVATE PROJECT_USES_SYCL_ACPP)
    endif()
    if(USE_SYCL_ICPX)
        target_compile_definitions(arbd_interactive_tests PRIVATE PROJECT_USES_SYCL_ICPX)
    endif()
endif()

if(USE_METAL)
    target_compile_definitions(arbd_interactive_tests PRIVATE USE_METAL)
    target_include_directories(arbd_interactive_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )
    # Make test depend on Metal shaders
    if(TARGET metal_shaders_main)
        add_dependencies(arbd_interactive_tests metal_shaders_main)
    endif()
endif()

if(USE_MPI)
    target_compile_definitions(arbd_interactive_tests PRIVATE USE_MPI)
    if(NOT CMAKE_CXX_COMPILER MATCHES "mpi")
        target_include_directories(arbd_interactive_tests PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    endif()
endif()

# Add to interactive_tests target
add_dependencies(interactive_tests arbd_interactive_tests)

# Create CTest
add_test(NAME InteractiveParticleTests COMMAND arbd_interactive_tests)
set_tests_properties(InteractiveParticleTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    LABELS "interactive;particles"
)

# PyBind module for interactive Python testing
if(PYBIND_AVAILABLE)
    message(STATUS "Creating Python interactive testing module")

    pybind11_add_module(arbd_interactive particle_tests.cpp)
    target_include_directories(arbd_interactive PRIVATE ${COMMON_TEST_INCLUDE_DIRS})
    target_link_libraries(arbd_interactive PRIVATE ${COMMON_TEST_LIBS})
    target_compile_definitions(arbd_interactive PRIVATE USE_PYBIND)

    # Backend-specific configuration for Python module
    if(USE_CUDA)
        target_compile_definitions(arbd_interactive PRIVATE USE_CUDA)
    endif()

    if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
        target_compile_definitions(arbd_interactive PRIVATE USE_SYCL)
        if(USE_SYCL_ACPP)
            target_compile_definitions(arbd_interactive PRIVATE PROJECT_USES_SYCL_ACPP)
        endif()
        if(USE_SYCL_ICPX)
            target_compile_definitions(arbd_interactive PRIVATE PROJECT_USES_SYCL_ICPX)
        endif()
    endif()

    if(USE_METAL)
        target_compile_definitions(arbd_interactive PRIVATE USE_METAL)
        target_include_directories(arbd_interactive PRIVATE
            ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
            ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
        )
        if(TARGET metal_shaders_main)
            add_dependencies(arbd_interactive metal_shaders_main)
        endif()
    endif()

    if(USE_MPI)
        target_compile_definitions(arbd_interactive PRIVATE USE_MPI)
        if(NOT CMAKE_CXX_COMPILER MATCHES "mpi")
            target_include_directories(arbd_interactive PRIVATE ${MPI_CXX_INCLUDE_DIRS})
        endif()
    endif()

    # Set module properties
    set_target_properties(arbd_interactive PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python
    )

    # Python test discovery (if pytest is available)
    find_program(PYTEST_EXECUTABLE pytest)
    if(PYTEST_EXECUTABLE)
        add_test(NAME PythonInteractiveTests
                 COMMAND ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/ -v
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(PythonInteractiveTests PROPERTIES
            LABELS "interactive;python;particles"
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:$ENV{PYTHONPATH}"
        )
    endif()

    message(STATUS "Python interactive module: arbd_interactive")
else()
    message(STATUS "PyBind not available - C++ interactive tests only")
endif()

message(STATUS "Interactive testing configured successfully")