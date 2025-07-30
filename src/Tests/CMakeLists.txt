#find_package(Catch2 3 REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2/extras)
include(Catch) # contains some cmake macros that will help with discovery of tests
include(CTest)


# Common libraries and include directories for all tests
set(COMMON_TEST_LIBS "lib${PROJECT_NAME}" Catch2::Catch2WithMain)
if(USE_CUDA)
    list(APPEND COMMON_TEST_LIBS curand)
endif()
if(USE_NCCL)
    list(APPEND COMMON_TEST_LIBS nccl)
endif()

# Metal Tests
if(USE_METAL AND APPLE)
    # Common Metal test settings
    set(METAL_TEST_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )
    set(METAL_TEST_LIBS
        "lib${PROJECT_NAME}"
        METAL_CPP
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
    )
    # Horrible METAL library loding system, just use the default.metallib
    # Compile Metal shaders
    # set(METAL_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Backends)
    # set(METAL_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    #
    # # Find all .metal files
    # file(GLOB METAL_SOURCES ${METAL_SHADER_DIR}/*.metal)
    #
    # # Create a list to hold the compiled .air files
    # set(METAL_AIR_FILES)
    #
    # # Compile each .metal file to .air
    # foreach(METAL_SOURCE ${METAL_SOURCES})
    #     get_filename_component(METAL_NAME ${METAL_SOURCE} NAME_WE)
    #     set(AIR_FILE ${METAL_OUTPUT_DIR}/${METAL_NAME}.air)
    #
    #     add_custom_command(
    #         OUTPUT ${AIR_FILE}
    #         COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCE} -o ${AIR_FILE}
    #         DEPENDS ${METAL_SOURCE}
    #         COMMENT "Compiling Metal shader: ${METAL_NAME}.metal"
    #         VERBATIM
    #     )
    #
    #     list(APPEND METAL_AIR_FILES ${AIR_FILE})
    # endforeach()
    #
    # # Create the Metal library
    # set(METAL_LIBRARY ${METAL_OUTPUT_DIR}/default.metallib)
    # add_custom_command(
    #     OUTPUT ${METAL_LIBRARY}
    #     COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METAL_LIBRARY}
    #     DEPENDS ${METAL_AIR_FILES}
    #     COMMENT "Creating Metal library: default.metallib"
    #     VERBATIM
    # )
    #
    # # Create a custom target for Metal shaders
    # add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIBRARY})
    #
    # # Copy the Metal library to the test working directory
    # add_custom_command(
    #     TARGET metal_shaders POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy ${METAL_LIBRARY} ${CMAKE_BINARY_DIR}/src/Tests/
    #     COMMENT "Copying Metal library to test directory"
    #     VERBATIM
    # )

    # Metal backend tests
    add_executable(arbd_test_metal
        Backends/metal_manager.cpp
        Backends/metal_backend.cpp
        Backends/generic_Buffer.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_metal PRIVATE ${METAL_TEST_INCLUDES})
    target_link_libraries(arbd_test_metal PRIVATE ${METAL_TEST_LIBS})

    # # Make sure Metal shaders are built before the test executable
    # add_dependencies(arbd_test_metal metal_shaders)

    add_test(NAME Metal_Tests COMMAND arbd_test_metal)
    set_tests_properties(Metal_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )

    install(TARGETS arbd_test_metal)
    # Metal math tests
    add_executable(arbd_test_metal_math
        #Math/metal_bitmask.cpp
        #All of the above works
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_metal_math PRIVATE ${METAL_TEST_INCLUDES})
    target_link_libraries(arbd_test_metal_math PRIVATE ${METAL_TEST_LIBS})

    add_test(NAME Metal_Math_Tests COMMAND arbd_test_metal_math)
    set_tests_properties(Metal_Math_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )

    install(TARGETS arbd_test_metal_math)
    message(STATUS "Added Metal math test executable: arbd_test_metal_math")
elseif(USE_METAL)
    message(WARNING "USE_METAL is ON but not on Apple platform - Metal tests will be excluded")
endif()

# SYCL Tests
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # Test for test_sycl.cpp - now uses Catch2's main function
    add_executable(arbd_test_sycl
        Backends/generic_Buffer.cpp
        Kernels/sycl_kernels.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

    add_executable(arbd_test_sycl_backend
        Backends/sycl_backend.cpp
        Backends/sycl_manager.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl_backend PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl_backend PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

    add_executable(arbd_test_sycl_math
        #Math/sycl_bitmask.cpp
        Math/RandomDevice.cpp #works on mac acpp
        Math/RandomProfile.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl_math PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl_math PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

    if(USE_SYCL_ACPP)
        target_link_libraries(arbd_test_sycl PRIVATE stdc++)
        target_link_libraries(arbd_test_sycl_backend PRIVATE stdc++)
        target_link_libraries(arbd_test_sycl_math PRIVATE stdc++)
        # Also link Boost libraries for AdaptiveCpp
        target_link_directories(arbd_test_sycl PRIVATE "${BOOST_LIB_DIR}")
        target_link_directories(arbd_test_sycl_backend PRIVATE "${BOOST_LIB_DIR}")
        target_link_directories(arbd_test_sycl_math PRIVATE "${BOOST_LIB_DIR}")
    endif()

    # Intel DPC++ specific linking for test executables
    if(USE_SYCL_ICPX AND CMAKE_CXX_COMPILER MATCHES "icpx")
        # Find Intel OneAPI SYCL installation directory
        get_filename_component(ICPX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ICPX_INSTALL_DIR "${ICPX_BIN_DIR}" DIRECTORY)
        set(ICPX_LIB_DIR "${ICPX_INSTALL_DIR}/lib")

        # Find and link the essential Intel DPC++ SYCL runtime libraries for tests
        find_library(SYCL_LIB NAMES "sycl" PATHS "${ICPX_LIB_DIR}" NO_DEFAULT_PATH)

        if(SYCL_LIB)
            # Add the Intel DPC++ library directory to linker search paths for tests
            target_link_directories(arbd_test_sycl PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(arbd_test_sycl_backend PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(arbd_test_sycl_math PUBLIC "${ICPX_LIB_DIR}")

            # Link SYCL runtime library to all SYCL test executables
            target_link_libraries(arbd_test_sycl PRIVATE ${SYCL_LIB})
            target_link_libraries(arbd_test_sycl_backend PRIVATE ${SYCL_LIB})
            target_link_libraries(arbd_test_sycl_math PRIVATE ${SYCL_LIB})

            message(STATUS "Linked Intel DPC++ SYCL runtime to test executables")
        else()
            message(WARNING "Could not find Intel DPC++ SYCL runtime library for tests in ${ICPX_LIB_DIR}")
        endif()

        # Link OpenCL for Intel GPU support in tests
        find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
        if(OPENCL_LIB)
            target_link_libraries(arbd_test_sycl PRIVATE ${OPENCL_LIB})
            target_link_libraries(arbd_test_sycl_backend PRIVATE ${OPENCL_LIB})
            target_link_libraries(arbd_test_sycl_math PRIVATE ${OPENCL_LIB})
            message(STATUS "Linked OpenCL to SYCL test executables")
        endif()
    endif()

    # Instead of catch_discover_tests which causes bus errors on ACPP,
    # we'll add a single test that runs the entire executable
    add_test(NAME SYCL_Tests COMMAND arbd_test_sycl)

    # Set environment variables for SYCL test execution
    set_tests_properties(SYCL_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )

    install(TARGETS arbd_test_sycl)
    message(STATUS "Added SYCL test executable: arbd_test_sycl")

    # Add test for backend tests
    add_test(NAME SYCL_Backend_Tests COMMAND arbd_test_sycl_backend)
    set_tests_properties(SYCL_Backend_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )

    install(TARGETS arbd_test_sycl_backend)
    message(STATUS "Added SYCL backend test executable: arbd_test_sycl_backend")

    add_test(NAME SYCL_Math_Tests COMMAND arbd_test_sycl_math)
    set_tests_properties(SYCL_Math_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )

    install(TARGETS arbd_test_sycl_math)
    message(STATUS "Added SYCL math test executable: arbd_test_sycl_math")

endif()

# CUDA Tests
if(USE_CUDA)
    add_executable(arbd_test_cuda
        #Backends/cuda_backend.cpp
        Backends/generic_Buffer.cpp
        Kernels/cuda_kernels.cu
        Kernels/cuda_kernels.cpp
        Math/Kernel_for_test.h
        Math/RandomTestKernels.cu
        Math/RandomDevice.cpp
        Math/RandomProfile.cpp
        #Math/vector3_precision.cu
        #Math/bitmask.cu
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --allow-unsupported-compiler -Xcompiler=-Wno-deprecated-declarations")
    target_include_directories(arbd_test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_cuda PRIVATE "lib${PROJECT_NAME}")

    # Link CUDA runtime libraries
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(arbd_test_cuda PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        curand)
    set_source_files_properties(
            Math/RandomTestKernels.cu
            Math/Kernel_for_test.h
            PROPERTIES
            LANGUAGE CUDA
            CUDA_SEPARABLE_COMPILATION ON
        )
    if(USE_NCCL)
        target_link_libraries(arbd_test_cuda PRIVATE nccl)
    endif()

    catch_discover_tests(arbd_test_cuda)
    install(TARGETS arbd_test_cuda)
    message(STATUS "Added CUDA test executable: arbd_test_cuda")
endif()

if(USE_MPI)
    add_executable(arbd_test_mpi mpitest.cpp)
    target_include_directories(arbd_test_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_mpi PRIVATE ${COMMON_TEST_LIBS})

    # Ensure MPI test uses only CXX, not CUDA compiler
    set_target_properties(arbd_test_mpi PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    # Explicitly set this as a CXX-only target to avoid CUDA flag inheritance
    set_source_files_properties(mpitest.cpp PROPERTIES LANGUAGE CXX)

    # Fix C++ standard library linking issues (same as main executable)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(arbd_test_mpi PRIVATE -Wl,--whole-archive stdc++ -Wl,--no-whole-archive)
        target_link_libraries(arbd_test_mpi PRIVATE gcc_s gcc m pthread dl)
        target_link_options(arbd_test_mpi PRIVATE -Wl,--no-as-needed)
    endif()

    catch_discover_tests(arbd_test_mpi)
    install(TARGETS arbd_test_mpi)
    message(STATUS "Added MPI test executable: arbd_test_mpi")
endif()
