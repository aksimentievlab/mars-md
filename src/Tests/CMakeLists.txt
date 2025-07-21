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
    add_executable(arbd_test_metal
        Backends/metal_backend.mm #Still not working
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_metal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
    target_link_libraries(arbd_test_metal PRIVATE "lib${PROJECT_NAME}" ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

    add_test(NAME Metal_Tests COMMAND arbd_test_metal)
    set_tests_properties(Metal_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )

    install(TARGETS arbd_test_metal)

    add_executable(arbd_test_metal_math
        Math/metal_bitmask.mm
        Kernels/metal_events.mm
        Kernels/metal_test.mm
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_metal_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
    target_link_libraries(arbd_test_metal_math PRIVATE "lib${PROJECT_NAME}" ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

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
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl PRIVATE "lib${PROJECT_NAME}")

    add_executable(arbd_test_sycl_backend
        Backends/sycl_manager.cpp
        Backends/sycl_backend.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl_backend PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl_backend PRIVATE "lib${PROJECT_NAME}")

    add_executable(arbd_test_sycl_math
        Math/sycl_bitmask.cpp
        Math/sycl_random.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl_math PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl_math PRIVATE "lib${PROJECT_NAME}")

    if(USE_SYCL_ACPP)
        target_link_libraries(arbd_test_sycl PRIVATE stdc++)
        target_link_libraries(arbd_test_sycl_backend PRIVATE stdc++)
        # Also link Boost libraries for AdaptiveCpp
        target_link_directories(arbd_test_sycl PRIVATE "${BOOST_LIB_DIR}")
        target_link_directories(arbd_test_sycl_backend PRIVATE "${BOOST_LIB_DIR}")
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
        matrix3.cu
        vector3_precision.cu
        bitmask.cu
        array.cu
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    target_include_directories(arbd_test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_cuda PRIVATE ${COMMON_TEST_LIBS})
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
