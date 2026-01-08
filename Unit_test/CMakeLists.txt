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
		${CMAKE_CURRENT_LIST_DIR}/Types/basegrid_device_tests.cpp
		${CMAKE_CURRENT_LIST_DIR}/Objects/Particles.cpp
		${CMAKE_CURRENT_LIST_DIR}/TestParticle.cpp
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
endif()

if(USE_MPI)
	find_package(MPI REQUIRED)
	target_link_libraries(arbd_unit_tests PRIVATE MPI::MPI_CXX)
endif()

# ZOrder tests executable
add_executable(arbd_zorder_tests
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    ${CMAKE_CURRENT_LIST_DIR}/System/zorder_tests.cpp
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

# Register tests with labels
add_test(NAME unit_device COMMAND arbd_unit_tests [device])
set_tests_properties(unit_device PROPERTIES LABELS "device")

add_test(NAME unit_zorder COMMAND arbd_zorder_tests [zorder])
set_tests_properties(unit_zorder PROPERTIES LABELS "zorder")

add_test(NAME unit_zorder_system COMMAND arbd_zorder_tests [zorder][system])
set_tests_properties(unit_zorder_system PROPERTIES LABELS "zorder;system")

add_test(NAME unit_zorder_pairlist COMMAND arbd_zorder_tests [zorder][pairlist])
set_tests_properties(unit_zorder_pairlist PROPERTIES LABELS "zorder;pairlist")

if(USE_MPI)
	add_test(NAME unit_mpi_2ranks
		COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
				$<TARGET_FILE:arbd_unit_tests> [mpi] ${MPIEXEC_POSTFLAGS})
	set_tests_properties(unit_mpi_2ranks PROPERTIES LABELS "mpi")
endif()
