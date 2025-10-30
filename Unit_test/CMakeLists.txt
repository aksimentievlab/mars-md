# Unit_test/CMakeLists.txt
# Minimal backend-focused tests (no references to test_unit.old)

cmake_minimum_required(VERSION 3.20)

include(CTest)
enable_testing()

# Single unified test binary using Catch2 amalgamated
add_executable(arbd_unit_tests
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Backend/backend_buffer_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Backend/mpi_tests.cpp
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

# Register tests with labels
add_test(NAME unit_device COMMAND arbd_unit_tests [device])
set_tests_properties(unit_device PROPERTIES LABELS "device")

if(USE_MPI)
	add_test(NAME unit_mpi_2ranks
		COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
				$<TARGET_FILE:arbd_unit_tests> [mpi] ${MPIEXEC_POSTFLAGS})
	set_tests_properties(unit_mpi_2ranks PROPERTIES LABELS "mpi")
endif()
