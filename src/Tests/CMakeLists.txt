#find_package(Catch2 3 REQUIRED)
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# include(doctest)
include(Catch) # contains some cmake macros that will help with discovery of tests
include(CTest)

## TODO, add a macro to add these projects, build with CUDA
add_executable("arbd_tests"
matrix3.cu
vector3_precision.cu
bitmask.cu
array.cu
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set_property(TARGET arbd_tests PROPERTY CXX_STANDARD 14)
set_property(TARGET arbd_tests PROPERTY CUDA_STANDARD 14)
# target_link_libraries("arbd_tests" PRIVATE "lib${PROJECT_NAME}" Catch2::Catch2)
target_link_libraries("arbd_tests" PRIVATE "lib${PROJECT_NAME}" Catch2::Catch2WithMain)

## Add optional libraries
if(USE_CUDA)
  target_link_libraries(arbd_tests PUBLIC curand)
endif()
if(USE_NCCL)
  target_link_libraries(arbd_tests PUBLIC nccl)
endif()
if(USE_NVTX)
  target_link_libraries(arbd_tests PUBLIC nvToolsExt)
endif()
if(USE_LOGGER)
  add_definitions(-DUSE_LOGGER)
  target_include_directories(arbd_tests PRIVATE ${spdlog_DIR})
  target_link_libraries(arbd_tests PRIVATE spdlog::spdlog_header_only)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${SPDLOG_LEVEL})
endif()

## catch_discover_tests("${PROJECT_NAME}_TESTS")

catch_discover_tests(arbd_tests)
install(TARGETS arbd_tests)
