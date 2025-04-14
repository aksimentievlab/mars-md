
add_library("lib${PROJECT_NAME}"
  ARBDException.cpp
  Resource.cu
  GPUManager.cpp
  ParticlePatch.cpp
  #SimSystem.cpp
  Integrator.cpp
  Integrator/CPU.cpp
  Integrator/CUDA.cu
  Interaction.cpp
  Interaction/CPU.cpp
  Interaction/CUDA.cu
  SignalManager.cpp
  PatchOp.cu
)
if(USE_LOGGER)
  add_definitions(-DUSE_LOGGER)
  target_include_directories("lib${PROJECT_NAME}" PRIVATE ${spdlog_DIR})
  target_link_libraries("lib${PROJECT_NAME}" PRIVATE spdlog::spdlog_header_only)	
  # target_link_libraries("lib${PROJECT_NAME}" spdlog)	
  # target_link_libraries("${PROJECT_NAME}" spdlog)
  # find_package(fmt)
  # target_link_libraries("lib${PROJECT_NAME}" PRIVATE fmt::fmt)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${SPDLOG_LEVEL})
endif()
