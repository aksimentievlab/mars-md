
# Add backend sources to parent LIB_SOURCES
list(APPEND LIB_SOURCES
    Backend/Resource.cpp
)

# Conditionally add MPI specific source files to the library
if(USE_MPI)
    list(APPEND LIB_SOURCES
        Backend/MPIManager.cpp
    )
    message(STATUS "Backend: Adding MPI specific sources to lib${PROJECT_NAME}")
endif()

# Conditionally add SYCL specific source files to the library
# We check if either SYCL ACPP or ICPX option is enabled from the top-level CMake.
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "Backend: Adding SYCL specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/SYCL/SYCLManager.cpp
    )
    # If your SYCL implementation uses a different file extension for device code
    # (though typically it's .cpp for SYCL), list them here too.
endif()

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)
    message(STATUS "Backend: Adding CUDA specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/CUDA/CUDAManager.cpp
        Backend/CUDA/KernelHelper.cu
    )
    # Ensure the CUDA language is enabled in your top-level project() call
    # when USE_CUDA is ON, so CMake knows how to compile .cu files.
    message(STATUS "Backend: CUDA is enabled for lib${PROJECT_NAME}. CUDA runtime linking may be handled by compiler or main executable.")
endif()

# Conditionally add Metal specific source files to the library
if(USE_METAL)
    message(STATUS "Backend: Adding Metal specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/METAL/METALManager.cpp
    )
    message(STATUS "Backend: Metal is enabled for lib${PROJECT_NAME}. Metal framework linking handled by main executable.")
endif()

