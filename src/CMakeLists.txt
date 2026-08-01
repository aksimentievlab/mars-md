# src/CMakeLists.txt
# Main CMakeLists.txt for the src directory
# Uses add_subdirectory() to include all subdirectories

# Initialize lists that will be populated by subdirectories
set(HOST_ONLY_HEADERS
    SignalManager.h
    SimManager.h
    ARBDException.h
    ARBDLogger.h
)
set(HOST_DEVICE_HEADERS
    "SimParam.h"
    "Header.h"
    "Constants.h"
)

set(DEVICE_ONLY_HEADERS "")

set(LIB_SOURCES "")

# SYCL-specific source lists (only used when SYCL is enabled)
if(USE_SYCL_ICPX OR USE_SYCL_ACPP)
    set(SYCL_DEVICE_SOURCES "")  # Sources with device kernels (need -fsycl -fsycl-targets)
    set(SYCL_HOST_SOURCES "")    # Sources using SYCL runtime API only (need -fsycl only)
endif()

# Root-level source files
list(APPEND LIB_SOURCES
    SignalManager.cpp
    SimManager.cpp
)

# Root-level headers
list(APPEND HOST_ONLY_HEADERS

)

# Add all subdirectories (order matters for dependencies)
add_subdirectory(Backend)
add_subdirectory(Types)
add_subdirectory(Objects)
add_subdirectory(Interactions)
add_subdirectory(IO)
add_subdirectory(System)
add_subdirectory(PatchOperation)

# Define the library with all collected sources
add_library("lib${PROJECT_NAME}" ${LIB_SOURCES})

# For SYCL builds, apply -fsycl-targets only to device sources
# Note: -fsycl is already set globally in root CMakeLists.txt
if(USE_SYCL_ICPX OR USE_SYCL_ACPP)
    # Some files are pure host code and should not be compiled with SYCL at all
    # These files don't include any SYCL headers or device code
    # NOTE: Most files need SYCL runtime support (Resource.h uses sycl::queue)
    # Only exclude files that don't use ANY backend abstractions
    set(PURE_HOST_SOURCES
        SignalManager.cpp
    )

    foreach(source ${PURE_HOST_SOURCES})
        get_filename_component(source_abs "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ABSOLUTE)
        if(EXISTS "${source_abs}")
            set_source_files_properties("${source_abs}"
                PROPERTIES
                COMPILE_FLAGS "-fno-sycl"
            )
        endif()
    endforeach()
    message(STATUS "Configuring SYCL compilation:")
    message(STATUS "  Device sources (with kernels): ${SYCL_DEVICE_SOURCES}")
    message(STATUS "  Host SYCL sources (runtime API): ${SYCL_HOST_SOURCES}")

    # Apply -fsycl-targets ONLY to device sources (files with kernels)
    # These files need device compilation
    foreach(source ${SYCL_DEVICE_SOURCES})
        get_filename_component(source_abs "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ABSOLUTE)
        if(EXISTS "${source_abs}")
            message(STATUS "  Marking ${source} for SYCL device compilation with targets")
            set_source_files_properties("${source_abs}"
                PROPERTIES
                COMPILE_FLAGS "${SYCL_DEVICE_EXTRA_FLAGS} -DSYCL_DEVICE_SOURCE"
            )
        else()
            message(WARNING "  SYCL device source not found: ${source_abs}")
        endif()
    endforeach()

    # Host SYCL sources just use global -fsycl (no extra flags needed)
    foreach(source ${SYCL_HOST_SOURCES})
        get_filename_component(source_abs "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ABSOLUTE)
        if(EXISTS "${source_abs}")
            message(STATUS "  Marking ${source} for SYCL host compilation")
            set_source_files_properties("${source_abs}"
                PROPERTIES
                COMPILE_FLAGS "-DSYCL_HOST_SOURCE"
            )
        else()
            message(WARNING "  SYCL host source not found: ${source_abs}")
        endif()
    endforeach()

    # All other sources compile with global -fsycl (for SYCL headers) but no device compilation

    # Add SYCL link flags to the library
    # Separate the flags properly - SYCL_LINK_FLAGS is a string, need to split it
    separate_arguments(SYCL_LINK_FLAGS_LIST UNIX_COMMAND "${SYCL_LINK_FLAGS}")
    target_link_options("lib${PROJECT_NAME}" PUBLIC ${SYCL_LINK_FLAGS_LIST})
endif()
target_link_libraries("lib${PROJECT_NAME}" PUBLIC nanovdb)

# Make the library depend on Metal shaders if Metal is enabled
if(USE_METAL AND TARGET metal_shaders_main)
    add_dependencies("lib${PROJECT_NAME}" metal_shaders_main)
endif()

# --- Include Directories ---
# CMAKE_CURRENT_SOURCE_DIR in this file refers to the "src" directory.
# This allows includes like #include "Backend/xx.h" from within any source
# file of this library, and also allows targets linking to this library to
# include public headers from "src/" (e.g., #include "ARBDException.h").
target_include_directories("lib${PROJECT_NAME}" PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- SYCL specific configuration for the library ---
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # For Intel DPC++ (icpx), linking is usually handled by -fsycl.
    message(STATUS "src: SYCL is enabled for lib${PROJECT_NAME}.")

    # Mark host-only headers to prevent device compilation
    foreach(header ${HOST_ONLY_HEADERS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${header}")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${header}"
                PROPERTIES
                HEADER_FILE_ONLY TRUE
                COMPILE_FLAGS "-DHOST_ONLY_HEADER"
            )
        endif()
    endforeach()

    # Ensure device-safe headers are available for device compilation
    foreach(header ${DEVICE_SAFE_HEADERS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${header}")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${header}"
                PROPERTIES
                HEADER_FILE_ONLY TRUE
                COMPILE_FLAGS "-DDEVICE_SAFE_HEADER"
            )
        endif()
    endforeach()

    # Explicit linking for AdaptiveCpp runtime libraries
    if(USE_SYCL_ACPP AND CMAKE_CXX_COMPILER MATCHES "acpp")
        # Find ACPP installation directory from the compiler path
        get_filename_component(ACPP_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ACPP_INSTALL_DIR "${ACPP_BIN_DIR}" DIRECTORY)
        set(ACPP_LIB_DIR "${ACPP_INSTALL_DIR}/lib")

        # Determine the appropriate library extension for the platform
        if(APPLE)
            set(ACPP_LIB_EXT "dylib")
        else()
            set(ACPP_LIB_EXT "so")
        endif()

        # Find and link the essential ACPP runtime libraries
        find_library(ACPP_RT_LIB NAMES "acpp-rt" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(ACPP_COMMON_LIB NAMES "acpp-common" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)

        if(ACPP_RT_LIB AND ACPP_COMMON_LIB)
            # Add the ACPP library directory to linker search paths
            target_link_directories("lib${PROJECT_NAME}" PUBLIC "${ACPP_LIB_DIR}")

            target_link_libraries("lib${PROJECT_NAME}" PUBLIC
                ${ACPP_RT_LIB}
                ${ACPP_COMMON_LIB}
            )
            message(STATUS "src: Successfully linked ACPP runtime libraries:")
            message(STATUS "  - ACPP Runtime: ${ACPP_RT_LIB}")
            message(STATUS "  - ACPP Common: ${ACPP_COMMON_LIB}")
        else()
            message(WARNING "Could not find required ACPP runtime libraries in ${ACPP_LIB_DIR}")
            message(WARNING "  - ACPP_RT_LIB: ${ACPP_RT_LIB}")
            message(WARNING "  - ACPP_COMMON_LIB: ${ACPP_COMMON_LIB}")
        endif()

        # Optional LLVM/Clang libraries - try to find them but don't fail if missing
        find_library(ACPP_CLANG_LIB NAMES "acpp-clang" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
        if(ACPP_CLANG_LIB)
            target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${ACPP_CLANG_LIB})
            message(STATUS "  - ACPP Clang: ${ACPP_CLANG_LIB}")

            # Link LLVM libraries required by libacpp-clang.so
            find_library(LLVM_LIB NAMES "LLVM-17" "LLVM" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
            find_library(CLANG_CPP_LIB NAMES "clang-cpp" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)

            if(LLVM_LIB)
                target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${LLVM_LIB})
                message(STATUS "  - LLVM: ${LLVM_LIB}")
            else()
                message(WARNING "Could not find LLVM library in ${ACPP_LIB_DIR}")
            endif()

            if(CLANG_CPP_LIB)
                target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${CLANG_CPP_LIB})
                message(STATUS "  - Clang C++: ${CLANG_CPP_LIB}")
            else()
                message(WARNING "Could not find Clang C++ library in ${ACPP_LIB_DIR}")
            endif()
        endif()

        message(STATUS "Configured lib${PROJECT_NAME} to avoid C++ standard flag conflicts with AdaptiveCpp")
    endif()

    # Explicit linking for Intel DPC++ runtime libraries
    if(USE_SYCL_ICPX AND CMAKE_CXX_COMPILER MATCHES "icpx")
        if(DEFINED ICPX_ROOT)
            set(ICPX_LIB_DIR "${ICPX_ROOT}/lib")
        else()
            get_filename_component(ICPX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
            get_filename_component(ICPX_INSTALL_DIR "${ICPX_BIN_DIR}" DIRECTORY)
            set(ICPX_LIB_DIR "${ICPX_INSTALL_DIR}/lib")
        endif()

        # Find and link the essential Intel DPC++ SYCL runtime libraries
        find_library(SYCL_LIB NAMES "sycl" PATHS "${ICPX_LIB_DIR}" NO_DEFAULT_PATH)

        if(SYCL_LIB)
            # Add the Intel DPC++ library directory to linker search paths
            target_link_directories("lib${PROJECT_NAME}" PUBLIC "${ICPX_LIB_DIR}")

            target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${SYCL_LIB})
            message(STATUS "src: Successfully linked Intel DPC++ SYCL runtime library:")
            message(STATUS "  - SYCL Runtime: ${SYCL_LIB}")
        else()
            message(WARNING "Could not find required Intel DPC++ SYCL runtime library in ${ICPX_LIB_DIR}")
            message(WARNING "  - SYCL_LIB: ${SYCL_LIB}")
        endif()

        # Link OpenCL for Intel GPU support
        find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
        if(OPENCL_LIB)
            target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${OPENCL_LIB})
            message(STATUS "  - OpenCL: ${OPENCL_LIB}")
        else()
            message(WARNING "Could not find OpenCL library for Intel DPC++")
        endif()

        message(STATUS "Configured lib${PROJECT_NAME} with Intel DPC++ SYCL runtime linking")
    endif()

    # Link OpenMP for AdaptiveCpp
    if(USE_SYCL_ACPP AND OpenMP_CXX_FOUND)
        target_link_libraries("lib${PROJECT_NAME}" PUBLIC OpenMP_Custom)
        message(STATUS "src: Linked custom OpenMP to lib${PROJECT_NAME} for AdaptiveCpp backend")
    endif()
endif()

# --- CUDA specific linking for the library ---
if(USE_CUDA)
    # Set CUDA properties for .cu files
    # Note: Individual .cu files should set their properties in their respective CMakeLists.txt
    # This is a global setting for the library
    set_target_properties("lib${PROJECT_NAME}" PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86"
    )

    # Find and link CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries("lib${PROJECT_NAME}"
        PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    )

    # Explicitly link C++ standard library for CUDA targets
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries("lib${PROJECT_NAME}" PRIVATE stdc++ gcc_s gcc m pthread dl)
        target_link_options("lib${PROJECT_NAME}" PRIVATE -Wl,--no-as-needed)
    else()
        target_link_libraries("lib${PROJECT_NAME}" PRIVATE stdc++)
    endif()

    message(STATUS "src: CUDA configured with separable compilation and toolkit linking")
endif()

# --- Metal specific linking and configuration for the library ---
if(USE_METAL)
    # Link Metal, Foundation, and QuartzCore frameworks to the library
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND QUARTZCORE_FRAMEWORK)
        target_link_libraries("lib${PROJECT_NAME}"
            PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${QUARTZCORE_FRAMEWORK}
        )
        message(STATUS "src: Linked Metal frameworks to lib${PROJECT_NAME}")
    else()
        message(FATAL_ERROR "Required frameworks not found for lib${PROJECT_NAME}")
    endif()

    # Add metal-cpp include directories
    target_include_directories("lib${PROJECT_NAME}"
        PUBLIC
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )

    # Add Metal-specific compile definitions
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC USE_METAL)

    # Ensure ARC is enabled for Objective-C++ files
    target_compile_options("lib${PROJECT_NAME}" PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )

    message(STATUS "src: Metal is enabled for lib${PROJECT_NAME}. Framework linking and ARC configured.")
endif()

# --- MPI specific linking and configuration for the library ---
if(USE_MPI)
    # Link MPI to the library
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        # When using MPI compiler wrapper, linking is automatic
        message(STATUS "src: MPI linking handled automatically by compiler wrapper for lib${PROJECT_NAME}")
    else()
        # Manual MPI linking when not using wrapper
        target_link_libraries("lib${PROJECT_NAME}"
            PUBLIC
            MPI::MPI_CXX
        )
        target_include_directories("lib${PROJECT_NAME}"
            PUBLIC
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "src: Manually linked MPI libraries to lib${PROJECT_NAME}")
    endif()

    # Add MPI compile definitions to the library
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC USE_MPI)

    message(STATUS "src: MPI is enabled for lib${PROJECT_NAME}. MPI linking configured.")
endif()
