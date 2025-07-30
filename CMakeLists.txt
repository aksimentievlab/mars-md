cmake_minimum_required(VERSION 3.25.5 FATAL_ERROR)
set(PROJECT_NAME_BASE "arbd")
if(DEFINED PROJECT_NAME_SUFFIX)
    set(PROJECT_NAME "${PROJECT_NAME_BASE}${PROJECT_NAME_SUFFIX}")
else()
    set(PROJECT_NAME "${PROJECT_NAME_BASE}")
endif()

# Single project() call
if(USE_CUDA)
    if(USE_METAL)
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX CUDA OBJCXX)
    else()
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX CUDA)
    endif()
else()
    if(USE_METAL)
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX OBJCXX)
    else()
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX)
    endif()
endif()

# --- Project Options ---
option(USE_CUDA "Use CUDA" OFF)
option(USE_SYCL_ACPP "Use SYCL with AdaptiveCpp (acpp)" OFF)
option(USE_SYCL_ICPX "Use SYCL with Intel DPC++ (icpx)" OFF)
option(DEBUG_BUILD "Build with debug flags (overrides CMAKE_BUILD_TYPE)" OFF)
option(USE_NCCL "Use NCCL for single node GPU peer communication" OFF)
option(USE_NVTX "Build with NVTX profiling ranges" OFF)
option(USE_PYBIND "Build Python bindings" OFF)
option(USE_METAL "Use Apple Silicon Metal" OFF)

# --- Basic CMake Setup ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define C++23 macros as empty to avoid compatibility issues
add_compile_definitions(_GLIBCXX23_CONSTEXPR=)

# Generate compile_commands.json for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find threading library for use across the project
find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/extern/OpenRAND/include)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Set CUDA architectures for compatibility with modern GPUs
    # Include common architectures: 7.0 (V100), 7.5 (T4), 8.0 (A100), 8.6 (A40), 8.9 (RTX 4090)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")

    # Add CUDA compiler flags for compatibility and force older GCC
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

    message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# --- Optimization Flags Configuration ---
# Set optimization level (can be overridden with -DOPT_LEVEL=2 or -DOPT_LEVEL=3)
if(NOT DEFINED OPT_LEVEL)
    set(OPT_LEVEL 3)
endif()

message(STATUS "Setting optimization level to -O${OPT_LEVEL}")

# Configure optimization flags for different compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # Add additional GCC-specific optimizations
    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native")
    endif()

    if(USE_CUDA AND CMAKE_CUDA_FLAGS_RELEASE)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang/AppleClang
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # Add additional Clang-specific optimizations
    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native")
    endif()

    if(USE_CUDA AND CMAKE_CUDA_FLAGS_RELEASE)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel compilers (icpx, icc)
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # Add Intel-specific optimizations
    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -xHost")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS_RELEASE "/O${OPT_LEVEL} /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O${OPT_LEVEL} /Zi /DNDEBUG")

else()
    # Generic fallback
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")
endif()

# Special handling for SYCL compilers (acpp, icpx)
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # SYCL compilers need optimization flags to be explicitly set
    message(STATUS "SYCL detected: Ensuring optimization flags are properly set")
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # For ACPP specifically, ensure optimization flags are added to general flags too
    if(USE_SYCL_ACPP)
        # Add optimization flags to general CXX flags to ensure they're always present
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O${OPT_LEVEL}")
        message(STATUS "ACPP: Added -O${OPT_LEVEL} to CMAKE_CXX_FLAGS")
    endif()
endif()

# Metal (Objective-C++) optimization flags
if(USE_METAL AND APPLE)
    set(CMAKE_OBJCXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_OBJCXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_OBJCXX_FLAGS_RELEASE "${CMAKE_OBJCXX_FLAGS_RELEASE} -march=native -mtune=native")
        set(CMAKE_OBJCXX_FLAGS_RELWITHDEBINFO "${CMAKE_OBJCXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native")
    endif()
endif()

# Ensure optimization flags are applied globally (including to external dependencies)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # Add optimization flags to the general CMAKE_CXX_FLAGS to ensure they're always present
    string(FIND "${CMAKE_CXX_FLAGS}" "-O" OPT_FLAG_FOUND)
    if(OPT_FLAG_FOUND EQUAL -1)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O${OPT_LEVEL}")
        message(STATUS "Added -O${OPT_LEVEL} to CMAKE_CXX_FLAGS for global optimization")
    endif()
endif()

message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ General flags: ${CMAKE_CXX_FLAGS}")
if(USE_CUDA)
    message(STATUS "CUDA Release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()
if(USE_METAL)
    message(STATUS "Objective-C++ Release flags: ${CMAKE_OBJCXX_FLAGS_RELEASE}")
endif()

# Metal specific setup
if(USE_METAL AND APPLE)
    # Enable Objective-C++ for .mm files
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)

    # Enable ARC (Automatic Reference Counting) for Objective-C++
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
endif()

# --- SYCL Configuration (Explicit ACPP or ICPX) ---
if(USE_SYCL_ACPP AND USE_SYCL_ICPX)
    message(FATAL_ERROR "Both USE_SYCL_ACPP and USE_SYCL_ICPX are ON. Please choose only one SYCL implementation for this build.")
endif()

if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "SYCL support is requested.")
    add_definitions(-DUSE_SYCL) # Generic definition for SYCL API usage in C++

    if(NOT CMAKE_CXX_COMPILER)
        message(FATAL_ERROR "A SYCL option (USE_SYCL_ACPP or USE_SYCL_ICPX) is ON, but CMAKE_CXX_COMPILER is not set.
Please specify your SYCL compiler via -D CMAKE_CXX_COMPILER=<path_to_sycl_compiler>
Example for USE_SYCL_ICPX on Linux:      cmake -D CMAKE_CXX_COMPILER=icpx -D USE_SYCL_ICPX=ON ..
Example for USE_SYCL_ACPP on macOS:    cmake -D CMAKE_CXX_COMPILER=/opt/homebrew/bin/acpp -D USE_SYCL_ACPP=ON ..")
    endif()

    set(SYCL_COMPILER_CONFIGURED_CORRECTLY FALSE) # Flag to check if config matches expectation

    if(USE_SYCL_ACPP)
        message(STATUS "Configuring for SYCL with AdaptiveCpp (acpp).")
        add_definitions(-DPROJECT_USES_SYCL_ACPP) # Specific definition for acpp

        if(CMAKE_CXX_COMPILER MATCHES "acpp")
            message(STATUS "Verified AdaptiveCpp (acpp) compiler: ${CMAKE_CXX_COMPILER}")

            # Add rpath to acpp libraries to solve linking issues
            get_filename_component(ACPP_COMPILER_PATH ${CMAKE_CXX_COMPILER} DIRECTORY)
            get_filename_component(ACPP_INSTALL_PREFIX ${ACPP_COMPILER_PATH}/.. REALPATH)
            set(ACPP_LIB_DIR ${ACPP_INSTALL_PREFIX}/lib)
            if(EXISTS ${ACPP_LIB_DIR})
                message(STATUS "Found AdaptiveCpp library path: ${ACPP_LIB_DIR}. Adding to RPATH.")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${ACPP_LIB_DIR}")
            else()
                message(WARNING "Could not find AdaptiveCpp library path at ${ACPP_LIB_DIR}. Linking may fail.")
            endif()

            set(ACPP_TARGET_FLAGS "-O2")
            if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
                message(STATUS "Setting AdaptiveCpp default target flags for macOS (CPU via OpenMP).")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
            elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                message(STATUS "Setting AdaptiveCpp default target flags for Linux (CUDA and OpenMP).")
                set(ACPP_TARGET_FLAGS "--acpp-targets=cuda,omp")

            else() # Other OS
                message(WARNING "AdaptiveCpp on unknown OS ${CMAKE_SYSTEM_NAME}. Defaulting to OpenMP target.")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
            endif()

            if(NOT ACPP_TARGET_FLAGS STREQUAL "")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ACPP_TARGET_FLAGS}")
                message(STATUS "Applied AdaptiveCpp flags: ${ACPP_TARGET_FLAGS}")
            endif()
            set(SYCL_COMPILER_CONFIGURED_CORRECTLY TRUE)
        else()
            message(WARNING "USE_SYCL_ACPP is ON, but CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}) does not appear to be 'acpp'.
Ensure CMAKE_CXX_COMPILER is set to your AdaptiveCpp executable path (e.g., /opt/homebrew/bin/acpp for macOS).")
        endif()
    endif()

    if(USE_SYCL_ICPX)
        message(STATUS "Configuring for SYCL with Intel DPC++ (icpx).")
        add_definitions(-DPROJECT_USES_SYCL_ICPX) # Specific definition for icpx

        # CMAKE_CXX_COMPILER_ID for icpx is typically "IntelLLVM".
        if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER MATCHES "icpx")
            message(STATUS "Verified Intel DPC++ (icpx) compiler: ${CMAKE_CXX_COMPILER}")

            # Configure SYCL targets based on available hardware
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                # On Linux, check for CUDA and configure accordingly
                find_program(NVIDIA_SMI nvidia-smi)
                if(NVIDIA_SMI)
                    execute_process(COMMAND ${NVIDIA_SMI} --query-gpu=count --format=csv,noheader,nounits
                                  OUTPUT_VARIABLE GPU_COUNT
                                  OUTPUT_STRIP_TRAILING_WHITESPACE
                                  ERROR_QUIET)
                    if(GPU_COUNT AND GPU_COUNT GREATER 0)
                        message(STATUS "Found ${GPU_COUNT} NVIDIA GPU(s), configuring SYCL for CUDA backend")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3")
                        # Additional CUDA-specific optimizations for SYCL
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80")
                    else()
                        message(STATUS "No NVIDIA GPUs found, using default SYCL targets")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
                    endif()
                else()
                    message(STATUS "nvidia-smi not found, using default SYCL targets")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
                endif()
            else()
                # For non-Linux systems, use default targets
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
            endif()

            set(SYCL_COMPILER_CONFIGURED_CORRECTLY TRUE)
            if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
                message(WARNING "Intel DPC++ (icpx) is not officially supported on macOS. Compilation might fail or be unstable.")
            endif()
        else()
            message(WARNING "USE_SYCL_ICPX is ON, but CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}, ID: ${CMAKE_CXX_COMPILER_ID})
does not appear to be Intel DPC++ (icpx). Ensure CMAKE_CXX_COMPILER is set to 'icpx'.")
        endif()
    endif()

    if(NOT SYCL_COMPILER_CONFIGURED_CORRECTLY) # If neither acpp nor icpx matched the selected option
         message(WARNING "A SYCL option was enabled, but the specified CMAKE_CXX_COMPILER does not match the expected SYCL compiler for that option.
Please review your -D CMAKE_CXX_COMPILER=... setting and the USE_SYCL_ACPP / USE_SYCL_ICPX CMake options.
Compilation will proceed, but may not use SYCL correctly.")
    endif()

endif()
# --- End SYCL Configuration ---

# --- Metal Configuration ---
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal is only available on Apple platforms")
    endif()

    message(STATUS "Metal support is enabled for Apple platforms.")
    add_definitions(-DUSE_METAL)

    # Add metal-cmake subdirectory
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake EXCLUDE_FROM_ALL)

    # Set up include paths for metal-cpp headers
    include_directories(
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )

    # Find Metal framework
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND QUARTZCORE_FRAMEWORK)
        message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
        message(STATUS "Found Foundation framework: ${FOUNDATION_FRAMEWORK}")
        message(STATUS "Found QuartzCore framework: ${QUARTZCORE_FRAMEWORK}")
    else()
        message(FATAL_ERROR "Required frameworks not found")
    endif()
    set(ARBD_METAL_FRAMEWORK_PATH "" CACHE PATH "Path to the macOS SDK Frameworks directory (e.g., /Applications/Xcode.app/...)")

    # If the user hasn't provided a path, set a default to the standard location.
    if(NOT ARBD_METAL_FRAMEWORK_PATH)
        set(ARBD_METAL_FRAMEWORK_PATH "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks")
    endif()

    # Verify that the specified path is valid. If not, stop the build with a helpful error.
    if(NOT EXISTS "${ARBD_METAL_FRAMEWORK_PATH}")
        message(FATAL_ERROR "========================================================================\n"
                            "macOS SDK Frameworks not found at the specified path:\n"
                            "  ${ARBD_METAL_FRAMEWORK_PATH}\n\n"
                            "If you are using a non-standard Xcode installation (e.g., Xcode-beta.app), "
                            "please specify the correct path by setting the ARBD_METAL_FRAMEWORK_PATH CMake variable.\n\n"
                            "Example:\n"
                            "  cmake -DARBD_METAL_FRAMEWORK_PATH=/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks ...\n"
                            "========================================================================")
    endif()

    # Add the framework path to the compiler flags for all targets.
    add_compile_options("-F${ARBD_METAL_FRAMEWORK_PATH}")

    # Add framework path for Metal.framework
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -F${ARBD_METAL_FRAMEWORK_PATH}")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -F${ARBD_METAL_FRAMEWORK_PATH}")
endif()
# --- End Metal Configuration ---

## Set flags and definitions after project()
if(USE_CUDA)
  add_definitions(-DUSE_CUDA) # Kept for CUDA-specific sections in C++
endif()

if(DEBUG_BUILD)
  set(CMAKE_BUILD_TYPE Debug)
  message(STATUS "DEBUG_BUILD is ON: Build type set to Debug.")
endif()

if(USE_NVTX)
  add_definitions(-DUSE_NVTX)
endif()

if(USE_NCCL AND USE_CUDA)
  add_definitions(-DUSE_NCCL)
endif()


if(USE_CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()
if(USE_CUDA)
    set(CMAKE_BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
endif()

add_subdirectory(src)
add_subdirectory(src/Tests)

add_executable("${PROJECT_NAME}" src/arbd.cpp)

if(USE_CUDA)
    target_sources("${PROJECT_NAME}" PRIVATE
        # src/SimManager.cu
        # src/useful.cu
    )
endif()

if(USE_PYBIND)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    add_subdirectory(extern/pybind11)
    add_subdirectory(src/pybindings)
  else()
    message(WARNING "pybind11 directory or CMakeLists.txt not found at extern/pybind11.")
  endif()
endif()

target_link_libraries("${PROJECT_NAME}" PUBLIC "lib${PROJECT_NAME}")

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME}
        PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::curand
        CUDA::cublas)
endif()

if(USE_NCCL AND USE_CUDA)
  target_link_libraries("${PROJECT_NAME}" PUBLIC nccl)
endif()

if(USE_NVTX)
  if(USE_CUDA AND TARGET CUDA::nvToolsExt)
    target_link_libraries("${PROJECT_NAME}" PUBLIC CUDA::nvToolsExt)
  else()
    message(STATUS "NVTX linking might require manual setup if not using CUDA or CUDA::nvToolsExt target.")
  endif()
endif()

if(USE_METAL)
    target_link_libraries("${PROJECT_NAME}"
        PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
    )

    # Add include directories for metal-cpp
    target_include_directories("${PROJECT_NAME}"
        PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )

    message(STATUS "Linked Metal frameworks and headers")
endif()
