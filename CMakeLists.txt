cmake_minimum_required(VERSION 3.25.5 FATAL_ERROR)
set(PROJECT_NAME_BASE "arbd")
if(DEFINED PROJECT_NAME_SUFFIX)
    set(PROJECT_NAME "${PROJECT_NAME_BASE}${PROJECT_NAME_SUFFIX}")
else()
    set(PROJECT_NAME "${PROJECT_NAME_BASE}")
endif()

# Single project() call
if(USE_CUDA)
    project("${PROJECT_NAME}" VERSION 2.0 LANGUAGES CXX CUDA)
else()
    project("${PROJECT_NAME}" VERSION 2.0 LANGUAGES CXX)
endif()


# --- Project Options ---
enable_testing()
option(USE_CUDA "Use CUDA" OFF)
option(USE_SYCL_ACPP "Use SYCL with AdaptiveCpp (acpp)" OFF)
option(USE_SYCL_ICPX "Use SYCL with Intel DPC++ (icpx)" OFF)
option(DEBUG_BUILD "Build with debug flags (overrides CMAKE_BUILD_TYPE)" OFF)
option(USE_NCCL "Use NCCL for single node GPU peer communication" OFF)
option(USE_NVTX "Build with NVTX profiling ranges" OFF)
option(USE_PYBIND "Build Python bindings" OFF)
option(USE_METAL "Use Apple Silicon Metal" OFF)
option(USE_MPI "Use MPI for distributed computing" OFF)

option(SYCL_DEVICE_TYPE "Device type for SYCL builds, can be Nvidia (0), AMD (1) or Intel (2). Default is 0 (Nvidia)." 0)
# Optional: override GPU arch for icpx SYCL targets (used when SYCL_DEVICE_TYPE is set)
set(SYCL_NVIDIA_GPU_ARCH "sm_80" CACHE STRING "NVIDIA GPU architecture for icpx SYCL (e.g. sm_80, sm_86, sm_89, sm_90). Used when SYCL_DEVICE_TYPE=0.")
set(SYCL_AMD_ARCH "gfx906" CACHE STRING "AMD GPU architecture for icpx SYCL (e.g. gfx906, gfx908, gfx90a). Used when SYCL_DEVICE_TYPE=1.")
set(SYCL_INTEL_GPU_DEVICE "pvc" CACHE STRING "Intel GPU device for icpx SYCL AOT (e.g. pvc, mtl-h, skl). Used when SYCL_DEVICE_TYPE=2. Empty = spir64_gen generic.")

# --- Backend Selection Validation ---
# Ensure only one backend is selected
set(BACKEND_COUNT 0)
if(USE_CUDA)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()
if(USE_METAL)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()

if(BACKEND_COUNT EQUAL 0)
    message(FATAL_ERROR "No backend selected. Please enable exactly one of: USE_CUDA, USE_METAL, USE_SYCL_ACPP, or USE_SYCL_ICPX")
elseif(BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR "Multiple backends selected. Please enable exactly one of: USE_CUDA, USE_METAL, USE_SYCL_ACPP, or USE_SYCL_ICPX")
endif()

# Validate SYCL options (only one SYCL variant should be selected)
if(USE_SYCL_ACPP AND USE_SYCL_ICPX)
    message(FATAL_ERROR "Both USE_SYCL_ACPP and USE_SYCL_ICPX are ON. Please choose only one SYCL implementation.")
endif()

# --- Basic CMake Setup ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_NANOVDB OFF)
if(USE_CUDA OR USE_SYCL_ACPP OR USE_SYCL_ICPX)
    set(BUILD_NANOVDB ON)
endif()

if(BUILD_NANOVDB)
set(NANOVDB_BUILD_TOOLS OFF CACHE BOOL "Disable NanoVDB tools")
set(NANOVDB_BUILD_UNITTESTS OFF CACHE BOOL "Disable NanoVDB tests")
set(NANOVDB_BUILD_EXAMPLES OFF CACHE BOOL "Disable NanoVDB examples")
set(NANOVDB_BUILD_PYTHON_MODULE OFF CACHE BOOL "Disable NanoVDB python module")
set(NANOVDB_USE_CUDA OFF CACHE BOOL "Disable NanoVDB CUDA support")
set(NANOVDB_USE_OPENVDB OFF CACHE BOOL "Disable NanoVDB OpenVDB support")
add_subdirectory(extern/openvdb/nanovdb/nanovdb)
endif()

# Define C++23 macros as empty to avoid compatibility issues
add_compile_definitions(_GLIBCXX23_CONSTEXPR=)

# Generate compile_commands.json for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find threading library for use across the project
find_package(Threads REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/extern)



if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120") #100, 120 are not supported by all GPUs and compilers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --allow-unsupported-compiler")
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

# --- SYCL Configuration (Explicit ACPP or ICPX) ---

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

            # AdaptiveCpp backends from SYCL_DEVICE_TYPE (0=Nvidia, 1=AMD, 2=Intel); same as icpx
            set(ACPP_TARGET_FLAGS "-O2")
            if(SYCL_DEVICE_TYPE EQUAL 1)
                message(STATUS "AdaptiveCpp: backend AMD HIP, arch=${SYCL_AMD_ARCH}")
                set(ACPP_TARGET_FLAGS "${ACPP_TARGET_FLAGS} --acpp-targets=hip:${SYCL_AMD_ARCH}")
            elseif(SYCL_DEVICE_TYPE EQUAL 2)
                message(STATUS "AdaptiveCpp: backend Intel Level Zero")
                set(ACPP_TARGET_FLAGS "${ACPP_TARGET_FLAGS} --acpp-targets=level_zero")
            else()
                message(STATUS "AdaptiveCpp: backend NVIDIA CUDA, arch=${SYCL_NVIDIA_GPU_ARCH}")
                set(ACPP_TARGET_FLAGS "${ACPP_TARGET_FLAGS} --acpp-targets=cuda:${SYCL_NVIDIA_GPU_ARCH}")
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
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-designator")
            message(STATUS "Suppressed -Wc99-designator warning for Intel DPC++")

            # Configure SYCL targets based on SYCL_DEVICE_TYPE (0=Nvidia, 1=AMD, 2=Intel)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
            if(SYCL_DEVICE_TYPE EQUAL 1)
                # AMD GPU: amdgcn-amd-amdhsa target (requires icpx built with --hip)
                message(STATUS "icpx: Configuring SYCL for AMD GPU (SYCL_DEVICE_TYPE=1), arch=${SYCL_AMD_ARCH}")
                set(SYCL_DEVICE_EXTRA_FLAGS "-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${SYCL_AMD_ARCH}")
                set(SYCL_LINK_FLAGS "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${SYCL_AMD_ARCH}")
            elseif(SYCL_DEVICE_TYPE EQUAL 2)
                # Intel GPU: spir64_gen AOT (Level Zero / Intel GPU)
                if(SYCL_INTEL_GPU_DEVICE)
                    message(STATUS "icpx: Configuring SYCL for Intel GPU (SYCL_DEVICE_TYPE=2), device=${SYCL_INTEL_GPU_DEVICE}")
                    # -Xsycl-target-backend takes one backend arg; "-device <name>" must be quoted together
                    set(SYCL_DEVICE_EXTRA_FLAGS "-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \"-device ${SYCL_INTEL_GPU_DEVICE}\"")
                    set(SYCL_LINK_FLAGS "-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \"-device ${SYCL_INTEL_GPU_DEVICE}\"")
                else()
                    message(STATUS "icpx: Configuring SYCL for Intel GPU (SYCL_DEVICE_TYPE=2), generic spir64_gen")
                    set(SYCL_DEVICE_EXTRA_FLAGS "-fsycl-targets=spir64_gen")
                    set(SYCL_LINK_FLAGS "-fsycl -fsycl-targets=spir64_gen")
                endif()
            else()
                # Nvidia (0): use user-selected SYCL_DEVICE_TYPE and SYCL_NVIDIA_GPU_ARCH
                message(STATUS "icpx: Configuring SYCL for NVIDIA GPU (SYCL_DEVICE_TYPE=0), arch=${SYCL_NVIDIA_GPU_ARCH}")
                set(SYCL_DEVICE_EXTRA_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${SYCL_NVIDIA_GPU_ARCH}")
                set(SYCL_LINK_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${SYCL_NVIDIA_GPU_ARCH}")
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
    #set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -F${ARBD_METAL_FRAMEWORK_PATH}")
endif()
# --- End Metal Configuration ---

# --- Metal Shader Compilation ---
if(USE_METAL)
    # Set up Metal shader compilation at project level
    set(METAL_OUTPUT_DIR ${CMAKE_BINARY_DIR})

    # Find all Metal source files in the src directory
    file(GLOB_RECURSE METAL_SOURCES
        ${CMAKE_SOURCE_DIR}/src/*.metal
        ${CMAKE_SOURCE_DIR}/src/Random/*.metal
    )

    # Also include test Metal files to ensure all kernels are available
    file(GLOB_RECURSE TEST_METAL_SOURCES
        ${CMAKE_SOURCE_DIR}/test_unit/*.metal
        ${CMAKE_SOURCE_DIR}/test_unit/Kernels/*.metal
        ${CMAKE_SOURCE_DIR}/test_unit/Math/*.metal
    )

    # Combine all Metal sources
    list(APPEND METAL_SOURCES ${TEST_METAL_SOURCES})

    if(METAL_SOURCES)
        message(STATUS "Found Metal shader sources: ${METAL_SOURCES}")

        # Create a list to hold the compiled .air files
        set(METAL_AIR_FILES)

        # Compile each .metal file to .air
        foreach(METAL_SOURCE ${METAL_SOURCES})
            get_filename_component(METAL_NAME ${METAL_SOURCE} NAME_WE)
            set(AIR_FILE ${METAL_OUTPUT_DIR}/${METAL_NAME}.air)

            add_custom_command(
                OUTPUT ${AIR_FILE}
                COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCE} -o ${AIR_FILE} -I${CMAKE_SOURCE_DIR}/src -gline-tables-only -frecord-sources
                DEPENDS ${METAL_SOURCE}
                COMMENT "Compiling Metal shader: ${METAL_NAME}.metal"
                VERBATIM
            )

            list(APPEND METAL_AIR_FILES ${AIR_FILE})
        endforeach()

        # Create the Metal library for the main project
        set(METAL_LIBRARY ${METAL_OUTPUT_DIR}/default.metallib)
        add_custom_command(
            OUTPUT ${METAL_LIBRARY}
            COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METAL_LIBRARY}
            DEPENDS ${METAL_SOURCES} ${METAL_AIR_FILES}
            COMMENT "Creating Metal library: default.metallib"
            VERBATIM
        )

        # Create a custom target for Metal shaders
        add_custom_target(metal_shaders_main ALL DEPENDS ${METAL_LIBRARY})

        message(STATUS "Metal shader compilation configured for main project")
    else()
        message(STATUS "No Metal shader sources found in src directory")
    endif()
endif()
# --- End Metal Shader Compilation ---

# --- MPI Configuration ---
if(USE_MPI)
    message(STATUS "MPI support is enabled.")

    # Find MPI package
    find_package(MPI REQUIRED COMPONENTS CXX)

    if(MPI_CXX_FOUND)
        message(STATUS "Found MPI:")
        message(STATUS "  MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
        message(STATUS "  MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "  MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")
        message(STATUS "  MPI version: ${MPI_CXX_VERSION}")
    else()
        message(FATAL_ERROR "MPI C++ support not found")
    endif()

    # Add MPI definitions
    add_definitions(-DUSE_MPI)

    # For OpenMPI RHEL 5.0.7 with GCC 14 - specific optimizations
    if(MPI_CXX_COMPILER MATCHES "mpi")
        # Detect OpenMPI version and configure accordingly
        execute_process(
            COMMAND ${MPI_CXX_COMPILER} --showme:version
            OUTPUT_VARIABLE MPI_VERSION_OUTPUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(MPI_VERSION_OUTPUT MATCHES "Open MPI")
            message(STATUS "Detected OpenMPI: ${MPI_VERSION_OUTPUT}")

            # Enable OpenMPI specific optimizations
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMPI_SKIP_MPICXX")

            # For RHEL systems with OpenMPI 5.0.7
            if(MPI_VERSION_OUTPUT MATCHES "5\\.0\\.")
                message(STATUS "Configuring for OpenMPI 5.0.x")
                # Add any specific flags for OpenMPI 5.0.x
            endif()
        endif()
    endif()

    # Configure MPI with SYCL if both are enabled
    if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
        message(STATUS "Configuring MPI with SYCL backend")

        # For Intel DPC++ with MPI
        if(USE_SYCL_ICPX)
            # Ensure MPI works with Intel DPC++ compiler
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPI_NO_CPPBIND")
            message(STATUS "Added MPI_NO_CPPBIND for Intel DPC++ compatibility")
        endif()

        # For AdaptiveCpp with MPI
        if(USE_SYCL_ACPP)
            # Ensure MPI works with AdaptiveCpp compiler
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPI_NO_CPPBIND")
            message(STATUS "Added MPI_NO_CPPBIND for AdaptiveCpp compatibility")
        endif()
    endif()

    # Set MPI compiler wrapper detection
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "Using MPI compiler wrapper: ${CMAKE_CXX_COMPILER}")
        # MPI flags are automatically handled by compiler wrapper
    else()
        message(STATUS "Using standard compiler with MPI libraries")
        # Manual MPI linking will be handled in target_link_libraries
    endif()
else()
    message(STATUS "MPI support is disabled.")
endif()
# --- End MPI Configuration ---

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
add_subdirectory(Unit_test)

add_executable("${PROJECT_NAME}" src/arbd.cpp)

if(USE_CUDA)
    target_sources("${PROJECT_NAME}" PRIVATE
        # src/SimManager.cu
        # src/useful.cu
    )
endif()

# Make the main executable depend on Metal shaders if Metal is enabled
if(USE_METAL AND TARGET metal_shaders_main)
    add_dependencies("${PROJECT_NAME}" metal_shaders_main)
endif()

if(USE_PYBIND)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    add_subdirectory(extern/pybind11)
    add_subdirectory(src/Python)
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
        METAL_CPP
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

    message(STATUS "Linked Metal frameworks, Metal-cpp library, and headers")
endif()

if(USE_MPI)
    # Link MPI to main executable
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        # When using MPI compiler wrapper, linking is automatic
        message(STATUS "MPI linking handled automatically by compiler wrapper")
    else()
        # Manual MPI linking when not using wrapper (e.g., icpx + MPI libraries)
        target_link_libraries("${PROJECT_NAME}"
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories("${PROJECT_NAME}"
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to main executable")
    endif()
endif()

