# RBOperation folder CMakeLists.txt
# Plasmonic rigid-body coupling (scuff-em). This is the only part of the tree
# that touches scuff-em, so the dependency is discovered and wrapped here
# instead of in the root CMakeLists.
#
# NOT wired into src/CMakeLists.txt yet. To enable, add to src/CMakeLists.txt:
#   add_subdirectory(RBOperation)                          # before add_library
#   target_link_libraries("lib${PROJECT_NAME}" PUBLIC ARBD::scuff)   # after it
# and configure with -DUSE_SCUFF=ON.
#
# scuff-em is built by ./linkscuff.sh, which installs into
# ${CMAKE_SOURCE_DIR}/build/scuff-link (headers under include/scuff-em, one
# merged libscuff.so under lib/ carrying its own rpath to MKL).

option(USE_SCUFF "Build the scuff-em plasmonic rigid-body coupling" OFF)

# --- Always built: host-set rigid-body forces, no scuff-em dependency --------
# RBHostFTManager is the single path for host-computed per-body external
# force/torque (scuff-em is one producer among several), so it must not be gated
# on USE_SCUFF.
list(APPEND HOST_DEVICE_HEADERS
    RBOperation/ApplyHostForce.h
)
list(APPEND HOST_ONLY_HEADERS
    RBOperation/RBHostFTManager.h
)

# ApplyExternalForcesKernel is declared with an extern template in
# ApplyHostForce.h, so its launch_cuda_kernel instantiation has to be forced into
# a real CUDA translation unit or every host-only .cpp calling push() picks up the
# non-CUDA stub in KernelHelper.cuh.
if(USE_CUDA)
    list(APPEND LIB_SOURCES
        RBOperation/RBOpKernel.cu
    )
    set_source_files_properties(
        RBOperation/RBOpKernel.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "RBOperation: Adding CUDA specific sources to lib${PROJECT_NAME}")
endif()

if(NOT USE_SCUFF)
    message(STATUS "RBOperation: USE_SCUFF=OFF, skipping scuff-em coupling")
    set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
    set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
    set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
    return()
endif()

# --- Locate scuff-em ---------------------------------------------------------
set(SCUFF_ROOT "${CMAKE_SOURCE_DIR}/build/scuff-link"
    CACHE PATH "scuff-em install prefix (see linkscuff.sh)")

# libscuff.h pulls libhrutil.h/libhmat.h with angle brackets, so the directory
# holding them must be on the include path directly - the installed pkg-config
# Cflags point one level too high (${prefix}/include).
find_path(SCUFF_INCLUDE_DIR
    NAMES libscuff.h
    HINTS "${SCUFF_ROOT}" ENV SCUFF_ROOT
    PATH_SUFFIXES include/scuff-em scuff-em include
)

find_library(SCUFF_LIBRARY
    NAMES scuff
    HINTS "${SCUFF_ROOT}" ENV SCUFF_ROOT
    PATH_SUFFIXES lib lib64
)

if(NOT SCUFF_INCLUDE_DIR OR NOT SCUFF_LIBRARY)
    message(FATAL_ERROR
        "USE_SCUFF=ON but scuff-em was not found under '${SCUFF_ROOT}'.\n"
        "  Build it first:  source /opt/intel/oneapi/setvars.sh && ./linkscuff.sh\n"
        "  Or point at an existing install with -DSCUFF_ROOT=<prefix>.\n"
        "  include dir: ${SCUFF_INCLUDE_DIR}\n"
        "  library:     ${SCUFF_LIBRARY}")
endif()

# GLOBAL so the parent scope can link it after add_subdirectory() returns.
if(NOT TARGET ARBD::scuff)
    add_library(ARBD::scuff UNKNOWN IMPORTED GLOBAL)
    set_target_properties(ARBD::scuff PROPERTIES
        IMPORTED_LOCATION "${SCUFF_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SCUFF_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "USE_SCUFF"
    )

    # scuff-em's BLAS/LAPACK (MKL) and readline dependencies are recorded in
    # libscuff.so together with an rpath, so only its own direct needs are
    # restated here. Do NOT define HAVE_CONFIG_H: it makes EquivalentEdgePairs.h
    # include scuff-em's private <config.h>, which is not installed.
    set_property(TARGET ARBD::scuff APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES Threads::Threads ${CMAKE_DL_LIBS} m)

    # scuff-em is compiled with OpenMP; carrying the runtime explicitly keeps
    # the link line correct when the consumer is icpx rather than g++.
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set_property(TARGET ARBD::scuff APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    endif()

    get_filename_component(_scuff_libdir "${SCUFF_LIBRARY}" DIRECTORY)
    list(APPEND CMAKE_BUILD_RPATH "${_scuff_libdir}")
    list(APPEND CMAKE_INSTALL_RPATH "${_scuff_libdir}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" PARENT_SCOPE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" PARENT_SCOPE)

    message(STATUS "RBOperation: scuff-em headers  ${SCUFF_INCLUDE_DIR}")
    message(STATUS "RBOperation: scuff-em library  ${SCUFF_LIBRARY}")
endif()

# --- scuff-em-only sources ---------------------------------------------------
# ScuffRigidBody.h is host-only: it includes libscuff.h, which no device compiler
# should ever see.
list(APPEND HOST_ONLY_HEADERS
    RBOperation/ScuffRigidBody.h
)

set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
