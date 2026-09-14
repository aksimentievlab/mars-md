
# nanobind_build_library() reads ${Python_INCLUDE_DIRS} as a plain variable.
# extern/nanobind runs its own find_package(Python) in *its* directory scope,
# which is a sibling of this one, so that value never reaches here and the
# nanobind-static sources fail on a missing Python.h. Find it again locally.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

## https://nanobind.readthedocs.io/en/latest/building.html
nanobind_add_module("py${PROJECT_NAME}"
    pymars.cpp
    pysystem.cpp
    pyobjects.cpp
    pybonded.cpp
    pynonbonded.cpp
    pytables.cpp
    pyloadfile.cpp
    pysim.cpp
)

# The target keeps the "py" prefix to avoid colliding with the mars executable
# target; the module file must be named for its NB_MODULE symbol (_core).
# It is imported as marsmd._core - see dev_notes.md.
set_target_properties("py${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "_core")

if(USE_CUDA)
    # liblibmars is a static archive of separably-compiled CUDA device code.
    # CUDA_RESOLVE_DEVICE_SYMBOLS triggers the device-link step on the MODULE;
    # whole-archive ensures every .cu.o from the static lib participates.
    set_target_properties("py${PROJECT_NAME}" PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries("py${PROJECT_NAME}" PUBLIC
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,lib${PROJECT_NAME}>"
    )
else()
    target_link_libraries("py${PROJECT_NAME}" PUBLIC "lib${PROJECT_NAME}")
endif()
