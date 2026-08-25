
## https://nanobind.readthedocs.io/en/latest/building.html
nanobind_add_module("py${PROJECT_NAME}"
    pymars.cpp
    pysystem.cpp
    pyobjects.cpp
    pybonded.cpp
    pyloadfile.cpp
    pysim.cpp
)

# The target keeps the "py" prefix to avoid colliding with the mars executable
# target, but the built module must be named for its NB_MODULE symbol
# (marsmd) or `import marsmd` won't find it.
set_target_properties("py${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "marsmd")

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
