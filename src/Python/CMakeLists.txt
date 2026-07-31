
## https://pybind11.readthedocs.io/en/latest/compiling.html
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# pybind11 enables LTO by default unless CMAKE_INTERPROCEDURAL_OPTIMIZATION is
# defined.  LTO breaks CUDA device linking for MODULE targets, leaving
# "fatbinData" unresolved at dlopen time (pybind/pybind11#4825).
if(USE_CUDA)
    set(_pyarbd_saved_ipo "${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

pybind11_add_module("py${PROJECT_NAME}" MODULE
    pyarbd.cpp
    pytypes.cpp
    pysystem.cpp
    pyobjects.cpp
    pybonded.cpp
    pyloadfile.cpp
    pysim.cpp
)

if(USE_CUDA)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION "${_pyarbd_saved_ipo}")

    # liblibarbd is a static archive of separably-compiled CUDA device code.
    # CUDA_RESOLVE_DEVICE_SYMBOLS triggers the device-link step on the MODULE;
    # whole-archive ensures every .cu.o from the static lib participates.
    set_target_properties("py${PROJECT_NAME}" PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
    target_compile_options("py${PROJECT_NAME}" PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fno-lto>")
    target_link_options("py${PROJECT_NAME}" PRIVATE -fno-lto)
    target_link_libraries("py${PROJECT_NAME}" PUBLIC
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,lib${PROJECT_NAME}>"
    )
else()
    target_link_libraries("py${PROJECT_NAME}" PUBLIC "lib${PROJECT_NAME}")
endif()
