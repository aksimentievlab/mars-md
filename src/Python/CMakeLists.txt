
## https://pybind11.readthedocs.io/en/latest/compiling.html
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)  

pybind11_add_module("py${PROJECT_NAME}" MODULE pyarbd.cpp pytypes.cpp)

target_link_libraries("py${PROJECT_NAME}" PUBLIC "lib${PROJECT_NAME}")
