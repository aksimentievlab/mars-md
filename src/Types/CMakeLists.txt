# Types folder CMakeLists.txt
# Handles type definitions and math utilities

list(APPEND LIB_SOURCES
    Types/Vector3.cpp
)

list(APPEND HOST_DEVICE_HEADERS
    Types/Vector3.h
    Types/Vector2.h
    Types/Matrix3.h
    Types/TypeName.h
    Types/IndexList.h
    Types/GridTerm.h
    Types/BaseGridDevice.h
    Types/Bitmask.h
    Types/Types.h
    Types/Math.h
    # Sparse grid support (PNanoVDB - vendored, no nanovdb C++ dependency)
    Types/PNanoVDB.h
    Types/NanoGridDevice.h
)

list(APPEND HOST_ONLY_HEADERS
    Types/Array.h
    Types/BaseGrid.h
)

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
