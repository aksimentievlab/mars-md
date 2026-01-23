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
    Types/BaseGridDevice.h
    Types/Bitmask.h
    Types/Grid.h
    Types/NanoGridHandle.h
    Types/NanoGridKernels.h
    Types/Types.h
    Types/Math.h
)

list(APPEND HOST_ONLY_HEADERS
    Types/Array.h
    Types/BaseGrid.h
)

# Conditionally add CUDA specific source files
if(USE_CUDA)
    list(APPEND LIB_SOURCES
        #Types/CUDA/NanoGrid.cu
    )
    message(STATUS "Types: Adding CUDA specific sources to lib${PROJECT_NAME}")
endif()

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
