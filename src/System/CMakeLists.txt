# System folder CMakeLists.txt
# Handles system-level source files including patch management and decomposition

list(APPEND LIB_SOURCES
    System/PatchManager.cpp
    System/SimSystem.cpp
    System/SystemState.cpp
    System/Decomposers.cpp
    System/Reservoir.cpp
    System/PatchDecomposer/Spatial.cpp
    System/PatchDecomposer/ZOrder.cpp
    System/PatchDecomposer/Gemoetric.cpp
    System/PatchDecomposer/RecursiveBisection.cpp
)

list(APPEND HOST_DEVICE_HEADERS
    System/PeriodicBox.h
    System/PressureCalculator.h
    System/SystemForward.h
)

list(APPEND HOST_ONLY_HEADERS
    System/PatchManager.h
    System/SimSystem.h
    System/SystemState.h
    System/Decomposers.h
    System/PatchDecomposer.h
    System/Reservoir.h
    System/PatchDecomposer/Spatial.h
    System/PatchDecomposer/ZOrder.h
    System/PatchDecomposer/Gemoetric.h
    System/PatchDecomposer/RecursiveBisection.h
)

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
