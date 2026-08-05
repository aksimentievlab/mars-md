# IO folder CMakeLists.txt
# Handles I/O related source files and headers

list(APPEND LIB_SOURCES
    IO/ConfigParser.cpp
    IO/DxIO.cpp
    IO/PsfPdbIO.cpp
    IO/vmdsock.cpp
    IO/imd.cpp
    IO/WKFUtils.cpp
)

list(APPEND HOST_ONLY_HEADERS
    IO/ConfigParser.h
    IO/Reader.h
    IO/DcdReader.h
    IO/DcdWriter.h
    IO/DxIO.h
    IO/TrajectoryWriter.h
    IO/FileHandle.h
    IO/BondConfigReader.h
    IO/PsfPdbIO.h
    #IO/NanoVDBIO.h
)

# Propagate to parent scope
set(LIB_SOURCES ${LIB_SOURCES} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)

