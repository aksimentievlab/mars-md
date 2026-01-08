# Objects folder CMakeLists.txt
# Handles object-related headers (mostly header-only)

list(APPEND HOST_DEVICE_HEADERS
    Objects/DeviceParticle.h
)

list(APPEND HOST_ONLY_HEADERS
    Objects/Grid.h
    Objects/Tables.h
    Objects/ParticleProperties.h
    #Objects/RigidBodyProperties.h
    #Objects/GroupSites.h
)

# Propagate to parent scope
set(HOST_DEVICE_HEADERS ${HOST_DEVICE_HEADERS} PARENT_SCOPE)
set(HOST_ONLY_HEADERS ${HOST_ONLY_HEADERS} PARENT_SCOPE)
