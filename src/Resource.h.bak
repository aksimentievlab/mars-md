#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "ARBDException.h"
#include "GPUManager.h"


/**
 * @brief Routine that returns the index of the calling resource,
 * regardless of whether it an MPI rank or a GPU.
 */
HOST DEVICE size_t caller_id();


/**
 * @brief Represents a resource that can store data and perform computations.
 */
struct Resource {
    /**
     * @brief Enum to specify the type of the resource (e.g., CPU or GPU).
     */
    enum ResourceType {CPU, MPI, GPU};
    ResourceType type; ///< Type of the resource.
    size_t id; ///< Unique identifier associated with the resource.
    Resource* parent; ///< Parent resource; nullptr unless type is GPU
        
    /**
     * @brief Check if resource is local to the calling thread. 
     */
    HOST DEVICE bool is_local() const {
	bool ret = true;
#ifdef __CUDACC__
	ret = (type == GPU);
	LOGWARN("Resource::is_local(): called from GPU but not fully implemented on GPU devices, returning {}",ret);
#else
	LOGINFO("Resource::is_local() out of CUDACC...");
	if (type == GPU && parent != nullptr) {
	    LOGINFO("Resource::is_local(): called from CPU for resource on GPU, checking parent...");
	    ret = parent->is_local();
	} else {
	    ret = (caller_id() == id);
	    LOGINFO("Resource::is_local(): called from CPU, returning {}", ret);
	}
#endif
	return ret;
    };
    // HOST DEVICE static bool is_local() { // check if thread/gpu idx matches some global idx };

    static Resource Local() {
	LOGWARN("Resource::Local() not properly implemented");
#ifdef __CUDACC__
	return Resource{ GPU, 0 };
#else
	return Resource{ CPU, 0 };
#endif
    };
    bool operator==(const Resource& other) const { return type == other.type && id == other.id; };
	
};
