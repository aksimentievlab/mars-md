#include "Resource.h"

HOST DEVICE size_t caller_id() {
#ifdef USE_MPI
    // LOGINFO("... caller_id(): MPI path");
    Exception( NotImplementedError, "caller_id() not implemented on GPU" );
    int world_rank;
    return MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#else
#ifdef __CUDA_ARCH__
    LOGINFO("... caller_id(): GPU path");
    Exception( NotImplementedError, "caller_id() not implemented on GPU" );
    return 0;
#else
    // LOGWARN("... caller_id() CPU path: return 0");
    return 0;
#endif
#endif
}
