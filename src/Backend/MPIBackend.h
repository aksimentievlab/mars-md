#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include <future>
#include <mpi.h>


/**
 * @brief MPI/Distributed Computing Backend
 *
 * MPI Operations:
 * - Process-to-process communication
 * - Rank management and identification
 * - Collective operations (broadcast, reduce, etc.)
 * - Distributed memory management
 */

namespace ARBD {
/**
 * @brief MPI Resource for distributed computing
 */
struct MPIResource {
  size_t rank;
  MPIResource *parent;

  MPIResource() : rank(0), parent(nullptr) {}
  MPIResource(size_t r) : rank(r), parent(nullptr) {}
  MPIResource(size_t r, MPIResource *p) : rank(r), parent(p) {}

  bool is_local() const {
#ifdef USE_MPI
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    return (current_rank == static_cast<int>(rank));
#else
    return true; // Without MPI, everything is "local"
#endif
  }

  static MPIResource Local() {
#ifdef USE_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return MPIResource{static_cast<size_t>(rank)};
#else
    return MPIResource{0};
#endif
  }

  std::string toString() const { return "MPI[" + std::to_string(rank) + "]"; }
};

/**
 * @brief MPI Backend Operations
 */
class MPIBackend {
public:
  /**
   * @brief Initialize MPI backend
   */
  static void init() {
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      LOGWARN("MPI does not support multithreading");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    LOGINFO("MPI Backend initialized: rank {} of {}", rank, size);
#else
    LOGINFO("MPI Backend: MPI support not enabled, using single process");
#endif
  }

  /**
   * @brief Finalize MPI backend
   */
  static void finalize() {
#ifdef USE_MPI
    MPI_Finalize();
#endif
  }

  /**
   * @brief Get current MPI rank
   */
  static int get_rank() {
#ifdef USE_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
  }

  /**
   * @brief Get total number of MPI processes
   */
  static int get_size() {
#ifdef USE_MPI
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#else
    return 1;
#endif
  }

  /**
   * @brief Synchronize all MPI processes
   */
  static void barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  /**
   * @brief Allocate distributed memory (host memory on each rank)
   */
  static void *allocate(size_t size, const MPIResource &resource) {
    void *ptr = std::malloc(size);
    if (!ptr) {
      ARBD::throw_value_error("Host memory allocation failed on rank {}",
                              resource.rank);
    }
    LOGINFO("MPI: Allocated {} bytes on rank {}", size, resource.rank);
    return ptr;
  }

  /**
   * @brief Deallocate distributed memory
   */
  static void deallocate(void *ptr, const MPIResource &resource) {
    if (ptr) {
      std::free(ptr);
      LOGINFO("MPI: Deallocated memory on rank {}", resource.rank);
    }
  }

  /**
   * @brief Copy data between MPI processes
   */
  template <typename T>
  static void send_data(const T *data, size_t count, int dest_rank,
                        int tag = 0) {
#ifdef USE_MPI
    MPI_Send(data, count * sizeof(T), MPI_BYTE, dest_rank, tag, MPI_COMM_WORLD);
    LOGINFO("MPI: Sent {} elements to rank {}", count, dest_rank);
#else
    ARBD::throw_not_implemented("MPI send requires USE_MPI flag");
#endif
  }

  /**
   * @brief Receive data from MPI process
   */
  template <typename T>
  static void recv_data(T *data, size_t count, int src_rank, int tag = 0) {
#ifdef USE_MPI
    MPI_Status status;
    MPI_Recv(data, count * sizeof(T), MPI_BYTE, src_rank, tag, MPI_COMM_WORLD,
             &status);
    LOGINFO("MPI: Received {} elements from rank {}", count, src_rank);
#else
    ARBD::throw_not_implemented("MPI recv requires USE_MPI flag");
#endif
  }

  /**
   * @brief Broadcast data from root to all processes
   */
  template <typename T>
  static void broadcast(T *data, size_t count, int root_rank) {
#ifdef USE_MPI
    MPI_Bcast(data, count * sizeof(T), MPI_BYTE, root_rank, MPI_COMM_WORLD);
    LOGINFO("MPI: Broadcast {} elements from rank {}", count, root_rank);
#else
    ARBD::throw_not_implemented("MPI broadcast requires USE_MPI flag");
#endif
  }

  /**
   * @brief Execute function on target rank and broadcast result
   */
  template <typename T, typename Func, typename... Args>
  static T execute_on_rank(int target_rank, Func &&func, Args &&...args) {
#ifdef USE_MPI
    T result{};
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    if (current_rank == target_rank) {
      // Target rank executes the function
      result = func(std::forward<Args>(args)...);
    }

    // Broadcast result to all ranks
    MPI_Bcast(&result, sizeof(T), MPI_BYTE, target_rank, MPI_COMM_WORLD);
    return result;
#else
    // Without MPI, just execute locally
    return func(std::forward<Args>(args)...);
#endif
  }

  /**
   * @brief Async execution on target rank
   */
  template <typename T, typename Func, typename... Args>
  static std::future<T> execute_on_rank_async(int target_rank, Func &&func,
                                              Args &&...args) {
    return std::async(std::launch::async, [=]() {
      return execute_on_rank<T>(target_rank, func, args...);
    });
  }
};

} // namespace ARBD
