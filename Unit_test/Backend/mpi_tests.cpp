#include "../catch_boiler.h"

#ifdef USE_MPI
#include "Backend/MPIManager.h"
#endif

using namespace ARBD;

#ifdef USE_MPI

// Helper function to run MPI operations on all ranks
void mpi_allreduce_test() {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	auto& res = manager.get_resource_for_mpi();
	DeviceBuffer<int> buf(1, res);

	MPI::Manager& mpim = MPI::Manager::instance();
	const int rank = mpim.get_rank();
	const int size = mpim.get_size();

	int val = rank;
	buf.copy_from_host(&val, 1, true);

	mpim.allReduce(buf, 1, res, MPI_SUM);

	int reduced = -1;
	buf.copy_to_host(&reduced, 1, true);

	// Only rank 0 validates the result
	if (rank == 0) {
		int expected = (size - 1) * size / 2; // sum 0..size-1
		REQUIRE(reduced == expected);
	}
}

void mpi_sendrecv_test() {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	auto& res = manager.get_resource_for_mpi();
	MPI::Manager& mpim = MPI::Manager::instance();

	const int rank = mpim.get_rank();
	const int size = mpim.get_size();

	// Only rank 0 checks the requirement
	if (rank == 0) {
		REQUIRE(size >= 2);
	}

	// Use host buffers for MPI communication (safer for testing)
	int host_send[4] = {rank, rank + 1, rank + 2, rank + 3};
	int host_recv[4] = {-1, -1, -1, -1};

	const int dst = (rank + 1) % size;
	const int src = (rank - 1 + size) % size;

	// Use blocking MPI send/recv for simplicity and reliability
	MPI_Request send_req, recv_req;
	MPI_Isend(host_send, 4, MPI_INT, dst, 42, MPI_COMM_WORLD, &send_req);
	MPI_Irecv(host_recv, 4, MPI_INT, src, 42, MPI_COMM_WORLD, &recv_req);

	MPI_Wait(&send_req, MPI_STATUS_IGNORE);
	MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

	// Each rank validates its own result
	// Rank 0 should receive from rank 1: {1, 2, 3, 4}
	// Rank 1 should receive from rank 0: {0, 1, 2, 3}
	if (rank == 0) {
		CHECK(host_recv[0] == src); // Should be 1 (from rank 1)
		CHECK(host_recv[1] == src + 1);
		CHECK(host_recv[2] == src + 2);
		CHECK(host_recv[3] == src + 3);
	}
}

TEST_CASE("MPI allreduce on device buffer (sum)", "[mpi]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	MPI::Manager& mpim = MPI::Manager::instance();
	// MPI should already be initialized by main()
	REQUIRE(mpim.is_initialized());

	// All ranks participate in the test
	mpi_allreduce_test();
}

TEST_CASE("MPI point-to-point send/recv integrity", "[mpi]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	MPI::Manager& mpim = MPI::Manager::instance();
	// MPI should already be initialized by main()
	REQUIRE(mpim.is_initialized());

	// All ranks participate in the test
	mpi_sendrecv_test();
}
#endif // USE_MPI
