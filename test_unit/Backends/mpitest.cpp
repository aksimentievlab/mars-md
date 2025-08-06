// File: src/Tests/test_mpi_circular_shift.cpp
#include "../catch_boiler.h"
#include "Math/Types.h"
#include "Backend/MPIBackend.h"
#include "ARBDLogger.h"
#include <vector>
#include <iomanip>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace ARBD;

// Global MPI initialization for all tests
struct MPITestSetup {
    MPITestSetup() {
#ifdef USE_MPI
        ARBD::MPIBackend::init();
#endif
    }
    
    ~MPITestSetup() {
#ifdef USE_MPI
        ARBD::MPIBackend::finalize();
#endif
    }
};

// Create a global instance to initialize/finalize MPI once
static MPITestSetup global_mpi_setup;

namespace Tests::MPI {

/**
 * @brief Circular Shift Test with ARBD Matrix Types
 * 
 * This test demonstrates sending matrices with pattern:
 * [1  0     0]
 * [0  rank  0]
 * [0  0     1]
 * 
 * Each PE creates a matrix with its rank in the (1,1) position
 * and exchanges with neighbors in a circular pattern.
 */
class MatrixCircularShift {
public:
    /**
     * @brief Direction of the circular shift
     */
    enum class Direction {
        UP,   // Send to PE with rank+1 (with wraparound)
        DOWN  // Send to PE with rank-1 (with wraparound)
    };

    /**
     * @brief Execute circular shift with matrix data
     * @param matrix_data Vector of matrices to send
     * @param direction Direction of the shift
     * @return Vector of matrices received from neighbor
     */
    template<typename T>
    static std::vector<ARBD::Matrix3_t<T>> execute_matrix_shift(
        const std::vector<ARBD::Matrix3_t<T>>& matrix_data, 
        Direction direction) {
#ifdef USE_MPI
        int rank = ARBD::MPIBackend::get_rank();
        int size = ARBD::MPIBackend::get_size();

        if (size != 8) {
            ARBD::throw_value_error("MatrixCircularShift requires exactly 8 MPI processes, got %d", size);
        }

        // Calculate neighbor ranks with circular wraparound
        int send_to, recv_from;
        if (direction == Direction::UP) {
            send_to = (rank + 1) % 8;
            recv_from = (rank + 7) % 8;  // equivalent to (rank - 1 + 8) % 8
        } else {
            send_to = (rank + 7) % 8;    // equivalent to (rank - 1 + 8) % 8
            recv_from = (rank + 1) % 8;
        }

        LOGINFO("PE {}: Sending {} matrices to PE {}, receiving from PE {}", 
               rank, matrix_data.size(), send_to, recv_from);

        // Prepare receive buffer
        std::vector<ARBD::Matrix3_t<T>> received_matrices(matrix_data.size());

        // Use MPI_Sendrecv for deadlock-free communication
        size_t matrix_size = sizeof(ARBD::Matrix3_t<T>);
        MPI_Sendrecv(matrix_data.data(), matrix_data.size() * matrix_size, MPI_BYTE, 
                     send_to, 0,
                     received_matrices.data(), received_matrices.size() * matrix_size, MPI_BYTE, 
                     recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        LOGINFO("PE {}: Successfully received {} matrices from PE {}", 
               rank, received_matrices.size(), recv_from);
        
        return received_matrices;
#else
        ARBD::throw_not_implemented("MatrixCircularShift requires MPI support");
#endif
    }

    /**
     * @brief Create the special matrix pattern: diag(1, rank, 1)
     */
    template<typename T>
    static ARBD::Matrix3_t<T> create_rank_matrix(int rank) {
        return ARBD::Matrix3_t<T>(
            T(1),    T(0),           T(0),
            T(0),    T(rank),        T(0),
            T(0),    T(0),           T(1)
        );
    }

    /**
     * @brief Verify that received matrix has correct pattern
     */
    template<typename T>
    static bool verify_matrix_pattern(const ARBD::Matrix3_t<T>& matrix, int expected_rank) {
        const T tolerance = T(1e-10);
        
        bool correct = 
            std::abs(matrix.xx - T(1)) < tolerance &&
            std::abs(matrix.xy - T(0)) < tolerance &&
            std::abs(matrix.xz - T(0)) < tolerance &&
            std::abs(matrix.yx - T(0)) < tolerance &&
            std::abs(matrix.yy - T(expected_rank)) < tolerance &&
            std::abs(matrix.yz - T(0)) < tolerance &&
            std::abs(matrix.zx - T(0)) < tolerance &&
            std::abs(matrix.zy - T(0)) < tolerance &&
            std::abs(matrix.zz - T(1)) < tolerance;
            
        return correct;
    }

    /**
     * @brief Print matrix in readable format
     */
    template<typename T>
    static void print_matrix(const ARBD::Matrix3_t<T>& matrix, const std::string& label) {
        std::cout << label << ":\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "[" << std::setw(4) << matrix.xx << " " << std::setw(4) << matrix.xy << " " << std::setw(4) << matrix.xz << "]\n";
        std::cout << "[" << std::setw(4) << matrix.yx << " " << std::setw(4) << matrix.yy << " " << std::setw(4) << matrix.yz << "]\n";
        std::cout << "[" << std::setw(4) << matrix.zx << " " << std::setw(4) << matrix.zy << " " << std::setw(4) << matrix.zz << "]\n\n";
    }
};

} // namespace Tests::MPI

// Catch2 Test Cases
TEST_CASE("MPI Matrix Circular Shift Basic Test", "[mpi][matrix][circular_shift][basic]") {
#ifdef USE_MPI
    REQUIRE(ARBD::MPIBackend::get_size() == 8);

    SECTION("Send rank matrices UP direction") {
        int rank = ARBD::MPIBackend::get_rank();
        
        // Create matrix with pattern [1 0 0; 0 rank 0; 0 0 1]
        auto my_matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(rank);
        
        // Print what we're sending
        std::cout << "PE " << rank << " sending matrix:\n";
        Tests::MPI::MatrixCircularShift::print_matrix(my_matrix, "Sent");
        
        // Verify our matrix has correct pattern
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(my_matrix, rank));
        
        // Prepare for sending
        std::vector<ARBD::Matrix3_t<double>> send_data = {my_matrix};
        
        // Execute circular shift UP
        auto received_matrices = Tests::MPI::MatrixCircularShift::execute_matrix_shift(
            send_data, Tests::MPI::MatrixCircularShift::Direction::UP);
        
        REQUIRE(received_matrices.size() == 1);
        
        // Determine who we should have received from
        int expected_sender = (rank + 7) % 8;  // Previous rank in UP direction
        
        // Print what we received
        std::cout << "PE " << rank << " received matrix from PE " << expected_sender << ":\n";
        Tests::MPI::MatrixCircularShift::print_matrix(received_matrices[0], "Received");
        
        // Verify received matrix has correct pattern
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(
            received_matrices[0], expected_sender));
        
        // Additional verification: check determinant
        double det = received_matrices[0].det();
        double expected_det = static_cast<double>(expected_sender);  // det([1,0,0;0,rank,0;0,0,1]) = rank
        REQUIRE(std::abs(det - expected_det) < 1e-10);
        
        LOGINFO("PE {}: Received matrix from PE {} with determinant {}", 
               rank, expected_sender, det);
    }

    SECTION("Send rank matrices DOWN direction") {
        int rank = ARBD::MPIBackend::get_rank();
        
        // Create matrix with pattern [1 0 0; 0 rank 0; 0 0 1]
        auto my_matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<float>(rank);
        
        // Verify our matrix has correct pattern
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(my_matrix, rank));
        
        // Prepare for sending
        std::vector<ARBD::Matrix3_t<float>> send_data = {my_matrix};
        
        // Execute circular shift DOWN
        auto received_matrices = Tests::MPI::MatrixCircularShift::execute_matrix_shift(
            send_data, Tests::MPI::MatrixCircularShift::Direction::DOWN);
        
        REQUIRE(received_matrices.size() == 1);
        
        // Determine who we should have received from
        int expected_sender = (rank + 1) % 8;  // Next rank in DOWN direction
        
        // Verify received matrix has correct pattern
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(
            received_matrices[0], expected_sender));
        
        // Check that middle element equals expected sender rank
        REQUIRE(std::abs(received_matrices[0].yy - static_cast<float>(expected_sender)) < 1e-6f);
        
        LOGINFO("PE {}: Received matrix from PE {} with yy element = {}", 
               rank, expected_sender, received_matrices[0].yy);
    }
#else
    WARN("MPI support not enabled, skipping matrix circular shift tests");
#endif
}

TEST_CASE("MPI Matrix Circular Shift Advanced Operations", "[mpi][matrix][circular_shift][advanced]") {
#ifdef USE_MPI
    REQUIRE(ARBD::MPIBackend::get_size() == 8);

    SECTION("Multiple matrices exchange") {
        int rank = ARBD::MPIBackend::get_rank();
        
        // Send multiple matrices with different patterns
        std::vector<ARBD::Matrix3_t<double>> my_matrices;
        
        // Matrix 1: Standard rank matrix
        my_matrices.push_back(Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(rank));
        
        // Matrix 2: Scaled version
        auto scaled_matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(rank);
        scaled_matrix = scaled_matrix * 2.0;  // Scale by 2
        my_matrices.push_back(scaled_matrix);
        
        // Matrix 3: Identity matrix
        my_matrices.emplace_back(1.0, 1.0, 1.0);  // Identity matrix
        
        LOGINFO("PE {}: Sending {} matrices", rank, my_matrices.size());
        
        // Execute circular shift
        auto received_matrices = Tests::MPI::MatrixCircularShift::execute_matrix_shift(
            my_matrices, Tests::MPI::MatrixCircularShift::Direction::UP);
        
        REQUIRE(received_matrices.size() == 3);
        
        int expected_sender = (rank + 7) % 8;
        
        // Verify first matrix (standard rank pattern)
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(
            received_matrices[0], expected_sender));
        
        // Verify second matrix (scaled pattern)
        REQUIRE(std::abs(received_matrices[1].xx - 2.0) < 1e-10);
        REQUIRE(std::abs(received_matrices[1].yy - 2.0 * expected_sender) < 1e-10);
        REQUIRE(std::abs(received_matrices[1].zz - 2.0) < 1e-10);
        
        // Verify third matrix (identity)
        REQUIRE(std::abs(received_matrices[2].xx - 1.0) < 1e-10);
        REQUIRE(std::abs(received_matrices[2].yy - 1.0) < 1e-10);
        REQUIRE(std::abs(received_matrices[2].zz - 1.0) < 1e-10);
        
        LOGINFO("PE {}: Successfully verified all {} received matrices from PE {}", 
               rank, received_matrices.size(), expected_sender);
    }

    SECTION("Matrix operations after exchange") {
        int rank = ARBD::MPIBackend::get_rank();
        
        // Create and send rank matrix
        auto my_matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(rank);
        std::vector<ARBD::Matrix3_t<double>> send_data = {my_matrix};
        
        auto received_matrices = Tests::MPI::MatrixCircularShift::execute_matrix_shift(
            send_data, Tests::MPI::MatrixCircularShift::Direction::UP);
        
        auto& received_matrix = received_matrices[0];
        int expected_sender = (rank + 7) % 8;
        
        // Only test matrix inverse if the received matrix is invertible (det != 0)
        // Matrix from rank 0 has determinant 0 and is singular
        if (expected_sender != 0) {
            // Perform matrix operations
            auto inverse_matrix = received_matrix.inverse();
            auto product = received_matrix.transform(inverse_matrix);
            
            // Product should be approximately identity
            REQUIRE(std::abs(product.xx - 1.0) < 1e-10);
            REQUIRE(std::abs(product.yy - 1.0) < 1e-10);
            REQUIRE(std::abs(product.zz - 1.0) < 1e-10);
            REQUIRE(std::abs(product.xy) < 1e-10);
            REQUIRE(std::abs(product.xz) < 1e-10);
            REQUIRE(std::abs(product.yx) < 1e-10);
            REQUIRE(std::abs(product.yz) < 1e-10);
            REQUIRE(std::abs(product.zx) < 1e-10);
            REQUIRE(std::abs(product.zy) < 1e-10);
        } else {
            // For rank 0 matrix, just verify it's singular
            double det = received_matrix.det();
            REQUIRE(std::abs(det) < 1e-10);  // Should be approximately 0
        }
        
        // Test vector transformation
        ARBD::Vector3_t<double> test_vector(1.0, 1.0, 1.0);
        auto transformed = received_matrix.transform(test_vector);
        
        // Expected result: [1, expected_sender, 1]
        REQUIRE(std::abs(transformed.x - 1.0) < 1e-10);
        REQUIRE(std::abs(transformed.y - static_cast<double>(expected_sender)) < 1e-10);
        REQUIRE(std::abs(transformed.z - 1.0) < 1e-10);
        
        LOGINFO("PE {}: Matrix operations verified for matrix from PE {}", 
               rank, expected_sender);
    }

    SECTION("Complete ring circulation test") {
        int rank = ARBD::MPIBackend::get_rank();
        
        // Start with our rank matrix
        auto current_matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(rank);
        auto original_matrix = current_matrix;
        
        // Perform 8 UP shifts - should return to original after full circle
        for (int shift = 0; shift < 8; ++shift) {
            std::vector<ARBD::Matrix3_t<double>> send_data = {current_matrix};
            auto received = Tests::MPI::MatrixCircularShift::execute_matrix_shift(
                send_data, Tests::MPI::MatrixCircularShift::Direction::UP);
            
            current_matrix = received[0];
            
            // Track which rank's matrix we should have
            int expected_rank = (rank + 7 - shift) % 8;  // Going backwards due to UP direction
            REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(
                current_matrix, expected_rank));
        }
        
        // After 8 shifts, we should have our original matrix back
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(
            current_matrix, rank));
        
        LOGINFO("PE {}: Complete ring circulation verified - returned to original matrix", rank);
    }
#else
    WARN("MPI support not enabled, skipping advanced matrix tests");
#endif
}

TEST_CASE("MPI Matrix Circular Shift Error Handling", "[mpi][matrix][circular_shift][error]") {
#ifdef USE_MPI
    SECTION("Correct number of processes check") {
        // This should pass if we have exactly 8 processes
        if (ARBD::MPIBackend::get_size() == 8) {
            int rank = ARBD::MPIBackend::get_rank();
            auto matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(rank);
            std::vector<ARBD::Matrix3_t<double>> send_data = {matrix};
            
            // This should not throw
            REQUIRE_NOTHROW(Tests::MPI::MatrixCircularShift::execute_matrix_shift(
                send_data, Tests::MPI::MatrixCircularShift::Direction::UP));
        } else {
            // If we don't have 8 processes, the function should throw
            auto matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(0);
            std::vector<ARBD::Matrix3_t<double>> send_data = {matrix};
            
            REQUIRE_THROWS_AS(Tests::MPI::MatrixCircularShift::execute_matrix_shift(
                send_data, Tests::MPI::MatrixCircularShift::Direction::UP),
                ARBD::Exception);
        }
    }

    SECTION("Matrix pattern verification") {
        // Test the verification function itself
        auto correct_matrix = Tests::MPI::MatrixCircularShift::create_rank_matrix<double>(5);
        REQUIRE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(correct_matrix, 5));
        
        // Create incorrect matrix
        ARBD::Matrix3_t<double> incorrect_matrix(
            1.0, 1.0, 0.0,  // Wrong: should be (1,0,0)
            0.0, 5.0, 0.0,
            0.0, 0.0, 1.0
        );
        REQUIRE_FALSE(Tests::MPI::MatrixCircularShift::verify_matrix_pattern(incorrect_matrix, 5));
    }
#else
    WARN("MPI support not enabled, skipping error handling tests");
#endif
}