#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include "ARBDLogger.h"
#include "ARBDException.h"
#include "Backend/Resource.h"
#include "Backend/CUDA/CUDAManager.h"
#include <cuda.h>
#include <nvfunctional>
#include "Math/Types.h"
#include "Math/TypeName.h"
#include "../catch_boiler.h"

using namespace ARBD;

namespace Tests::Vector3 {
    enum BinaryOp_t { ADD, CROSS, DOT, SUB, FINAL };
    BinaryOp_t& operator++(BinaryOp_t& op) { return op = static_cast<BinaryOp_t>( 1+static_cast<int>(op) ); }

    auto get_binary_op_name( BinaryOp_t op ) {
	switch (op) {
	case ADD:
	    return "add";
	case SUB:
	    return "subtract";
	case CROSS:
	    return "cross";
	case DOT:
	    return "dot";
	default:
	    return "";
	}
    }

    template<typename R, typename T, typename U>
    __global__ void binary_op_test_kernel( BinaryOp_t op, R* result, T in1, U in2 ) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
	    switch (op) {
	    case ADD:
		*result = static_cast<R>(in2 + in1);
		break;
	    case SUB:
		*result = static_cast<R>(in2 - in1);
		break;
	    case CROSS:
		*result = static_cast<R>(in2.cross(in1));
		break;
	    case DOT:
		*result = static_cast<R>(in2.dot(in1));
		break;
	    default:
		*result = static_cast<R>(in2 + in1);
		break;
	    }
	}
    }
    
    template<typename R, typename T, typename U>
    R get_cpu_result(BinaryOp_t op, T in1, U in2) {
	switch (op) {
	case ADD:
	    return static_cast<R>(in2 + in1);
	case SUB:
	    return static_cast<R>(in2 - in1);
	case CROSS:
	    return static_cast<R>(in2.cross(in1));
	case DOT:
	    return static_cast<R>(in2.dot(in1));
	default:
	    return static_cast<R>(in2 + in1);
	}
    }

    template<typename T, typename U>
    void check_vectors_equal( T&& cpu, U&& gpu) {
	CHECK( ARBD::type_name<decltype(cpu)>() == ARBD::type_name<decltype(gpu)>() );
	CHECK( cpu.x == gpu.x );
	CHECK( cpu.y == gpu.y );
	CHECK( cpu.z == gpu.z );
	CHECK( cpu.w == gpu.w );
    }

    template<typename A, typename B>
    void run_tests() {
	using T = ARBD::Vector3_t<A>;
	using U = ARBD::Vector3_t<B>;
	using R = std::common_type_t<T,U>;
    
	T v1(1,1.005,0);
	U v2(0,2,0);
	R *gpu_result_d, gpu_result, cpu_result;
	ARBD::check_cuda_error(cudaMalloc((void **)&gpu_result_d, sizeof(R)), __FILE__, __LINE__);

	for (BinaryOp_t op = ADD; op < FINAL; ++op) {
	          LOGINFO("Testing operation: %s", get_binary_op_name( op ));
	    binary_op_test_kernel<R,T,U><<<1,1>>>(op, gpu_result_d, v1, v2);
	    ARBD::check_cuda_error(cudaMemcpy(&gpu_result, gpu_result_d, sizeof(R), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	    ARBD::check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__);
	
	    // Get cpu_result
	    cpu_result = get_cpu_result<R,T,U>(op, v1, v2);

	    // Check consistency
	    check_vectors_equal(cpu_result, gpu_result);
	}
	ARBD::check_cuda_error(cudaFree(gpu_result_d), __FILE__, __LINE__);
    }

    TEST_CASE( "Check that Vector3_t binary operations are identical on GPU and CPU", "[Vector3]" ){
	// INFO("Test case start");
    
	run_tests<int,int>();
	run_tests<float,float>();
	run_tests<double,double>();

	run_tests<float,double>();
	run_tests<double,float>();
	run_tests<int,double>();
    }
}
