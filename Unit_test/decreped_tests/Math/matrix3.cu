#include "../catch_boiler.h"
#include "Math/Types.h"
#include "Math/TypeName.h"
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Resource.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "ARBDLogger.h"
#include "ARBDException.h"
#include "Math/TypeName.h"
namespace Catch {
    template<typename T, bool b1, bool b2>
    struct StringMaker<ARBD::Matrix3_t<T,b1,b2>> {
        static std::string convert( ARBD::Matrix3_t<T,b1,b2> const& value ) {
            return value.to_string();
        }
    };
}


DEF_RUN_TRIAL

namespace Tests::Unary::Matrix3 {
    template<typename R, typename T>
    struct DeterminantOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.det()); } };

    template<typename R, typename T>
    struct NormalizeDetOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized().det()); } };
	
    template<typename R, typename T>
    struct InverseOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.inverse()); } };

    template<typename R, typename T>
    struct TransposeOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.transpose()); } };

    template<typename A, bool is_diag=false, bool check_diag=false>
    void run_tests() {
	using T = ARBD::Matrix3_t<A, is_diag, check_diag>;
	// std::ostream test_info;
	INFO( "Testing " << ARBD::type_name<T>() << " unary operators that" );
	{
	    using R = A;
	    INFO(  "    return " << ARBD::type_name<R>() );
	    run_trial<DeterminantOp<R,T>,R,T>( "Testing determinant", R(6), T(1,2,3) );
	    run_trial<NormalizeDetOp<R,T>,R,T>( "Testing that normalized matrix has determinant == 1", R(1), T(1,1,1) );
	    // run_trial<NormalizeDetOp<R,T>,R,T>( "Testing that normalized matrix has determinant == 1", R(1), T(2,2,2) );
	    // run_trial<NormalizeDetOp<R,T>,R,T>( "Testing that normalized matrix has determinant == 1", R(1), T(1,2,3) );
	}

	{ // Test operators that return Matrix
	    using R = T;
	    INFO( "    return " << ARBD::type_name<R>() );
	    run_trial<TransposeOp<R,T>,R,T>( "Testing transpose",
					     R(1,4,7,
					       2,5,8,
					       3,6,9),
					     T(1,2,3,
					       4,5,6,
					       7,8,9) );
	    run_trial<NegateOp<R,T>,R,T>( "Testing negation", R(-1,-2,-3), T(1,2,3) );
	    run_trial<InverseOp<R,T>,R,T>( "Testing inversion",
					   R(3.0/16, 0.25, -5.0/16,
					     0.25, 0, 0.25, -5.0/16,
					     0.25, 3.0/16),
					   T(1,2,-1,
					     2,1,2,
					     -1,2,1) );
	    run_trial<NormalizedOp<R,T>,R,T>( "Testing Matrix3_t<>.normalized()", R(1,1,1), T(1,1,1) );
	    run_trial<NormalizedOp<R,T>,R,T>( "Testing Matrix3_t<>.normalized()", R(1,1,1), T(2,2,2) );
	    run_trial<NormalizedOp<R,T>,R,T>( "Testing Matrix3_t<>.normalized()", R(1,1,1), T(1,2,3) );
	}
    }
    TEST_CASE( "Check that Matrix3_t unary operations are identical on GPU and CPU", "[Tests::Unary::Matrix3]" ) {
	// INFO("Test case start");
	const bool is_diag = false;
	const bool check_diag = true;
	run_tests<double, is_diag, check_diag>();
	run_tests<float, is_diag, check_diag>();
	// run_tests<int, is_diag, check_diag>(); // commented because NormalizedOp test fails with int type
    }
}

namespace Tests::Binary::Matrix3 {
    // Custom Operations for Matrix3
    template<typename R, typename T, typename U> 
    struct CrossOp { HOST DEVICE static R op(T a, U b) { return static_cast<R>(a.cross(b)); } };

    // Run
    template<typename A, typename B, bool is_diag=false, bool check_diag=false>
    void run_tests() {
	using T = ARBD::Matrix3_t<A, is_diag, check_diag>;

	{ // Test binary operators that return Matrix where U b is a scalar
	    using U = B;
	    using R = T;
	    run_trial<MultOp<R,T,U>,R,T,U>( "Scale", R(2,4,6), T(1,2,3), U(2) );
	}

	{ // Test binary operators that return Vector where U b is Vector
	    using U = ARBD::Vector3_t<B>;
	    using R = U;
	    run_trial<MultOp<R,T,U>,R,T,U>( "Matrix.transform(Vector)",
					    R(1+4+9,
					      4+10+18,
					      7+16+27),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    U(1,2,3) );
	}

	{ // Test binary operators that return Vector and U b is Vector
	    using U = ARBD::Vector3_t<B>;
	    using R = U;
	    run_trial<MultOp<R,T,U>,R,T,U>( "Matrix element multiplication",
					    R(1+4+9,
					      4+10+18,
					      7+16+27),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    U(1,2,3) );
	}

	{ // Test binary operators that return Matrix and U b is Matrix
	    using U = ARBD::Matrix3_t<B,is_diag,check_diag>;
	    using R = std::common_type_t<T,U>;
	    run_trial<AddOp<R,T,U>,R,T,U>( "Matrix addition",
					    R(2,4,6,
					      8,10,12,
					      14,16,18),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    T(1,2,3,
					      4,5,6,
					      7,8,9));

	    run_trial<SubOp<R,T,U>,R,T,U>( "Matrix subtraction",
					    R(0,0,0),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    T(1,2,3,
					      4,5,6,
					      7,8,9) );

	    run_trial<MultOp<R,T,U>,R,T,U>( "Matrix transformation",
					    R(1+8+21, 2+10+24, 3+12+27,
					      4+20+42, 8+25+48, 12+30+54,
					      7+32+63, 14+40+72, 21+48+81),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    T(1,2,3,
					      4,5,6,
					      7,8,9) );
	}	
    }
    TEST_CASE( "Check that Matrix3_t binary operations are identical on GPU and CPU", "[Tests::Binary::Matrix3]" ) {
	// INFO("Test case start");
	const bool is_diag = false;
	const bool check_diag = true;
	run_tests<double,double, is_diag, check_diag>();
	// run_tests<float,double, is_diag, check_diag>();
	//run_tests<int,double, is_diag, check_diag>();

	//run_tests<double,float, is_diag, check_diag>();
	run_tests<float,float, is_diag, check_diag>();
	// run_tests<int,float, is_diag, check_diag>();

	//run_tests<double,int, is_diag, check_diag>();
	//run_tests<float,int, is_diag, check_diag>();

	// run_tests<int,int, is_diag, check_diag>();

    }
}
