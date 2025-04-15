/*********************************************************************
 * @file  Vector3.h
 * 
 * @brief Declaration of templated Vector3_t class.
 *********************************************************************/
#pragma once
#include <memory>
#include <limits>
#include <type_traits> // for std::common_type<T,U>
#include <sstream>

#ifdef __CUDA_ARCH__
#include <cuda/std/limits>
template<typename T>
using numeric_limits = ::cuda::std::numeric_limits<T>;
#else
template<typename T>
using numeric_limits = ::std::numeric_limits<T>;
#endif

/**
 * 3D vector utility class with common operations implemented on CPU and GPU.
 * 
 * Implemented with 4D data storage for better GPU alignment; extra
 * data can be stored in fourth varaible this->w
 *
 * @tparam T the type of data stored in the four fields, x,y,z,w; T
 * Should usually be float or double.
g */
template<typename T>
class MY_ALIGN(4*sizeof(T)) Vector3_t {
public:
    HOST DEVICE inline Vector3_t<T>() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
	HOST DEVICE inline Vector3_t<T>(T s):x(s), y(s), z(s), w(s) {}
	HOST DEVICE inline Vector3_t<T>(const Vector3_t<T>& v):x(v.x), y(v.y), z(v.z), w(v.w)  {}
	HOST DEVICE inline Vector3_t<T>(T x, T y, T z) : x(x), y(y), z(z), w(0) {}
	HOST DEVICE inline Vector3_t<T>(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
	// HOST DEVICE inline Vector3_t<T>(const T* d) : x(d[0]), y(d[1]), z(d[2]), w(0) {}
        HOST DEVICE inline Vector3_t<T>(const float4 a) : x(a.x), y(a.y), z(a.z), w(a.w) {}

#if __cplusplus >= 201703L
	// TODO: test if c++17 constexpr performs better than std::common_type, otherwise get rid of this #if block
	template <typename U>
	    HOST DEVICE inline Vector3_t<typename std::common_type<T,U>::type> cross(const Vector3_t<U>& w) const {
	    if constexpr(sizeof(U) < sizeof(T)) {
		Vector3_t<T> v;
		v.x = y*w.z - z*w.y;
		v.y = z*w.x - x*w.z;
		v.z = x*w.y - y*w.x;
		return v;
	    } else {
		Vector3_t<U> v;
		v.x = y*w.z - z*w.y;
		v.y = z*w.x - x*w.z;
		v.z = x*w.y - y*w.x;
		return v;
	    }
	}
#else
	template <typename U>
	    HOST DEVICE inline Vector3_t<typename std::common_type<T,U>::type> cross(const Vector3_t<U>& w) const {
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> v;
	    v.x = y*w.z - z*w.y;
	    v.y = z*w.x - x*w.z;
	    v.z = x*w.y - y*w.x;
	    return v;
	}
#endif

	HOST DEVICE inline Vector3_t<T>& operator=(const Vector3_t<T>& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}
	HOST DEVICE inline Vector3_t<T>& operator=(const Vector3_t<T>&& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator+=(const Vector3_t<T>&& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator-=(const Vector3_t<T>&& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator/=(float s) {
		const float sinv = 1.0f/s;
		x *= sinv;
		y *= sinv;
		z *= sinv;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T> operator-() const {
		Vector3_t<T> v;
		v.x = -x;
		v.y = -y;
		v.z = -z;
		return v;
	}

	template<typename U> 
	    HOST DEVICE inline Vector3_t<std::common_type_t<T,U>> operator+(const Vector3_t<U>& w ) const {
	    using TU = typename std::common_type_t<T,U>;
	    Vector3_t<TU> v;
	    v.x = x + w.x;
	    v.y = y + w.y;
	    v.z = z + w.z;
	    return v;
	}

	template<typename U>
	HOST DEVICE inline Vector3_t<std::common_type_t<T,U>> operator-(const Vector3_t<U>& w ) const {
	    Vector3_t<std::common_type_t<T,U>> v;
	    v.x = x - w.x;
	    v.y = y - w.y;
	    v.z = z - w.z;
	    return v;
	}

	template<typename U>
	HOST DEVICE inline Vector3_t<std::common_type_t<T,U>> operator*(U&& s) const {
	    // TODO, throw exception if int
	    using TU = typename std::common_type_t<T,U>;
	    Vector3_t<TU> v;
	    v.x = s*x;
	    v.y = s*y;
	    v.z = s*z;
	    return v;
	}

	template<typename U>
	HOST DEVICE inline Vector3_t<std::common_type_t<T,U>> operator/(U& s) const {
	    const U inv = U(1.0)/s;
	    return (*this)*inv;
	}

	template<typename U>
	HOST DEVICE inline auto dot(const Vector3_t<U>& w) const { return x*w.x + y*w.y + z*w.z; }

	HOST DEVICE inline float length() const { return sqrtf(x*x + y*y + z*z); }
	HOST DEVICE inline float length2() const { return x*x + y*y + z*z; }
	HOST DEVICE inline float rLength() const {
		float l = length();
		if (l != 0.0f)
			return 1.0f / l;
		return 0.0f;
	}

	HOST DEVICE inline float rLength2() const {
		float l2 = length2();
		if (l2 != 0.0f) return 1.0f / l2;
		return 0.0f;
	}

	HOST DEVICE inline Vector3_t<T> element_floor() {
	    return Vector3_t<T>( floor(x), floor(y), floor(z) );
	}


	template<typename U>
	HOST DEVICE inline auto element_mult(const U w[]) {
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> ret( x*w[0], y*w[1], z*w[2]);
	    return ret;
	}
	template<typename U>
	HOST DEVICE inline auto element_mult(const Vector3_t<U>&& w) {
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> ret( x*w.x, y*w.y, z*w.z);
	    return ret;
	}
	template<typename U>
	HOST DEVICE inline auto element_mult(const Vector3_t<U>& w) {
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> ret( x*w.x, y*w.y, z*w.z);
	    return ret;
	}


	template<typename U>
	HOST DEVICE static inline auto element_mult(const Vector3_t<T>&& v, const Vector3_t<U>&& w) {
	    using TU = typename std::common_type<T,U>::type;
		Vector3_t<TU> ret(
			v.x*w.x,
			v.y*w.y,
			v.z*w.z);
		return ret;
	}

	template<typename U>
	HOST DEVICE static inline auto element_mult(const Vector3_t<T>&& v, const U w[]) {
	    using TU = typename std::common_type<T,U>::type;
		Vector3_t<TU> ret(
			v.x*w[0],
			v.y*w[1],
			v.z*w[2]);
		return ret;
	}

	HOST DEVICE static inline Vector3_t<T> element_sqrt(const Vector3_t<T>&& w) {
		Vector3_t<T> ret(
			sqrt(w.x),
			sqrt(w.y),
			sqrt(w.z));
		return ret;
	}
	
	// Numeric limits
	HOST DEVICE static inline T highest() { return numeric_limits<T>::max(); }
	HOST DEVICE static inline T lowest() { return numeric_limits<T>::lowest(); }

	// String
	HOST DEVICE inline void print() const {
		printf("%0.3f %0.3f %0.3f\n", x,y,z);
	}

	auto to_string_old() const {
	    char s[128];
	    sprintf(s, "%.10g %.10g %.10g (%.10g)", x, y, z, w);
	    s[127] = 0;
	    return std::string(s);
	}
	auto to_string() const {
	    std::ostringstream oss;
	    oss << x << " " << y << " " << z << " (" << w << ")";
	    return oss.str();
	}

	template<typename U>
	    HOST DEVICE inline bool operator==(U b) const {
	    return x == b.x && y == b.y && z == b.z && w == b.w;
	}

	T x, y, z, w; //append a member w	
};



/* template<typename T> */
/* HOST DEVICE inline Vector3_t<T> operator*(float s, Vector3_t<T> v) { */
/* 	v.x *= s; */
/* 	v.y *= s; */
/* 	v.z *= s; */
/* 	return v; */
/* } */

// template<typename T>
// HOST DEVICE inline Vector3_t<T> operator/(Vector3_t<T> v, float s) {
// 	const float sinv = 1.0f/s;
// 	return v*sinv;
// }

template<typename T, typename U>
HOST DEVICE inline auto operator/(const U&& s, const Vector3_t<T>&& v) {
    // TODO, throw exception if int
    using TU = typename std::common_type<T,U>::type;
    Vector3_t<TU> ret;
    ret.x = s / v.x;
    ret.y = s / v.y;
    ret.z = s / v.z;
    return ret;
}

// template<typename T, typename U>
// HOST DEVICE inline auto operator*(const U&& s, const Vector3_t<T>&& v) {
//     return v*s;
// }
// template<typename T, typename U>
// HOST DEVICE inline auto operator*(const U& s, const Vector3_t<T>& v) {
//     return v*s;
// }

// template<typename T>
// HOST DEVICE inline auto operator*(const float&& s, const Vector3_t<T>&& v) {
//     return v*s;
// }

template<typename T>
HOST DEVICE inline auto operator*(const float& s, const Vector3_t<T>& v) {
    return v*s;
}

// Provide common type for vectors
namespace std {
    template<typename T,typename U>
    struct common_type<Vector3_t<T>, Vector3_t<U>> {
	using type = Vector3_t<common_type_t<T,U>>;
    };
} 
