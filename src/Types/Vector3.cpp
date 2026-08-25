#include "Vector3.h"

#ifdef HOST_GUARD

namespace MARS {

template<typename T>
HOST void Vector3_t<T>::print() const {
	LOGINFO("%0.3f %0.3f %0.3f",
			static_cast<float>(x),
			static_cast<float>(y),
			static_cast<float>(z));
}

template<typename T>
HOST std::string Vector3_t<T>::to_string() const {
	std::ostringstream oss;
	oss << x << " " << y << " " << z << " (" << t << ")";
	return oss.str();
}

// Explicit instantiations for common types
template HOST void Vector3_t<float>::print() const;
template HOST void Vector3_t<double>::print() const;
template HOST void Vector3_t<int>::print() const;

template std::string Vector3_t<float>::to_string() const;
template std::string Vector3_t<double>::to_string() const;
template std::string Vector3_t<int>::to_string() const;

} // namespace MARS

#endif // HOST_GUARD
