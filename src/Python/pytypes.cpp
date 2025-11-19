#include "Types/Array.h"
#include "Types/Vector3.h"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ARBD;

// ============================================================================
// NUMPY CONVERSION UTILITIES
// ============================================================================

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import array_to_vector
 * >>> a = np.array([1.0, 0.0, 0.0])
 * >>> v = array_to_vector(a)
 * >>> print(v)
 * Vector3(1.0, 0.0, 0.0)
 */
template<typename T>
Vector3_t<T> array_to_vector(py::array_t<T> a) {
	py::buffer_info info = a.request();
	if (info.ndim != 1)
		throw std::runtime_error("Number of dimensions must be one");
	if (info.size < 3 || info.size > 4)
		throw std::runtime_error("Size of array must be 3 or 4");

	T* ptr = static_cast<T*>(info.ptr);
	if (info.size == 3)
		return Vector3_t<T>(ptr[0], ptr[1], ptr[2]);
	else
		return Vector3_t<T>(ptr[0], ptr[1], ptr[2], ptr[3]);
}

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import array_to_vector_arr
 * >>> a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
 * >>> v = array_to_vector_arr(a)
 * >>> print(v)
 * Vector3Array([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
 * ```
 */
template<typename T>
Array<Vector3_t<T>> array_to_vector_arr(py::array_t<T> a) {
	py::buffer_info info = a.request();
	if (info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");
	if (info.shape[1] < 3 || info.shape[1] > 4)
		throw std::runtime_error("Second dimension of numpy array must contain 3 or 4 elements");

	T* ptr = static_cast<T*>(info.ptr);
	Array<Vector3_t<T>> arr(static_cast<size_t>(info.shape[0]));

	if (info.shape[1] == 3) {
		for (size_t i = 0; i < info.shape[0]; ++i) {
			size_t j = i * info.shape[1];
			arr[i] = Vector3_t<T>(ptr[j], ptr[j + 1], ptr[j + 2]);
		}
	} else {
		for (size_t i = 0; i < info.shape[0]; ++i) {
			size_t j = i * info.shape[1];
			arr[i] = Vector3_t<T>(ptr[j], ptr[j + 1], ptr[j + 2], ptr[j + 3]);
		}
	}
	return arr;
}

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import vector_arr_to_numpy_array
 * >>> v = Vector3Array([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
 * >>> a = vector_arr_to_numpy_array(v)
 * >>> print(a)
 * array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
 */
template<typename T>
py::array_t<T> vector_arr_to_numpy_array(const Array<Vector3_t<T>>& arr) {
	auto result = py::array_t<T>({static_cast<py::ssize_t>(arr.size()), 3});
	py::buffer_info info = result.request();
	T* ptr = static_cast<T*>(info.ptr);

	for (size_t i = 0; i < arr.size(); ++i) {
		size_t j = i * 3;
		ptr[j] = arr[i].x;
		ptr[j + 1] = arr[i].y;
		ptr[j + 2] = arr[i].z;
	}

	return result;
}

// ============================================================================
// VECTOR3 BINDINGS
// ============================================================================

template<typename T>
void declare_vector(py::module& m, const std::string& typestr) {
	using Class = Vector3_t<T>;
	std::string pyclass_name = std::string("Vector3_t_") + typestr;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<T>())
		.def(py::init<T, T, T>())
		.def(py::init([](py::array_t<T> a) { return array_to_vector<T>(a); }))
		// Operators
		.def(py::self + py::self)
		.def(py::self *= float())
		.def(float() * py::self)
		.def(py::self * float())
		.def(-py::self)
		// Accessors
		.def_readwrite("x", &Class::x)
		.def_readwrite("y", &Class::y)
		.def_readwrite("z", &Class::z)
		.def_readwrite("w", &Class::t)
		// Conversions
		.def("__repr__", &Class::to_string)
		.def("__getitem__",
			 [](const Class& v, int i) {
				 if (i < 0 || i >= 4)
					 throw py::index_error();
				 return (&v.x)[i];
			 })
		.def("__setitem__", [](Class& v, int i, T value) {
			if (i < 0 || i >= 4)
				throw py::index_error();
			(&v.x)[i] = value;
		});
}
/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import Vector3Array
 * >>> v = Vector3Array(3)
 * >>> v[0] = [1.0, 0.0, 0.0]
 * >>> v[1] = [0.0, 1.0, 0.0]
 * >>> v[2] = [0.0, 0.0, 1.0]
 * >>> print(v)
 * Vector3Array([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
 * ```
 */
template<typename T>
void declare_array(py::module& m, const std::string& typestr) {
	using Class = Array<Vector3_t<T>>;
	std::string pyclass_name = std::string("Vector3_Array_t_") + typestr;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<size_t>(), py::arg("size"))
		.def(py::init([](py::array_t<T> a) { return array_to_vector_arr<T>(a); }))
		// .def("as_array", [](Array<Vector3_t<T>>& a) { return vector_arr_to_numpy_array<T>(a); })
		.def("size", &Class::size)
		.def("__len__", &Class::size)
		.def("__getitem__",
			 [](const Class& a, size_t i) {
				 if (i >= a.size())
					 throw py::index_error();
				 return a[i];
			 })
		.def("__setitem__", [](Class& a, size_t i, const Vector3_t<T>& v) {
			if (i >= a.size())
				throw py::index_error();
			a[i] = v;
		});
}

// ============================================================================
// MAIN INITIALIZATION FUNCTION
// ============================================================================

void init_pytypes(py::module_& m) {
	// Vector3 types
	declare_vector<int>(m, "int");
	declare_vector<float>(m, "float");
	declare_vector<double>(m, "double");
	// Create aliases
	m.attr("Vector3") = m.attr("Vector3_t_float");

	// Vector3 arrays
	declare_array<int>(m, "int");
	declare_array<float>(m, "float");
	declare_array<double>(m, "double");
	m.attr("VectorArr") = m.attr("Vector3_Array_t_float");
}
