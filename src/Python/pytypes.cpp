#include "Types/Array.h"
#include "Types/Matrix3.h"
#include "Types/Types.h"
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

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import array_to_matrix
 * >>> a = np.eye(3)
 * >>> mat = array_to_matrix(a)
 * ```
 * Rows in, columns out: Matrix3_t stores column vectors, so a numpy array
 * (read row-major) is transposed into the column-vector constructor.
 */
template<typename T>
Matrix3_t<T> array_to_matrix(py::array_t<T> a) {
	py::buffer_info info = a.request();
	if (info.ndim != 2 || info.shape[0] != 3 || info.shape[1] != 3)
		throw std::runtime_error("Matrix3 numpy array must have shape (3, 3)");

	T* ptr = static_cast<T*>(info.ptr);
	Vector3_t<T> row0(ptr[0], ptr[1], ptr[2]);
	Vector3_t<T> row1(ptr[3], ptr[4], ptr[5]);
	Vector3_t<T> row2(ptr[6], ptr[7], ptr[8]);
	return Matrix3_t<T>(Vector3_t<T>(row0.x, row1.x, row2.x),
						Vector3_t<T>(row0.y, row1.y, row2.y),
						Vector3_t<T>(row0.z, row1.z, row2.z));
}

template<typename T>
py::array_t<T> matrix_to_numpy_array(const Matrix3_t<T>& mat) {
	auto result = py::array_t<T>({3, 3});
	py::buffer_info info = result.request();
	T* ptr = static_cast<T*>(info.ptr);
	const auto& c0 = mat.ex();
	const auto& c1 = mat.ey();
	const auto& c2 = mat.ez();
	ptr[0] = c0.x;
	ptr[1] = c1.x;
	ptr[2] = c2.x;
	ptr[3] = c0.y;
	ptr[4] = c1.y;
	ptr[5] = c2.y;
	ptr[6] = c0.z;
	ptr[7] = c1.z;
	ptr[8] = c2.z;
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
// MATRIX3 BINDINGS
// ============================================================================

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import Matrix3
 * >>> identity = Matrix3()
 * >>> diag = Matrix3(1.0, 2.0, 3.0)             # diagonal (spacing) matrix
 * >>> basis = Matrix3(ex, ey, ez)               # from three Vector3 columns
 * >>> from_np = Matrix3(np.eye(3))
 * >>> v = basis.transform(Vector3(1.0, 0.0, 0.0))
 * ```
 */
template<typename T>
void declare_matrix(py::module& m, const std::string& typestr) {
	using Class = Matrix3_t<T>;
	using Vec = Vector3_t<T>;
	std::string pyclass_name = std::string("Matrix3_t_") + typestr;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<T>())
		.def(py::init<T, T, T>())
		.def(py::init<const Vec&, const Vec&, const Vec&>())
		.def(py::init([](py::array_t<T> a) { return array_to_matrix<T>(a); }))
		.def_property(
			"ex",
			[](const Class& mat) -> const Vec& { return mat.ex(); },
			[](Class& mat, const Vec& v) { mat.ex() = v; })
		.def_property(
			"ey",
			[](const Class& mat) -> const Vec& { return mat.ey(); },
			[](Class& mat, const Vec& v) { mat.ey() = v; })
		.def_property(
			"ez",
			[](const Class& mat) -> const Vec& { return mat.ez(); },
			[](Class& mat, const Vec& v) { mat.ez() = v; })
		.def(py::self * py::self)
		.def(py::self + py::self)
		.def(py::self * T())
		.def("transform", &Class::transform, py::arg("v"))
		.def("__mul__", [](const Class& mat, const Vec& v) { return mat.transform(v); })
		.def("transpose", &Class::transpose)
		.def("inverse", &Class::inverse)
		.def("det", &Class::det)
		.def("to_numpy", &matrix_to_numpy_array<T>)
		.def("__repr__", [](const Class& mat) {
			return "Matrix3(ex=" + mat.ex().to_string() + ", ey=" + mat.ey().to_string() +
				   ", ez=" + mat.ez().to_string() + ", det=" + std::to_string(mat.det()) + ")";
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

	// Matrix3 types
	declare_matrix<float>(m, "float");
	m.attr("Matrix3") = m.attr("Matrix3_t_float");
}
