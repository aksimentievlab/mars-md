#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern void init_pytypes(py::module_& m);
extern void init_pyconfig(py::module_& m);

PYBIND11_MODULE(pyarbd, m) {
	init_pytypes(m);
	init_pyconfig(m);
}
