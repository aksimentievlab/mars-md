#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern void init_pysystem(py::module_& m);
extern void init_pyobjects(py::module_& m);
extern void init_pybonded(py::module_& m);
extern void init_pyloadfile(py::module_& m);
extern void init_pysim(py::module_& m);

PYBIND11_MODULE(arbd, m) {
	init_pysystem(m);
	init_pyobjects(m);
	init_pybonded(m);
	init_pyloadfile(m);
	init_pysim(m);
}
