#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

namespace nb = nanobind;

extern void init_pysystem(nb::module_& m);
extern void init_pyobjects(nb::module_& m);
extern void init_pybonded(nb::module_& m);
extern void init_pyloadfile(nb::module_& m);
extern void init_pysim(nb::module_& m);

NB_MODULE(marsmd, m) {
	init_pysystem(m);
	init_pyobjects(m);
	init_pybonded(m);
	init_pyloadfile(m);
	init_pysim(m);
}
