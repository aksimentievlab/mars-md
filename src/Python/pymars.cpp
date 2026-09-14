#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

namespace nb = nanobind;

extern void init_pysystem(nb::module_& m);
extern void init_pyobjects(nb::module_& m);
extern void init_pybonded(nb::module_& m);
extern void init_pynonbonded(nb::module_& m);
extern void init_pyloadfile(nb::module_& m);
extern void init_pysim(nb::module_& m);
extern void init_pytables(nb::module_& m);

/**
 * @brief The compiled MARS core, imported as `marsmd._core`.
 *
 * `marsmd/__init__.py` re-exports it; `marsmd` is the public name.
 */
NB_MODULE(_core, m) {
	m.doc() = "MARS simulation engine (compiled core)";

	init_pyloadfile(m);
	init_pyobjects(m);
	init_pybonded(m);
	init_pynonbonded(m);
	init_pytables(m);
	init_pysystem(m);
	init_pysim(m);
}
