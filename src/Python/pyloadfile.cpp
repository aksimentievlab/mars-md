/**
  @brief Python bindings for loadfile function. Load Grid files or Tabulated files into ARBD.
  @note Example usage (in Python):
  ```python
  >>> from arbd2v import loadfile
  >>> loadfile("grid.dx")
  >>> loadfile("tabulated.txt")
  ```
*/
#include "Objects/Grid.h"
#include "SimParam.h"

// pybind11 core
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

void declare_loadfile(py::module& m) {
	py::enum_<GridFormat>(m, "GridFormat")
		.value("Dense", GridFormat::Dense)
		.value("Sparse", GridFormat::Sparse);

	py::enum_<InteractionForm>(m, "InteractionForm")
		.value("Grid", InteractionForm::Grid)
		.value("Tabulated", InteractionForm::Tabulated);

	py::enum_<InterpolationOrder>(m, "InterpolationOrder")
		.value("Linear", InterpolationOrder::Linear)
		.value("Cubic", InterpolationOrder::Cubic);

	py::class_<GridKey>(m, "GridKey")
		.def(py::init<const std::string&>(), py::arg("name"))
		.def(py::init<const std::string&, const GridFormat&>(), py::arg("name"), py::arg("format"))
		.def_readwrite("name", &GridKey::name)
		.def_readwrite("format", &GridKey::format)
		.def_readwrite("grid_id", &GridKey::grid_id)
		.def("__repr__", [](const GridKey& gk) {
			return "GridKey(name='" + gk.name +
				   "', format=" + (gk.format == GridFormat::Dense ? "Dense" : "Sparse") +
				   ", grid_id=" + std::to_string(gk.grid_id) + ")";
		});
}
void init_pyloadfile(py::module_& m) {
	declare_loadfile(m);
}
