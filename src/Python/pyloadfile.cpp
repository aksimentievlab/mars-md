/**
  @brief Python bindings for grid and tabulated-potential loading.

  @note Example usage (in Python):
  ```python
  >>> from arbd2v import Grid, GridManager, Table, TabulatedType
  >>> import numpy as np

  # Dense grid from a .dx file, via GridManager (this is what SimSystem
  # actually uses internally - see SimSystem.get_grid_manager()):
  >>> gm = GridManager()
  >>> key = gm.add_dense_grid("potential.dx")
  >>> key.grid_id
  0

  # Dense grid built directly from a numpy array (e.g. evaluated over
  # np.meshgrid(..., indexing="ij") - see Grid.from_numpy's docstring):
  >>> values = np.zeros((32, 32, 32), dtype=np.float32)
  >>> grid = Grid.from_numpy(values, origin=Vector3(-16, -16, -16), spacing=1.0)

  # Tabulated potential built directly from a Y-value list/array instead of
  # a file, with X generated from start/stop or start/step_size:
  >>> table = Table(TabulatedType.NonBondedPair)
  >>> table.set_values([0.0, 1.0, 4.0, 9.0, 16.0], start=0.0, step_size=1.0)
  ```

  Sparse (.vdb/.nvdb) grids are intentionally not supported here yet:
  GridManager::add_sparse_grid (Objects/Grid.h) itself still raises
  NotImplementedError - the NanoVDB device path needs an offline
  .dx -> .vdb -> .nvdb converter that doesn't exist yet (see todo.md's
  Phase 4.1 design notes). Calling it from Python surfaces that same error.
*/
#include "Objects/Grid.h"
#include "Objects/Tables.h"
#include "PyTypeCasters.h"

// pybind11 core
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

namespace {

/**
 * @brief Build a dense Grid directly from a 3D numpy array, e.g. one
 *        evaluated over np.meshgrid(..., indexing="ij") - matching
 *        BaseGrid's own linear index convention (z fastest, x slowest), the
 *        same order numpy uses by default for a C-contiguous array of shape
 *        (nx, ny, nz).
 * @param values Array of shape (nx, ny, nz).
 * @param origin World-space position of grid index (0, 0, 0).
 * @param spacing Per-axis grid spacing (dx, dy, dz); use `basis` instead for
 *        a non-orthogonal grid.
 */
BaseGrid<float> grid_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> values,
								Vector3 origin,
								Vector3 spacing) {
	py::buffer_info info = values.request();
	if (info.ndim != 3) {
		throw std::runtime_error(
			"Grid.from_numpy: values must be a 3D array of shape (nx, ny, nz); "
			"got ndim=" +
			std::to_string(info.ndim));
	}

	const auto nx = static_cast<idx_t>(info.shape[0]);
	const auto ny = static_cast<idx_t>(info.shape[1]);
	const auto nz = static_cast<idx_t>(info.shape[2]);

	Matrix3 basis(spacing.x, spacing.y, spacing.z);
	BaseGrid<float> grid(basis, origin, nx, ny, nz);
	std::memcpy(grid.data(), info.ptr, static_cast<size_t>(info.size) * sizeof(float));
	return grid;
}

/**
 * @brief Same as grid_from_numpy, but for a non-orthogonal grid: takes a
 *        full basis matrix instead of per-axis spacing.
 */
BaseGrid<float> grid_from_numpy_basis(
	py::array_t<float, py::array::c_style | py::array::forcecast> values,
	Vector3 origin,
	Matrix3 basis) {
	py::buffer_info info = values.request();
	if (info.ndim != 3) {
		throw std::runtime_error(
			"Grid.from_numpy_basis: values must be a 3D array of shape (nx, ny, nz); "
			"got ndim=" +
			std::to_string(info.ndim));
	}

	const auto nx = static_cast<idx_t>(info.shape[0]);
	const auto ny = static_cast<idx_t>(info.shape[1]);
	const auto nz = static_cast<idx_t>(info.shape[2]);

	BaseGrid<float> grid(basis, origin, nx, ny, nz);
	std::memcpy(grid.data(), info.ptr, static_cast<size_t>(info.size) * sizeof(float));
	return grid;
}

/**
 * @brief Grid values as a 3D numpy array, shape (nx, ny, nz), matching
 *        grid_from_numpy's expected layout.
 */
py::array_t<float> grid_to_numpy(const BaseGrid<float>& grid) {
	py::array_t<float> result({static_cast<py::ssize_t>(grid.nx()),
							   static_cast<py::ssize_t>(grid.ny()),
							   static_cast<py::ssize_t>(grid.nz())});
	py::buffer_info info = result.request();
	std::memcpy(info.ptr, grid.data(), static_cast<size_t>(grid.size()) * sizeof(float));
	return result;
}

// BaseGrid<T>::index(ix, iy, iz) is private (host-only convenience is not
// part of its device-shared API), so the linear-index formula - z fastest,
// x slowest, matching numpy's default C order for shape (nx, ny, nz) - is
// duplicated here from its public nx()/ny()/nz() instead.
idx_t grid_flat_index(const BaseGrid<float>& grid, py::tuple ijk) {
	if (ijk.size() != 3)
		throw py::index_error("Grid index must be a length-3 (ix, iy, iz) tuple");
	const auto ix = ijk[0].cast<idx_t>();
	const auto iy = ijk[1].cast<idx_t>();
	const auto iz = ijk[2].cast<idx_t>();
	return iz + iy * grid.nz() + ix * grid.ny() * grid.nz();
}

} // namespace

void declare_loadfile(py::module& m) {
	py::enum_<GridFormat>(m, "GridFormat")
		.value("Dense", GridFormat::Dense)
		.value("Sparse", GridFormat::Sparse);

	py::enum_<GridType>(m, "GridType")
		.value("Potential", GridType::Potential)
		.value("Diffusion", GridType::Diffusion)
		.value("PMF", GridType::PMF)
		.value("Force", GridType::Force)
		.value("Density", GridType::Density);

	py::enum_<InterpolationOrder>(m, "InterpolationOrder")
		.value("Linear", InterpolationOrder::Linear)
		.value("Cubic", InterpolationOrder::Cubic);

	py::class_<GridKey>(m, "GridKey")
		.def(py::init<>())
		.def(py::init<const std::string&, const GridFormat&>(), py::arg("name"), py::arg("format"))
		.def_readwrite("name", &GridKey::name)
		.def_readwrite("format", &GridKey::format)
		.def_readwrite("grid_id", &GridKey::grid_id)
		.def_readwrite("interpolation_order", &GridKey::interpolation_order)
		.def("is_valid", &GridKey::is_valid)
		.def("__repr__", [](const GridKey& gk) {
			return "GridKey(name='" + gk.name +
				   "', format=" + (gk.format == GridFormat::Dense ? "Dense" : "Sparse") +
				   ", grid_id=" + std::to_string(gk.grid_id) + ")";
		});

	//========================================================================
	// Grid - host-side dense grid data (BaseGrid<float>)
	//========================================================================
	py::class_<BaseGrid<float>>(m, "Grid")
		.def(py::init<>(), "Create an empty 1x1x1 grid")
		.def(py::init([](Vector3 origin, Matrix3 basis, idx_t nx, idx_t ny, idx_t nz) {
				 return BaseGrid<float>(basis, origin, nx, ny, nz);
			 }),
			 py::arg("origin"),
			 py::arg("basis"),
			 py::arg("nx"),
			 py::arg("ny"),
			 py::arg("nz"),
			 "Create a grid with an explicit basis matrix and dimensions")
		.def(py::init<const Vector3&, float>(),
			 py::arg("box_size"),
			 py::arg("dx"),
			 "Create an orthogonal grid spanning box_size, centered at the origin, with "
			 "spacing dx")
		.def_static("from_numpy",
					&grid_from_numpy,
					py::arg("values"),
					py::arg("origin") = Vector3(0.0f, 0.0f, 0.0f),
					py::arg("spacing") = Vector3(1.0f, 1.0f, 1.0f),
					"Build a dense grid from a 3D numpy array of shape (nx, ny, nz) - e.g. "
					"evaluated over np.meshgrid(x, y, z, indexing='ij') - with per-axis "
					"spacing. Use from_numpy_basis for a non-orthogonal grid.")
		.def_static("from_numpy_basis",
					&grid_from_numpy_basis,
					py::arg("values"),
					py::arg("origin"),
					py::arg("basis"),
					"Build a dense grid from a 3D numpy array with a full (possibly "
					"non-orthogonal) basis matrix")
		.def("to_numpy", &grid_to_numpy, "Grid values as a 3D numpy array, shape (nx, ny, nz)")
		.def("nx", &BaseGrid<float>::nx)
		.def("ny", &BaseGrid<float>::ny)
		.def("nz", &BaseGrid<float>::nz)
		.def("size", &BaseGrid<float>::size)
		.def_property_readonly("origin",
								static_cast<const Vector3& (BaseGrid<float>::*)() const>(
									&BaseGrid<float>::origin))
		.def_property_readonly(
			"basis",
			static_cast<const Matrix3& (BaseGrid<float>::*)() const>(&BaseGrid<float>::basis))
		.def("__getitem__",
			 [](const BaseGrid<float>& g, idx_t i) { return g[i]; })
		.def("__getitem__",
			 [](const BaseGrid<float>& g, py::tuple ijk) { return g[grid_flat_index(g, ijk)]; })
		.def("__setitem__",
			 [](BaseGrid<float>& g, idx_t i, float value) { g[i] = value; })
		.def("__setitem__",
			 [](BaseGrid<float>& g, py::tuple ijk, float value) {
				 g[grid_flat_index(g, ijk)] = value;
			 })
		.def("__len__", &BaseGrid<float>::size)
		.def("__repr__", [](const BaseGrid<float>& g) {
			return "Grid(nx=" + std::to_string(g.nx()) + ", ny=" + std::to_string(g.ny()) +
				   ", nz=" + std::to_string(g.nz()) + ")";
		});

	//========================================================================
	// GridManager - unified grid loading/lookup (dense grids only for now;
	// sparse/.vdb loading is deferred, see file docstring)
	//========================================================================
	py::class_<GridManager>(m, "GridManager")
		.def(py::init<>())
		.def("add_grid",
			 &GridManager::add_grid,
			 py::arg("name"),
			 "Load a grid, dispatching on file extension (.dx -> dense; anything else -> "
			 "sparse, which currently raises NotImplementedError)")
		.def("add_dense_grid",
			 &GridManager::add_dense_grid,
			 py::arg("filename"),
			 "Load a dense grid from a .dx file")
		.def("add_sparse_grid",
			 &GridManager::add_sparse_grid,
			 py::arg("filename"),
			 "Not yet implemented - sparse/.vdb grid support is deferred (see Objects/Grid.h)")
		.def("get_grid_key", &GridManager::get_grid_key, py::arg("filename"))
		.def("has_grid", &GridManager::has_grid, py::arg("filename"))
		.def("get_grid_format", &GridManager::get_grid_format, py::arg("grid_id"))
		.def("num_grids", &GridManager::num_grids)
		.def("build_device_arrays",
			 &GridManager::build_device_arrays,
			 "Replicate all loaded grids to device memory on every configured resource")
		.def("__repr__", [](const GridManager& gm) {
			return "GridManager(num_grids=" + std::to_string(gm.num_grids()) + ")";
		});
}

void init_pyloadfile(py::module_& m) {
	declare_loadfile(m);

	py::enum_<TabulatedType>(m, "TabulatedType")
		.value("NonBondedPair", TabulatedType::NonBondedPair)
		.value("Bond", TabulatedType::Bond)
		.value("Angle", TabulatedType::Angle)
		.value("Dihedral", TabulatedType::Dihedral)
		.value("Default", TabulatedType::Default);

	py::class_<Table>(m, "Table")
		.def(py::init<TabulatedType>(), py::arg("type") = TabulatedType::Default)
		.def_readwrite("name", &Table::name)
		.def_readwrite("type", &Table::type)
		.def_readwrite("step_size", &Table::step_size)
		.def_readwrite("start", &Table::start)
		.def_readwrite("X", &Table::X)
		.def_readwrite("Y", &Table::Y)
		.def("read_file",
			 &Table::read_file,
			 py::arg("file_name"),
			 py::arg("name") = "",
			 "Load X/Y pairs from a two-column text file")
		.def("set_values",
			 &Table::set_values,
			 py::arg("y_values"),
			 py::arg("start"),
			 py::arg("stop") = py::none(),
			 py::arg("step_size") = py::none(),
			 "Populate the table directly from a Y-value list or numpy array (no file), "
			 "with X generated evenly from start using either stop or step_size (exactly "
			 "one of the two)")
		.def("__repr__", [](const Table& t) {
			const char* type_name = "Default";
			switch (t.type) {
			case TabulatedType::NonBondedPair:
				type_name = "NonBondedPair";
				break;
			case TabulatedType::Bond:
				type_name = "Bond";
				break;
			case TabulatedType::Angle:
				type_name = "Angle";
				break;
			case TabulatedType::Dihedral:
				type_name = "Dihedral";
				break;
			case TabulatedType::Default:
				type_name = "Default";
				break;
			}
			return "Table(name='" + t.name + "', type=" + type_name +
				   ", size=" + std::to_string(t.Y.size()) + ")";
		});
}
