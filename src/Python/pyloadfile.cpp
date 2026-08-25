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
  .dx -> .vdb -> .nvdb converter that doesn't exist yet.
	Calling it from Python surfaces that same error.
*/
#include "Objects/Grid.h"
#include "Objects/Tables.h"
#include "PyTypeCasters.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace MARS;

namespace {

using InArray = nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

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
BaseGrid<mars_real> grid_from_numpy(InArray values, Vector3 origin, Vector3 spacing) {
	const auto nx = static_cast<idx_t>(values.shape(0));
	const auto ny = static_cast<idx_t>(values.shape(1));
	const auto nz = static_cast<idx_t>(values.shape(2));

	Matrix3 basis(spacing.x, spacing.y, spacing.z);
	BaseGrid<mars_real> grid(basis, origin, nx, ny, nz);
	std::memcpy(grid.data(), values.data(), values.size() * sizeof(float));
	return grid;
}

/**
 * @brief Same as grid_from_numpy, but for a non-orthogonal grid: takes a
 *        full basis matrix instead of per-axis spacing.
 */
BaseGrid<mars_real> grid_from_numpy_basis(InArray values, Vector3 origin, Matrix3 basis) {
	const auto nx = static_cast<idx_t>(values.shape(0));
	const auto ny = static_cast<idx_t>(values.shape(1));
	const auto nz = static_cast<idx_t>(values.shape(2));

	BaseGrid<mars_real> grid(basis, origin, nx, ny, nz);
	std::memcpy(grid.data(), values.data(), values.size() * sizeof(float));
	return grid;
}

/**
 * @brief Grid values as a 3D numpy array, shape (nx, ny, nz), matching
 *        grid_from_numpy's expected layout.
 */
nb::ndarray<float, nb::numpy, nb::ndim<3>> grid_to_numpy(const BaseGrid<mars_real>& grid) {
	const size_t n = static_cast<size_t>(grid.size());
	float* data = new float[n];
	std::memcpy(data, grid.data(), n * sizeof(float));

	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });

	size_t shape[3] = {static_cast<size_t>(grid.nx()),
					   static_cast<size_t>(grid.ny()),
					   static_cast<size_t>(grid.nz())};
	return nb::ndarray<float, nb::numpy, nb::ndim<3>>(data, 3, shape, owner);
}

// BaseGrid<T>::index(ix, iy, iz) is private (host-only convenience is not
// part of its device-shared API), so the linear-index formula - z fastest,
// x slowest, matching numpy's default C order for shape (nx, ny, nz) - is
// duplicated here from its public nx()/ny()/nz() instead.
idx_t grid_flat_index(const BaseGrid<mars_real>& grid, nb::tuple ijk) {
	if (ijk.size() != 3)
		throw nb::index_error("Grid index must be a length-3 (ix, iy, iz) tuple");
	const auto ix = nb::cast<idx_t>(ijk[0]);
	const auto iy = nb::cast<idx_t>(ijk[1]);
	const auto iz = nb::cast<idx_t>(ijk[2]);
	return iz + iy * grid.nz() + ix * grid.ny() * grid.nz();
}

} // namespace

void declare_loadfile(nb::module_& m) {
	nb::enum_<GridFormat>(m, "GridFormat")
		.value("Dense", GridFormat::Dense)
		.value("Sparse", GridFormat::Sparse);

	nb::enum_<GridType>(m, "GridType")
		.value("Potential", GridType::Potential)
		.value("Diffusion", GridType::Diffusion)
		.value("PMF", GridType::PMF)
		.value("Force", GridType::Force)
		.value("Density", GridType::Density);

	nb::enum_<InterpolationOrder>(m, "InterpolationOrder")
		.value("Linear", InterpolationOrder::Linear)
		.value("Cubic", InterpolationOrder::Cubic);

	nb::class_<GridKey>(m, "GridKey")
		.def(nb::init<>())
		.def(nb::init<const std::string&, const GridFormat&>(), nb::arg("name"), nb::arg("format"))
		.def_rw("name", &GridKey::name)
		.def_rw("format", &GridKey::format)
		.def_rw("grid_id", &GridKey::grid_id)
		.def_rw("interpolation_order", &GridKey::interpolation_order)
		.def("is_valid", &GridKey::is_valid)
		.def("__repr__", [](const GridKey& gk) {
			return "GridKey(name='" + gk.name +
				   "', format=" + (gk.format == GridFormat::Dense ? "Dense" : "Sparse") +
				   ", grid_id=" + std::to_string(gk.grid_id) + ")";
		});

	//========================================================================
	// Grid - host-side dense grid data (BaseGrid<mars_real>)
	//========================================================================
	nb::class_<BaseGrid<mars_real>>(m, "Grid")
		.def(nb::init<>(), "Create an empty 1x1x1 grid")
		.def(
			"__init__",
			[](BaseGrid<mars_real>* self,
			   Vector3 origin,
			   Matrix3 basis,
			   idx_t nx,
			   idx_t ny,
			   idx_t nz) { new (self) BaseGrid<mars_real>(basis, origin, nx, ny, nz); },
			nb::arg("origin"),
			nb::arg("basis"),
			nb::arg("nx"),
			nb::arg("ny"),
			nb::arg("nz"),
			"Create a grid with an explicit basis matrix and dimensions")
		.def(nb::init<const Vector3&, float>(),
			 nb::arg("box_size"),
			 nb::arg("dx"),
			 "Create an orthogonal grid spanning box_size, centered at the origin, with "
			 "spacing dx")
		.def_static("from_numpy",
					&grid_from_numpy,
					nb::arg("values"),
					nb::arg("origin") = Vector3(0.0f, 0.0f, 0.0f),
					nb::arg("spacing") = Vector3(1.0f, 1.0f, 1.0f),
					"Build a dense grid from a 3D numpy array of shape (nx, ny, nz) - e.g. "
					"evaluated over np.meshgrid(x, y, z, indexing='ij') - with per-axis "
					"spacing. Use from_numpy_basis for a non-orthogonal grid.")
		.def_static("from_numpy_basis",
					&grid_from_numpy_basis,
					nb::arg("values"),
					nb::arg("origin"),
					nb::arg("basis"),
					"Build a dense grid from a 3D numpy array with a full (possibly "
					"non-orthogonal) basis matrix")
		.def("to_numpy", &grid_to_numpy, "Grid values as a 3D numpy array, shape (nx, ny, nz)")
		.def("nx", &BaseGrid<mars_real>::nx)
		.def("ny", &BaseGrid<mars_real>::ny)
		.def("nz", &BaseGrid<mars_real>::nz)
		.def("size", &BaseGrid<mars_real>::size)
		.def_prop_ro("origin",
					 static_cast<const Vector3& (BaseGrid<mars_real>::*)() const>(
						 &BaseGrid<mars_real>::origin))
		.def_prop_ro("basis",
					 static_cast<const Matrix3& (BaseGrid<mars_real>::*)() const>(
						 &BaseGrid<mars_real>::basis))
		.def("__getitem__", [](const BaseGrid<mars_real>& g, idx_t i) { return g[i]; })
		.def("__getitem__",
			 [](const BaseGrid<mars_real>& g, nb::tuple ijk) { return g[grid_flat_index(g, ijk)]; })
		.def("__setitem__", [](BaseGrid<mars_real>& g, idx_t i, float value) { g[i] = value; })
		.def("__setitem__",
			 [](BaseGrid<mars_real>& g, nb::tuple ijk, float value) {
				 g[grid_flat_index(g, ijk)] = value;
			 })
		.def("__len__", &BaseGrid<mars_real>::size)
		.def("__repr__", [](const BaseGrid<mars_real>& g) {
			return "Grid(nx=" + std::to_string(g.nx()) + ", ny=" + std::to_string(g.ny()) +
				   ", nz=" + std::to_string(g.nz()) + ")";
		});
	//========================================================================
	// GridManager - unified grid loading/lookup (dense grids only for now;
	// sparse/.vdb loading is deferred, see file docstring)
	//========================================================================
	nb::class_<GridManager>(m, "GridManager")
		.def(nb::init<>())
		.def("add_grid",
			 [](GridManager& gm, const std::string& name) { return gm.add_grid(name); },
			 nb::arg("name"))
		.def("add_dense_grid",
			 [](GridManager& gm, const std::string& filename) {
				 return gm.add_dense_grid(filename);
			 },
			 nb::arg("filename"),
			 "Load a dense grid from a .dx file")
		.def("add_sparse_grid",
			 [](GridManager& gm, const std::string& filename) {
				 return gm.add_sparse_grid(filename);
			 },
			 nb::arg("filename"),
			 "Not yet implemented - sparse/.vdb grid support is deferred (see Objects/Grid.h)")
		.def("get_grid_key",
			 [](GridManager& gm, const std::string& filename) {
				 return gm.get_grid_key(filename);
			 },
			 nb::arg("filename"))
		.def("has_grid",
			 [](GridManager& gm, const std::string& filename) { return gm.has_grid(filename); },
			 nb::arg("filename"))
		.def("get_grid_format",
			 [](GridManager& gm, idx_t grid_id) { return gm.get_grid_format(grid_id); },
			 nb::arg("grid_id"))
		.def("num_grids", [](const GridManager& gm) { return gm.num_grids(); })
		.def("build_device_arrays",
			 [](GridManager& gm) { gm.build_device_arrays(); },
			 "Replicate all loaded grids to device memory on every configured resource")
		.def("__repr__", [](const GridManager& gm) {
			return "GridManager(num_grids=" + std::to_string(gm.num_grids()) + ")";
		});
}

void init_pyloadfile(nb::module_& m) {
	declare_loadfile(m);

	nb::enum_<TabulatedType>(m, "TabulatedType")
		.value("NonBondedPair", TabulatedType::NonBondedPair)
		.value("Bond", TabulatedType::Bond)
		.value("Angle", TabulatedType::Angle)
		.value("Dihedral", TabulatedType::Dihedral)
		.value("Default", TabulatedType::Default);

	nb::class_<Table>(m, "Table")
		.def(nb::init<TabulatedType>(), nb::arg("type") = TabulatedType::Default)
		.def_rw("name", &Table::name)
		.def_rw("type", &Table::type)
		.def_rw("step_size", &Table::step_size)
		.def_rw("start", &Table::start)
		.def_rw("X", &Table::X)
		.def_rw("Y", &Table::Y)
		.def("read_file",
			 &Table::read_file,
			 nb::arg("file_name"),
			 nb::arg("name") = "",
			 "Load X/Y pairs from a two-column text file")
		.def("set_values",
			 &Table::set_values,
			 nb::arg("y_values"),
			 nb::arg("start"),
			 nb::arg("stop") = nb::none(),
			 nb::arg("step_size") = nb::none(),
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
