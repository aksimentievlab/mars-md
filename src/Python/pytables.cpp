#include "Objects/Tables.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace MARS;

/**
 * @brief Bindings for TablesRegistry, the host-side tabulated-potential cache.
 *
 * Reached through SimSystem.get_tables_registry(). The registry owns device
 * buffers, so it is always exposed by reference, never copied into Python.
 *
 * @note The get_or_load_* methods return the function_index that a Bond,
 * Angle or Dihedral must carry; a term left at -1 is an out-of-bounds table
 * lookup on the device.
 *
 * @example
 * ```python
 * >>> reg = sys.get_tables_registry()
 * >>> idx = reg.get_or_load_bond("tab/bond.dat", "run.bd")
 * >>> reg.get_bond_name_to_idx()["tab/bond.dat"] == idx
 * True
 * ```
 */
void init_pytables(nb::module_& m) {
	nb::class_<TablesRegistry>(m, "TablesRegistry")
		.def(nb::init<>())
		.def("get_or_load_bond",
			 &TablesRegistry::get_or_load_bond,
			 nb::arg("file_name"),
			 nb::arg("config_file_path") = "",
			 "Load (or reuse) a tabulated bond potential; returns its function_index")
		.def("get_or_load_angle",
			 &TablesRegistry::get_or_load_angle,
			 nb::arg("file_name"),
			 nb::arg("config_file_path") = "",
			 "Load (or reuse) a tabulated angle potential; returns its function_index")
		.def("get_or_load_dihedral",
			 &TablesRegistry::get_or_load_dihedral,
			 nb::arg("file_name"),
			 nb::arg("config_file_path") = "",
			 "Load (or reuse) a tabulated dihedral potential; returns its function_index")
		.def("load_nonbonded",
			 &TablesRegistry::load_nonbonded,
			 nb::arg("file_name"),
			 nb::arg("config_file_path") = "",
			 "Load (or reuse) a tabulated nonbonded pair potential; returns the table index "
			 "a PairNonBonded takes")
		// In-memory registration: build a Table with set_values(), then hand it
		// over. Angle/dihedral X is in degrees, as in the .dat files.
		.def("add_nonbonded",
			 &TablesRegistry::add_nonbonded,
			 nb::arg("table"),
			 "Register an in-memory nonbonded pair table under table.name; returns its index")
		.def("add_bond",
			 &TablesRegistry::add_bond,
			 nb::arg("table"),
			 "Register an in-memory bond table under table.name; returns its function_index")
		.def("add_angle",
			 &TablesRegistry::add_angle,
			 nb::arg("table"),
			 "Register an in-memory angle table (X in degrees); returns its function_index")
		.def("add_dihedral",
			 &TablesRegistry::add_dihedral,
			 nb::arg("table"),
			 "Register an in-memory dihedral table (X in degrees); returns its function_index")
		// Name -> function_index maps. Copied into plain dicts on the way out.
		.def("get_bond_name_to_idx", &TablesRegistry::get_bond_name_to_idx)
		.def("get_angle_name_to_idx", &TablesRegistry::get_angle_name_to_idx)
		.def("get_dihedral_name_to_idx", &TablesRegistry::get_dihedral_name_to_idx)
		.def("get_bond_functions", &TablesRegistry::get_bond_functions)
		.def("get_angle_functions", &TablesRegistry::get_angle_functions)
		.def("get_dihedral_functions", &TablesRegistry::get_dihedral_functions)
		.def("get_nonbonded_functions", &TablesRegistry::get_nonbonded_functions)
		.def("get_nonbonded_name_to_idx", &TablesRegistry::get_nonbonded_name_to_idx)
		.def("build_device_arrays",
			 &TablesRegistry::build_device_arrays,
			 "Upload every loaded table to each configured resource")
		.def("__repr__", [](const TablesRegistry& reg) {
			return "TablesRegistry(bonds=" + std::to_string(reg.get_bond_functions().size()) +
				   ", angles=" + std::to_string(reg.get_angle_functions().size()) +
				   ", dihedrals=" + std::to_string(reg.get_dihedral_functions().size()) +
				   ", nonbonded=" + std::to_string(reg.get_nonbonded_functions().size()) + ")";
		});
}
