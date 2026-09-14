#include "Interactions/NonBondedInteraction.h"
#include "Objects/ParticleProperties.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace MARS;

/**
 * @brief Bindings for the nonbonded pair list, SimSystem.get_nonbonded_interactions().
 *
 * @example
 * ```python
 * >>> pairs = sys.get_nonbonded_interactions()
 * >>> pairs.add_pair_nonbonded(PairNonBonded(A, B, "coulomb"))
 * >>> idx = sys.get_tables_registry().add_nonbonded(table)
 * >>> pairs.add_pair_nonbonded(PairNonBonded(A, B, table.name, idx))
 * ```
 */
void init_pynonbonded(nb::module_& m) {
	nb::class_<PairNonBonded>(m, "PairNonBonded")
		.def(
			"__init__",
			[](PairNonBonded* self,
			   const ParticleType& type_a,
			   const ParticleType& type_b,
			   const std::string& function_name,
			   int table_index) {
				new (self) PairNonBonded(type_a.name, type_b.name, function_name, table_index);
			},
			nb::arg("type_a"),
			nb::arg("type_b"),
			nb::arg("function_name"),
			nb::arg("table_index") = -1,
			"Pair between two ParticleType objects; omit table_index for an analytical term")
		.def(
			"__init__",
			[](PairNonBonded* self,
			   int type_id_1,
			   int type_id_2,
			   const std::string& function_name,
			   int table_index) {
				new (self) PairNonBonded(type_id_1, type_id_2, function_name, table_index);
			},
			nb::arg("type_id_1"),
			nb::arg("type_id_2"),
			nb::arg("function_name"),
			nb::arg("table_index") = -1,
			"Pair between two type ids; omit table_index for an analytical term")
		.def_ro("type_name_1", &PairNonBonded::type_name_1)
		.def_ro("type_name_2", &PairNonBonded::type_name_2)
		.def_ro("type_id_1", &PairNonBonded::type_id_1)
		.def_ro("type_id_2", &PairNonBonded::type_id_2)
		.def_ro("name", &PairNonBonded::function_name)
		.def_ro("form", &PairNonBonded::form)
		.def_ro("function_index", &PairNonBonded::function_index)
		.def("is_analytical", &PairNonBonded::is_analytical)
		.def("__repr__", [](const PairNonBonded& p) {
			const std::string a =
				p.type_name_1.empty() ? std::to_string(p.type_id_1) : p.type_name_1;
			const std::string b =
				p.type_name_2.empty() ? std::to_string(p.type_id_2) : p.type_name_2;
			return "PairNonBonded(" + a + ", " + b + ", '" + p.function_name + "'" +
				   (p.is_analytical() ? "" : ", table_index=" + std::to_string(p.function_index)) +
				   ")";
		});

	nb::class_<SolventParams>(m, "SolventParams")
		.def(nb::init<>())
		.def_rw("debye_length", &SolventParams::debye_length)
		.def_rw("dielectric", &SolventParams::dielectric)
		.def_rw("onck_kappa", &SolventParams::onck_kappa)
		.def_rw("onck_sz", &SolventParams::onck_sz)
		.def_rw("onck_z", &SolventParams::onck_z)
		.def("__repr__", [](const SolventParams& p) {
			return "SolventParams(debye_length=" + std::to_string(p.debye_length) +
				   ", dielectric=" + std::to_string(p.dielectric) +
				   ", onck_kappa=" + std::to_string(p.onck_kappa) + ")";
		});

	nb::class_<NonBondedInteractions>(m, "NonBondedInteractions")
		.def(nb::init<>())
		.def("set_solvent_params",
			 &NonBondedInteractions::set_solvent_params,
			 nb::arg("params"),
			 "Screening constants of the Debye-Huckel and Onck terms")
		.def("get_solvent_params", &NonBondedInteractions::get_solvent_params)
		.def("add_pair_nonbonded",
			 &NonBondedInteractions::add_pair_nonbonded,
			 nb::arg("pair"),
			 "Append a pair; a pair already declared is kept and this one ignored")
		.def("get_pair_nonbonded",
			 &NonBondedInteractions::get_pair_nonbonded,
			 "Copy of the declared pairs")
		.def("get_num_pair_nonbonded", &NonBondedInteractions::get_num_pair_nonbonded)
		.def("enabled_terms",
			 &NonBondedInteractions::enabled_terms,
			 "Bitmask of every pair term the declared pairs use")
		.def("__repr__", [](const NonBondedInteractions& nb_) {
			return "NonBondedInteractions(pairs=" + std::to_string(nb_.get_num_pair_nonbonded()) +
				   ")";
		});
}
