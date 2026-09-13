#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/ParticleProperties.h"
#include "PyTypeCasters.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace MARS;
// ============================================================================
// NONBONDED INTERACTION BINDINGS
// ============================================================================
// A pair is declared between two ParticleType objects, not raw type ids -
// ParticleType.id isn't assigned until SimSystem::assign_particle_type_ids()
// runs, so an id read at construction time would be meaningless. Only the
// names are captured here; SimSystem::build_name_to_id_maps() resolves them
// (see PairNonBonded::resolve_type_names).s
void declare_nonbonded_interaction(nb::module_& m) {
	nb::class_<PairNonBonded>(m, "PairNonBonded")
		.def(
			"__init__",
			[](PairNonBonded* self,
			   const ParticleType& type_a,
			   const ParticleType& type_b,
			   const std::string& function_name) {
				new (self) PairNonBonded(type_a.name, type_b.name, function_name);
			},
			nb::arg("type_a"),
			nb::arg("type_b"),
			nb::arg("function_name"))
		.def_ro("type_name_1", &PairNonBonded::type_name_1)
		.def_ro("type_name_2", &PairNonBonded::type_name_2)
		.def_rw("name", &PairNonBonded::function_name)
		.def_rw("form", &PairNonBonded::form)
		.def("__repr__", [](const PairNonBonded& p) {
			return "PairNonBonded(type_a='" + p.type_name_1 + "', type_b='" + p.type_name_2 +
				   "', name='" + p.function_name + "')";
		});

	// `type_id` is not exposed: NonBondedInteractions::assign_id() overwrites
	// it with the term's own index, so it is an output, not an input.
	nb::class_<LongRangeNonBonded>(m, "LongRangeNonBonded")
		.def(nb::init<>())
		.def_rw("name", &LongRangeNonBonded::function_name)
		.def_rw("form", &LongRangeNonBonded::form)
		.def("__repr__", [](const LongRangeNonBonded& l) {
			return "LongRangeNonBonded(name='" + l.function_name + "')";
		});

	nb::class_<NonBondedInteractions>(m, "NonBondedInteractions")
		.def(nb::init<std::vector<PairNonBonded>, std::vector<LongRangeNonBonded>>(),
			 nb::arg("pair_nonbonded") = std::vector<PairNonBonded>{},
			 nb::arg("long_range_nonbonded") = std::vector<LongRangeNonBonded>{})
		.def("add_pair_nonbonded", &NonBondedInteractions::add_pair_nonbonded, nb::arg("pair"))
		.def("add_long_range_nonbonded",
			 &NonBondedInteractions::add_long_range_nonbonded,
			 nb::arg("long_range"))
		.def("get_num_pair_nonbonded", &NonBondedInteractions::get_num_pair_nonbonded)
		.def("get_num_long_range_nonbonded", &NonBondedInteractions::get_num_long_range_nonbonded)
		.def("__repr__", [](const NonBondedInteractions& nb_) {
			return "NonBondedInteractions(pairs=" + std::to_string(nb_.get_num_pair_nonbonded()) +
				   ", long_range=" + std::to_string(nb_.get_num_long_range_nonbonded()) + ")";
		});
}
void init_pynonbonded(nb::module_& m) {
	// Non-bonded interactions
	declare_nonbonded_interaction(m);
}
