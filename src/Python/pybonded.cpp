#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/ParticleProperties.h"
#include "PyTypeCasters.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ARBD;

// ============================================================================
// BONDED INTERACTION BINDINGS
// ============================================================================

void declare_bond(nb::module_& m) {
	nb::enum_<BondFlag>(m, "BondFlag")
		.value("DEFAULT", BondFlag::DEFAULT)
		.value("REPLACE", BondFlag::REPLACE)
		.value("ADD", BondFlag::ADD);

	nb::enum_<AnalyticalBondType>(m, "AnalyticalBondType")
		.value("Harmonic", AnalyticalBondType::Harmonic)
		.value("Morse", AnalyticalBondType::Morse)
		.value("FENE", AnalyticalBondType::FENE)
		.value("Half_Harmonic", AnalyticalBondType::Half_Harmonic)
		.value("WLCSK", AnalyticalBondType::WLCSK);

	nb::enum_<AnalyticalAngleType>(m, "AnalyticalAngleType")
		.value("Harmonic", AnalyticalAngleType::Harmonic)
		.value("Morse", AnalyticalAngleType::Morse)
		.value("FENE", AnalyticalAngleType::FENE)
		.value("Half_Harmonic", AnalyticalAngleType::Half_Harmonic)
		.value("WLCSK", AnalyticalAngleType::WLCSK);

	nb::enum_<AnalyticalDihedralType>(m, "AnalyticalDihedralType")
		.value("Harmonic", AnalyticalDihedralType::Harmonic)
		.value("Morse", AnalyticalDihedralType::Morse)
		.value("FENE", AnalyticalDihedralType::FENE)
		.value("Half_Harmonic", AnalyticalDihedralType::Half_Harmonic)
		.value("WLCSK", AnalyticalDihedralType::WLCSK);

	nb::enum_<InteractionForm>(m, "InteractionForm")
		.value("Grid", InteractionForm::Grid)
		.value("Tabulated", InteractionForm::Tabulated)
		.value("Analytical", InteractionForm::Analytical);

	// Bonded terms name their particles with real Particle objects, matching
	// arbdmodel's `add_bond(i=b1, j=b2, bond=...)`. The engine addresses
	// particles by index, but indices don't exist until
	// SimManager.stage_particles() fixes the ordering, so only the particles'
	// handles are captured here - see ParticleUids in
	// Interactions/BondedInteraction.h. `ind1`/`ind2` are therefore not
	// exposed: they are outputs of that resolution, not inputs.
	nb::class_<Bond>(m, "Bond")
		.def(nb::init<>())
		.def("__init__",
			 [](Bond* self, const ParticleIO& i, const ParticleIO& j, const std::string& name) {
				 new (self) Bond{};
				 self->uids.uid1 = i.uid;
				 self->uids.uid2 = j.uid;
				 self->function_name = name;
			 },
			 nb::arg("i"),
			 nb::arg("j"),
			 nb::arg("bond"))
		.def_rw("name", &Bond::function_name)
		.def_rw("form", &Bond::form)
		.def_rw("flag", &Bond::flag)
		.def("add_exclusion", &Bond::add_exclusion)
		.def("__repr__", [](const Bond& b) { return "Bond(name='" + b.function_name + "')"; });
}

void declare_angle(nb::module_& m) {
	nb::class_<Angle>(m, "Angle")
		.def(nb::init<>())
		.def("__init__",
			 [](Angle* self,
				const ParticleIO& i,
				const ParticleIO& j,
				const ParticleIO& k,
				const std::string& name) {
				 new (self) Angle{};
				 self->uids.uid1 = i.uid;
				 self->uids.uid2 = j.uid;
				 self->uids.uid3 = k.uid;
				 self->function_name = name;
			 },
			 nb::arg("i"),
			 nb::arg("j"),
			 nb::arg("k"),
			 nb::arg("angle"))
		.def_rw("name", &Angle::function_name)
		.def_rw("form", &Angle::form)
		.def("__repr__", [](const Angle& a) { return "Angle(name='" + a.function_name + "')"; });
}

void declare_dihedral(nb::module_& m) {
	nb::class_<Dihedral>(m, "Dihedral")
		.def(nb::init<>())
		.def("__init__",
			 [](Dihedral* self,
				const ParticleIO& i,
				const ParticleIO& j,
				const ParticleIO& k,
				const ParticleIO& l,
				const std::string& name) {
				 new (self) Dihedral{};
				 self->uids.uid1 = i.uid;
				 self->uids.uid2 = j.uid;
				 self->uids.uid3 = k.uid;
				 self->uids.uid4 = l.uid;
				 self->function_name = name;
			 },
			 nb::arg("i"),
			 nb::arg("j"),
			 nb::arg("k"),
			 nb::arg("l"),
			 nb::arg("dihedral"))
		.def_rw("name", &Dihedral::function_name)
		.def_rw("form", &Dihedral::form)
		.def("__repr__",
			 [](const Dihedral& d) { return "Dihedral(name='" + d.function_name + "')"; });
}

void declare_exclude(nb::module_& m) {
	nb::class_<Exclude>(m, "Exclude")
		.def(nb::init<>())
		.def("__init__",
			 [](Exclude* self, const ParticleIO& i, const ParticleIO& j) {
				 new (self) Exclude{};
				 self->uids.uid1 = i.uid;
				 self->uids.uid2 = j.uid;
			 },
			 nb::arg("i"),
			 nb::arg("j"))
		.def("__eq__", [](const Exclude& a, const Exclude& b) { return a == b; })
		.def("__ne__", [](const Exclude& a, const Exclude& b) { return a != b; })
		.def("__lt__", [](const Exclude& a, const Exclude& b) { return a < b; })
		.def("__repr__", [](const Exclude& e) {
			return "Exclude(ind1=" + std::to_string(e.ind1) + ", ind2=" + std::to_string(e.ind2) +
				   ")";
		});
}

void declare_restraint(nb::module_& m) {
	nb::class_<Restraint>(m, "Restraint")
		.def(nb::init<>())
		.def("__init__",
			 [](Restraint* self, const ParticleIO& i, Vector3 r0, float k) {
				 new (self) Restraint{};
				 self->uids.uid1 = i.uid;
				 self->r0 = r0;
				 self->k = k;
			 },
			 nb::arg("i"),
			 nb::arg("r0"),
			 nb::arg("k"))
		.def_rw("r0", &Restraint::r0)
		.def_rw("k", &Restraint::k)
		.def("__repr__", [](const Restraint& r) {
			return "Restraint(r0=" + r.r0.to_string() + ", k=" + std::to_string(r.k) + ")";
		});
}

// ============================================================================
// BONDED INTERACTION MANAGER BINDINGS
// ============================================================================

/**
 * @note Example usage (in Python):
 * ```python
 * >>> from arbd2v import BondedInteraction
 * >>> bi = BondedInteraction()
 * >>> print(bi)
 * BondedInteraction()
 */
void declare_bonded_interaction(nb::module_& m) {
	nb::class_<BondedInteractions>(m, "BondedInteractions")
		.def(nb::init<std::vector<Bond>,
					  std::vector<Angle>,
					  std::vector<Dihedral>,
					  std::vector<Exclude>,
					  std::vector<Restraint>>(),
			 nb::arg("bonds") = std::vector<Bond>{},
			 nb::arg("angles") = std::vector<Angle>{},
			 nb::arg("dihedrals") = std::vector<Dihedral>{},
			 nb::arg("exclusions") = std::vector<Exclude>{},
			 nb::arg("restraints") = std::vector<Restraint>{})
		.def("add_bond", &BondedInteractions::add_bond, nb::arg("bond"))
		.def("add_angle", &BondedInteractions::add_angle, nb::arg("angle"))
		.def("add_dihedral", &BondedInteractions::add_dihedral, nb::arg("dihedral"))
		.def("add_exclusion", &BondedInteractions::add_exclude, nb::arg("exclusion"))
		.def("add_restraint", &BondedInteractions::add_restraint, nb::arg("restraint"))
		.def("get_num_bonds", &BondedInteractions::get_num_bonds)
		.def("get_num_angles", &BondedInteractions::get_num_angles)
		.def("get_num_dihedrals", &BondedInteractions::get_num_dihedrals)
		.def("__repr__", [](const BondedInteractions& bi) {
			return "BondedInteraction(bonds=" + std::to_string(bi.get_num_bonds()) +
				   ", angles=" + std::to_string(bi.get_num_angles()) +
				   ", dihedrals=" + std::to_string(bi.get_num_dihedrals()) + ")";
		});
}

// ============================================================================
// NONBONDED INTERACTION BINDINGS
// ============================================================================
// Unlike Bond/Angle/Dihedral (per-run topology, staged into SimManager via
// send_bonded_interactions - see pysim.cpp), NonBondedInteractions lives on
// SimSystem itself (SimSystem::get_nonbonded_interactions(), pysystem.cpp):
// pair/long-range parameters are tied to particle type definitions, which
// are also SimSystem-owned.

// A pair is declared between two ParticleType objects, not raw type ids -
// ParticleType.id isn't assigned until SimSystem::assign_particle_type_ids()
// runs, so an id read at construction time would be meaningless. Only the
// names are captured here; SimSystem::build_name_to_id_maps() resolves them
// (see PairNonBonded::resolve_type_names).
void declare_nonbonded_interaction(nb::module_& m) {
	nb::class_<PairNonBonded>(m, "PairNonBonded")
		.def("__init__",
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

// ============================================================================
// POTENTIAL REGISTRATION BINDINGS
// ============================================================================

void declare_register_potential(nb::module_& m) {
	nb::class_<Register_Potential>(m, "RegisterPotential")
		.def(nb::init<>())
		.def("register_potential", &Register_Potential::register_potential)
		.def("get_id", &Register_Potential::get_id)
		.def("__repr__", [](const Register_Potential&) { return "RegisterPotential()"; });
}

// ============================================================================
// MAIN INITIALIZATION FUNCTION
// ============================================================================

void init_pybonded(nb::module_& m) {
	// Bonded interactions
	declare_bond(m);
	declare_angle(m);
	declare_dihedral(m);
	declare_exclude(m);
	declare_restraint(m);

	// Interaction managers
	declare_bonded_interaction(m);
	declare_nonbonded_interaction(m);
	declare_register_potential(m);
}
