#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
#include "Objects/ParticleProperties.h"
#include "PyTypeCasters.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace MARS;

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
		.def(
			"__init__",
			[](Bond* self, const ParticleIO& i, const ParticleIO& j, const std::string& name) {
				new (self) Bond{};
				self->uids.uid1 = i.uid;
				self->uids.uid2 = j.uid;
				self->function_name = name;
			},
			nb::arg("i"),
			nb::arg("j"),
			nb::arg("bond"))
		/// Index-based form, for `.bd` topology files, which name raw indices.
		.def(
			"__init__",
			[](Bond* self, int ind1, int ind2, const std::string& name) {
				new (self) Bond{};
				self->ind1 = ind1;
				self->ind2 = ind2;
				self->function_name = name;
			},
			nb::arg("ind1"),
			nb::arg("ind2"),
			nb::arg("bond"))
		.def_rw("name", &Bond::function_name)
		.def_rw("form", &Bond::form)
		.def_rw("flag", &Bond::flag)
		.def_prop_ro("ind1", [](const Bond& b) { return b.ind1; })
		.def_prop_ro("ind2", [](const Bond& b) { return b.ind2; })
		/// Resolved by TablesRegistry; -1 until then, which the device reads as
		/// an out-of-bounds table lookup.
		.def_rw("function_index", &Bond::function_index)
		.def("__repr__", [](const Bond& b) { return "Bond(name='" + b.function_name + "')"; });
}

void declare_angle(nb::module_& m) {
	nb::class_<Angle>(m, "Angle")
		.def(nb::init<>())
		.def(
			"__init__",
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
		/// Index-based form, for `.bd` topology files, which name raw indices.
		.def(
			"__init__",
			[](Angle* self, int ind1, int ind2, int ind3, const std::string& name) {
				new (self) Angle{};
				self->ind1 = ind1;
				self->ind2 = ind2;
				self->ind3 = ind3;
				self->function_name = name;
				self->form = InteractionForm::Tabulated;
			},
			nb::arg("ind1"),
			nb::arg("ind2"),
			nb::arg("ind3"),
			nb::arg("angle"))
		.def_rw("name", &Angle::function_name)
		.def_rw("form", &Angle::form)
		.def_prop_ro("ind1", [](const Angle& a) { return a.ind1; })
		.def_prop_ro("ind2", [](const Angle& a) { return a.ind2; })
		.def_prop_ro("ind3", [](const Angle& a) { return a.ind3; })
		.def_rw("function_index", &Angle::function_index)
		.def("__repr__", [](const Angle& a) { return "Angle(name='" + a.function_name + "')"; });
}

void declare_dihedral(nb::module_& m) {
	nb::class_<Dihedral>(m, "Dihedral")
		.def(nb::init<>())
		.def(
			"__init__",
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
		/// Index-based form, for `.bd` topology files, which name raw indices.
		.def(
			"__init__",
			[](Dihedral* self, int ind1, int ind2, int ind3, int ind4, const std::string& name) {
				new (self) Dihedral{};
				self->ind1 = ind1;
				self->ind2 = ind2;
				self->ind3 = ind3;
				self->ind4 = ind4;
				self->function_name = name;
				self->form = InteractionForm::Tabulated;
			},
			nb::arg("ind1"),
			nb::arg("ind2"),
			nb::arg("ind3"),
			nb::arg("ind4"),
			nb::arg("dihedral"))
		.def_rw("name", &Dihedral::function_name)
		.def_rw("form", &Dihedral::form)
		.def_prop_ro("ind1", [](const Dihedral& d) { return d.ind1; })
		.def_prop_ro("ind2", [](const Dihedral& d) { return d.ind2; })
		.def_prop_ro("ind3", [](const Dihedral& d) { return d.ind3; })
		.def_prop_ro("ind4", [](const Dihedral& d) { return d.ind4; })
		.def_rw("function_index", &Dihedral::function_index)
		.def("__repr__",
			 [](const Dihedral& d) { return "Dihedral(name='" + d.function_name + "')"; });
}

void declare_exclude(nb::module_& m) {
	nb::class_<Exclude>(m, "Exclude")
		.def(nb::init<>())
		.def(
			"__init__",
			[](Exclude* self, const ParticleIO& i, const ParticleIO& j) {
				new (self) Exclude{};
				self->uids.uid1 = i.uid;
				self->uids.uid2 = j.uid;
			},
			nb::arg("i"),
			nb::arg("j"))
		/// Index-based form, for `.bd` topology files, which name raw indices.
		.def(nb::init<int, int>(), nb::arg("ind1"), nb::arg("ind2"))
		.def_prop_ro("ind1", [](const Exclude& e) { return e.ind1; })
		.def_prop_ro("ind2", [](const Exclude& e) { return e.ind2; })
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
		.def(
			"__init__",
			[](Restraint* self, const ParticleIO& i, Vector3 r0, float k) {
				new (self) Restraint{};
				self->uids.uid1 = i.uid;
				self->r0 = r0;
				self->k = k;
			},
			nb::arg("i"),
			nb::arg("r0"),
			nb::arg("k"))
		/// Index-based form, for `.bd` topology files, which name raw indices.
		.def(nb::init<int, Vector3, float>(), nb::arg("ind"), nb::arg("r0"), nb::arg("k"))
		.def_prop_ro("ind", [](const Restraint& r) { return r.ind; })
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
 * @example:
 * ```python
 * >>> from marsmd import BondedInteraction
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
		// Read-side snapshots. nanobind/stl/vector.h converts by value, so the
		// returned lists are copies - mutating them does nothing to the system.
		.def("get_bonds", &BondedInteractions::get_bonds)
		.def("get_angles", &BondedInteractions::get_angles)
		.def("get_dihedrals", &BondedInteractions::get_dihedrals)
		.def("get_exclusions", &BondedInteractions::get_exclusions)
		.def("get_restraints", &BondedInteractions::get_restraints)
		.def("make_exclusions",
			 &BondedInteractions::make_exclusions,
			 nb::arg("num_particles"),
			 nb::arg("exclusion_depth"),
			 "Generate exclusions by walking the bond graph to the given depth")
		.def("clear", &BondedInteractions::clear)
		.def("__repr__", [](const BondedInteractions& bi) {
			return "BondedInteraction(bonds=" + std::to_string(bi.get_num_bonds()) +
				   ", angles=" + std::to_string(bi.get_num_angles()) +
				   ", dihedrals=" + std::to_string(bi.get_num_dihedrals()) + ")";
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
	declare_bonded_interaction(m);
	declare_register_potential(m);
}
