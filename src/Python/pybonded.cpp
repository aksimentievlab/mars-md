#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/ParticleProperties.h"
#include "PyTypeCasters.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ARBD;

// ============================================================================
// BONDED INTERACTION BINDINGS
// ============================================================================

void declare_bond(py::module& m) {
	py::enum_<BondFlag>(m, "BondFlag")
		.value("DEFAULT", BondFlag::DEFAULT)
		.value("REPLACE", BondFlag::REPLACE)
		.value("ADD", BondFlag::ADD);

	py::enum_<AnalyticalBondType>(m, "AnalyticalBondType")
		.value("Harmonic", AnalyticalBondType::Harmonic)
		.value("Morse", AnalyticalBondType::Morse)
		.value("FENE", AnalyticalBondType::FENE)
		.value("Half_Harmonic", AnalyticalBondType::Half_Harmonic)
		.value("WLCSK", AnalyticalBondType::WLCSK);

	py::enum_<AnalyticalAngleType>(m, "AnalyticalAngleType")
		.value("Harmonic", AnalyticalAngleType::Harmonic)
		.value("Morse", AnalyticalAngleType::Morse)
		.value("FENE", AnalyticalAngleType::FENE)
		.value("Half_Harmonic", AnalyticalAngleType::Half_Harmonic)
		.value("WLCSK", AnalyticalAngleType::WLCSK);

	py::enum_<AnalyticalDihedralType>(m, "AnalyticalDihedralType")
		.value("Harmonic", AnalyticalDihedralType::Harmonic)
		.value("Morse", AnalyticalDihedralType::Morse)
		.value("FENE", AnalyticalDihedralType::FENE)
		.value("Half_Harmonic", AnalyticalDihedralType::Half_Harmonic)
		.value("WLCSK", AnalyticalDihedralType::WLCSK);

	py::enum_<InteractionForm>(m, "InteractionForm")
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
	py::class_<Bond>(m, "Bond")
		.def(py::init<>())
		.def(py::init([](const ParticleIO& i, const ParticleIO& j, const std::string& name) {
				 Bond b{};
				 b.uids.uid1 = i.uid;
				 b.uids.uid2 = j.uid;
				 b.function_name = name;
				 return b;
			 }),
			 py::arg("i"),
			 py::arg("j"),
			 py::arg("bond"))
		.def_readwrite("name", &Bond::function_name)
		.def_readwrite("form", &Bond::form)
		.def_readwrite("flag", &Bond::flag)
		.def("add_exclusion", &Bond::add_exclusion)
		.def("__repr__", [](const Bond& b) { return "Bond(name='" + b.function_name + "')"; });
}

void declare_angle(py::module& m) {
	py::class_<Angle>(m, "Angle")
		.def(py::init<>())
		.def(py::init([](const ParticleIO& i,
						 const ParticleIO& j,
						 const ParticleIO& k,
						 const std::string& name) {
				 Angle a{};
				 a.uids.uid1 = i.uid;
				 a.uids.uid2 = j.uid;
				 a.uids.uid3 = k.uid;
				 a.function_name = name;
				 return a;
			 }),
			 py::arg("i"),
			 py::arg("j"),
			 py::arg("k"),
			 py::arg("angle"))
		.def_readwrite("name", &Angle::function_name)
		.def_readwrite("form", &Angle::form)
		.def("__repr__", [](const Angle& a) { return "Angle(name='" + a.function_name + "')"; });
}

void declare_dihedral(py::module& m) {
	py::class_<Dihedral>(m, "Dihedral")
		.def(py::init<>())
		.def(py::init([](const ParticleIO& i,
						 const ParticleIO& j,
						 const ParticleIO& k,
						 const ParticleIO& l,
						 const std::string& name) {
				 Dihedral d{};
				 d.uids.uid1 = i.uid;
				 d.uids.uid2 = j.uid;
				 d.uids.uid3 = k.uid;
				 d.uids.uid4 = l.uid;
				 d.function_name = name;
				 return d;
			 }),
			 py::arg("i"),
			 py::arg("j"),
			 py::arg("k"),
			 py::arg("l"),
			 py::arg("dihedral"))
		.def_readwrite("name", &Dihedral::function_name)
		.def_readwrite("form", &Dihedral::form)
		.def("__repr__",
			 [](const Dihedral& d) { return "Dihedral(name='" + d.function_name + "')"; });
}

void declare_exclude(py::module& m) {
	py::class_<Exclude>(m, "Exclude")
		.def(py::init<>())
		.def(py::init([](const ParticleIO& i, const ParticleIO& j) {
				 Exclude e{};
				 e.uids.uid1 = i.uid;
				 e.uids.uid2 = j.uid;
				 return e;
			 }),
			 py::arg("i"),
			 py::arg("j"))
		.def("__eq__", [](const Exclude& a, const Exclude& b) { return a == b; })
		.def("__ne__", [](const Exclude& a, const Exclude& b) { return a != b; })
		.def("__lt__", [](const Exclude& a, const Exclude& b) { return a < b; })
		.def("__repr__", [](const Exclude& e) {
			return "Exclude(ind1=" + std::to_string(e.ind1) + ", ind2=" + std::to_string(e.ind2) +
				   ")";
		});
}

void declare_restraint(py::module& m) {
	py::class_<Restraint>(m, "Restraint")
		.def(py::init<>())
		.def(py::init([](const ParticleIO& i, Vector3 r0, float k) {
				 Restraint r{};
				 r.uids.uid1 = i.uid;
				 r.r0 = r0;
				 r.k = k;
				 return r;
			 }),
			 py::arg("i"),
			 py::arg("r0"),
			 py::arg("k"))
		.def_readwrite("r0", &Restraint::r0)
		.def_readwrite("k", &Restraint::k)
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
void declare_bonded_interaction(py::module& m) {
	py::class_<BondedInteractions>(m, "BondedInteractions")
		.def(py::init<std::vector<Bond>,
					  std::vector<Angle>,
					  std::vector<Dihedral>,
					  std::vector<Exclude>,
					  std::vector<Restraint>>(),
			 py::arg("bonds") = std::vector<Bond>{},
			 py::arg("angles") = std::vector<Angle>{},
			 py::arg("dihedrals") = std::vector<Dihedral>{},
			 py::arg("exclusions") = std::vector<Exclude>{},
			 py::arg("restraints") = std::vector<Restraint>{})
		.def("add_bond", &BondedInteractions::add_bond, py::arg("bond"))
		.def("add_angle", &BondedInteractions::add_angle, py::arg("angle"))
		.def("add_dihedral", &BondedInteractions::add_dihedral, py::arg("dihedral"))
		.def("add_exclusion", &BondedInteractions::add_exclude, py::arg("exclusion"))
		.def("add_restraint", &BondedInteractions::add_restraint, py::arg("restraint"))
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
void declare_nonbonded_interaction(py::module& m) {
	py::class_<PairNonBonded>(m, "PairNonBonded")
		.def(py::init([](const ParticleType& type_a,
						 const ParticleType& type_b,
						 const std::string& function_name) {
				 return PairNonBonded(type_a.name, type_b.name, function_name);
			 }),
			 py::arg("type_a"),
			 py::arg("type_b"),
			 py::arg("function_name"))
		.def_readonly("type_name_1", &PairNonBonded::type_name_1)
		.def_readonly("type_name_2", &PairNonBonded::type_name_2)
		.def_readwrite("name", &PairNonBonded::function_name)
		.def_readwrite("form", &PairNonBonded::form)
		.def("__repr__", [](const PairNonBonded& p) {
			return "PairNonBonded(type_a='" + p.type_name_1 + "', type_b='" + p.type_name_2 +
				   "', name='" + p.function_name + "')";
		});

	// `type_id` is not exposed: NonBondedInteractions::assign_id() overwrites
	// it with the term's own index, so it is an output, not an input.
	py::class_<LongRangeNonBonded>(m, "LongRangeNonBonded")
		.def(py::init<>())
		.def_readwrite("name", &LongRangeNonBonded::function_name)
		.def_readwrite("form", &LongRangeNonBonded::form)
		.def("__repr__", [](const LongRangeNonBonded& l) {
			return "LongRangeNonBonded(name='" + l.function_name + "')";
		});

	py::class_<NonBondedInteractions>(m, "NonBondedInteractions")
		.def(py::init<std::vector<PairNonBonded>, std::vector<LongRangeNonBonded>>(),
			 py::arg("pair_nonbonded") = std::vector<PairNonBonded>{},
			 py::arg("long_range_nonbonded") = std::vector<LongRangeNonBonded>{})
		.def("add_pair_nonbonded", &NonBondedInteractions::add_pair_nonbonded, py::arg("pair"))
		.def("add_long_range_nonbonded",
			 &NonBondedInteractions::add_long_range_nonbonded,
			 py::arg("long_range"))
		.def("get_num_pair_nonbonded", &NonBondedInteractions::get_num_pair_nonbonded)
		.def("get_num_long_range_nonbonded", &NonBondedInteractions::get_num_long_range_nonbonded)
		.def("__repr__", [](const NonBondedInteractions& nb) {
			return "NonBondedInteractions(pairs=" + std::to_string(nb.get_num_pair_nonbonded()) +
				   ", long_range=" + std::to_string(nb.get_num_long_range_nonbonded()) + ")";
		});
}

// ============================================================================
// POTENTIAL REGISTRATION BINDINGS
// ============================================================================

void declare_register_potential(py::module& m) {
	py::class_<Register_Potential>(m, "RegisterPotential")
		.def(py::init<>())
		.def("register_potential", &Register_Potential::register_potential)
		.def("get_id", &Register_Potential::get_id)
		.def("__repr__", [](const Register_Potential& rp) { return "RegisterPotential()"; });
}

// ============================================================================
// MAIN INITIALIZATION FUNCTION
// ============================================================================

void init_pybonded(py::module_& m) {
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
