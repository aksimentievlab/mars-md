#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
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
		.value("Tabulated", InteractionForm::Tabulated)
		.value("Analytical", InteractionForm::Analytical);

	py::class_<Bond>(m, "Bond")
		.def(py::init<>())
		.def(py::init<int, int, std::vector<Exclude>&>(),
			 py::arg("ind1"),
			 py::arg("ind2"),
			 py::arg("exclusions"))
		.def_readwrite("ind1", &Bond::ind1)
		.def_readwrite("ind2", &Bond::ind2)
		.def_readwrite("name", &Bond::function_name)
		.def_readwrite("form", &Bond::form)
		.def_readwrite("function_index", &Bond::function_index)
		.def_readwrite("flag", &Bond::flag)
		.def("add_exclusion", &Bond::add_exclusion)
		.def("__repr__", [](const Bond& b) {
			return "Bond(ind1=" + std::to_string(b.ind1) + ", ind2=" + std::to_string(b.ind2) +
				   ", name='" + b.function_name + "')";
		});
}

void declare_angle(py::module& m) {
	py::class_<Angle>(m, "Angle")
		.def(py::init<>())
		.def_readwrite("ind1", &Angle::ind1)
		.def_readwrite("ind2", &Angle::ind2)
		.def_readwrite("ind3", &Angle::ind3)
		.def_readwrite("name", &Angle::function_name)
		.def_readwrite("form", &Angle::form)
		.def_readwrite("function_index", &Angle::function_index)
		.def("__repr__", [](const Angle& a) {
			return "Angle(ind1=" + std::to_string(a.ind1) + ", ind2=" + std::to_string(a.ind2) +
				   ", ind3=" + std::to_string(a.ind3) + ", name='" + a.function_name + "')";
		});
}

void declare_dihedral(py::module& m) {
	py::class_<Dihedral>(m, "Dihedral")
		.def(py::init<>())
		.def_readwrite("ind1", &Dihedral::ind1)
		.def_readwrite("ind2", &Dihedral::ind2)
		.def_readwrite("ind3", &Dihedral::ind3)
		.def_readwrite("ind4", &Dihedral::ind4)
		.def_readwrite("name", &Dihedral::function_name)
		.def_readwrite("form", &Dihedral::form)
		.def_readwrite("function_index", &Dihedral::function_index)
		.def("__repr__", [](const Dihedral& d) {
			return "Dihedral(ind1=" + std::to_string(d.ind1) + ", ind2=" + std::to_string(d.ind2) +
				   ", ind3=" + std::to_string(d.ind3) + ", ind4=" + std::to_string(d.ind4) +
				   ", name='" + d.function_name + "')";
		});
}

void declare_exclude(py::module& m) {
	py::class_<Exclude>(m, "Exclude")
		.def(py::init<>())
		.def(py::init<int, int>(), py::arg("ind1"), py::arg("ind2"))
		.def_readwrite("ind1", &Exclude::ind1)
		.def_readwrite("ind2", &Exclude::ind2)
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
		.def(py::init<int, Vector3, float>(), py::arg("id"), py::arg("r0"), py::arg("k"))
		.def_readwrite("ind", &Restraint::ind)
		.def_readwrite("r0", &Restraint::r0)
		.def_readwrite("k", &Restraint::k)
		.def("__repr__", [](const Restraint& r) {
			return "Restraint(ind=" + std::to_string(r.ind) + ", r0=" + r.r0.to_string() +
				   ", k=" + std::to_string(r.k) + ")";
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
		.def(py::init<std::vector<Bond>, std::vector<Angle>, std::vector<Dihedral>>(),
			 py::arg("bonds"),
			 py::arg("angles"),
			 py::arg("dihedrals"))
		.def("add_bond", &BondedInteractions::add_bond)
		.def("add_angle", &BondedInteractions::add_angle)
		.def("add_dihedral", &BondedInteractions::add_dihedral)
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
	declare_register_potential(m);
}
