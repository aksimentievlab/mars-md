#include "Interactions/BondedInteraction.h"
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

	py::class_<Bond>(m, "Bond")
		.def(py::init<>())
		.def_readwrite("ind1", &Bond::ind1)
		.def_readwrite("ind2", &Bond::ind2)
		.def_readwrite("name", &Bond::name)
		.def_readwrite("form", &Bond::form)
		.def_readwrite("functionIndex", &Bond::functionIndex)
		.def_readwrite("flag", &Bond::flag)
		.def("add_exclusion", &Bond::add_exclusion)
		.def("__repr__", [](const Bond& b) {
			return "Bond(ind1=" + std::to_string(b.ind1) + ", ind2=" + std::to_string(b.ind2) +
				   ", name='" + b.name + "')";
		});
}

void declare_angle(py::module& m) {
	py::class_<Angle>(m, "Angle")
		.def(py::init<>())
		.def_readwrite("ind1", &Angle::ind1)
		.def_readwrite("ind2", &Angle::ind2)
		.def_readwrite("ind3", &Angle::ind3)
		.def_readwrite("name", &Angle::name)
		.def_readwrite("form", &Angle::form)
		.def_readwrite("functionIndex", &Angle::functionIndex)
		.def("__repr__", [](const Angle& a) {
			return "Angle(ind1=" + std::to_string(a.ind1) + ", ind2=" + std::to_string(a.ind2) +
				   ", ind3=" + std::to_string(a.ind3) + ", name='" + a.name + "')";
		});
}

void declare_dihedral(py::module& m) {
	py::class_<Dihedral>(m, "Dihedral")
		.def(py::init<>())
		.def_readwrite("ind1", &Dihedral::ind1)
		.def_readwrite("ind2", &Dihedral::ind2)
		.def_readwrite("ind3", &Dihedral::ind3)
		.def_readwrite("ind4", &Dihedral::ind4)
		.def_readwrite("name", &Dihedral::name)
		.def_readwrite("form", &Dihedral::form)
		.def_readwrite("functionIndex", &Dihedral::functionIndex)
		.def("__repr__", [](const Dihedral& d) {
			return "Dihedral(ind1=" + std::to_string(d.ind1) + ", ind2=" + std::to_string(d.ind2) +
				   ", ind3=" + std::to_string(d.ind3) + ", ind4=" + std::to_string(d.ind4) +
				   ", name='" + d.name + "')";
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
	py::class_<BondedInteraction>(m, "BondedInteraction")
		.def(py::init<>())
		.def("addBond", &BondedInteraction::addBond)
		.def("addAngle", &BondedInteraction::addAngle)
		.def("addDihedral", &BondedInteraction::addDihedral)
		.def("prepareDeviceData", &BondedInteraction::prepareDeviceData)
		.def("cleanupDeviceData", &BondedInteraction::cleanupDeviceData)
		.def("getNumBonds", &BondedInteraction::getNumBonds)
		.def("getNumAngles", &BondedInteraction::getNumAngles)
		.def("getNumDihedrals", &BondedInteraction::getNumDihedrals)
		.def("__repr__", [](const BondedInteraction& bi) {
			return "BondedInteraction(bonds=" + std::to_string(bi.getNumBonds()) +
				   ", angles=" + std::to_string(bi.getNumAngles()) +
				   ", dihedrals=" + std::to_string(bi.getNumDihedrals()) + ")";
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
