#include "TablesRegistry.h"

namespace ARBD {

TablesRegistry& TablesRegistry::instance() {
	static TablesRegistry inst;
	return inst;
}

int TablesRegistry::getOrLoadAngle(const std::string& name, const std::string& path) {
	auto it = angleNameToIdx_.find(name);
	if (it != angleNameToIdx_.end())
		return it->second;
	angles_.emplace_back(path.c_str(), BondedPotentialType::ANGLE);
	int idx = static_cast<int>(angles_.size() - 1);
	angleNameToIdx_[name] = idx;
	return idx;
}

int TablesRegistry::getOrLoadDihedral(const std::string& name, const std::string& path) {
	auto it = dihedralNameToIdx_.find(name);
	if (it != dihedralNameToIdx_.end())
		return it->second;
	dihedrals_.emplace_back(path.c_str(), BondedPotentialType::DIHEDRAL);
	int idx = static_cast<int>(dihedrals_.size() - 1);
	dihedralNameToIdx_[name] = idx;
	return idx;
}

int TablesRegistry::getOrLoadBond(const std::string& name, const std::string& path) {
	auto it = bondNameToIdx_.find(name);
	if (it != bondNameToIdx_.end())
		return it->second;
	// Use BondedPotential for bonds as well (distance-based)
	// Store in bonds_ as TabulatedPotential is radial, but we unify on BondedPotential
	// For now, keep bonds_ empty and reuse angles_ vector with BOND type if needed in caller.
	// To provide a stable index space, push a placeholder TabulatedPotential and return index.
	// However, prefer using BondedPotential for bonds via getAngle path with BOND type.
	// Implement bonds via separate BondedPotential vector for clarity later if needed.
	// Here, emulate bond storage by using dihedrals_ container is not appropriate; keep a stub.
	// Minimal implementation: treat bonds as angles_ with BOND type.
	angles_.emplace_back(path.c_str(), BondedPotentialType::BOND);
	int idx = static_cast<int>(angles_.size() - 1);
	bondNameToIdx_[name] = idx;
	return idx;
}

const BondedPotential* TablesRegistry::getAngle(int idx) const {
	return idx >= 0 && idx < static_cast<int>(angles_.size()) ? &angles_[idx] : nullptr;
}

const BondedPotential* TablesRegistry::getDihedral(int idx) const {
	return idx >= 0 && idx < static_cast<int>(dihedrals_.size()) ? &dihedrals_[idx] : nullptr;
}

const BondedPotential* TablesRegistry::getBond(int idx) const {
	// Not implemented as TabulatedPotential; callers can use BondedPotential with BOND type.
	return nullptr;
}

} // namespace ARBD
