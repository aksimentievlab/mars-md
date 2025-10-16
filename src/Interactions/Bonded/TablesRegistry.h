#pragma once
#include "Header.h"
#include "TabulatedPotential.h"
#include <unordered_map>

namespace ARBD {

// Registry to load and cache tabulated potentials by name
class TablesRegistry {
  public:
	static TablesRegistry& instance();

	int getOrLoadAngle(const std::string& name, const std::string& path);
	int getOrLoadDihedral(const std::string& name, const std::string& path);
	int getOrLoadBond(const std::string& name, const std::string& path);

	const BondedPotential* getAngle(int idx) const;
	const BondedPotential* getDihedral(int idx) const;
	const BondedPotential* getBond(int idx) const;

  private:
	TablesRegistry() = default;

	std::vector<BondedPotential> angles_;
	std::vector<BondedPotential> dihedrals_;
	std::vector<BondedPotential> bonds_;

	std::unordered_map<std::string, int> angleNameToIdx_;
	std::unordered_map<std::string, int> dihedralNameToIdx_;
	std::unordered_map<std::string, int> bondNameToIdx_;
};

} // namespace ARBD
