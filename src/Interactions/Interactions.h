#pragma once
#include "Header.h"
#include "Types/Types.h"

namespace ARBD {

struct ForceEnergy {
	float force_magnitude;
	float energy;
};
// ============================================================================
// POTENTIAL REGISTRATION
// ============================================================================

/** Future pybind class for new analytical pairwise potential registration */
class Register_Potential {
  public:
	int register_potential(const std::string& name) {
		if (m_name_to_id.find(name) == m_name_to_id.end()) {
			int new_id = m_name_to_id.size();
			m_name_to_id[name] = new_id;
		}
		return m_name_to_id[name];
	}

	int get_id(const std::string& name) const {
		return m_name_to_id.at(name);
	}

  private:
	std::unordered_map<std::string, int> m_name_to_id;
	int bodies_{2};				// 2 for pairwise, 3 for triplet, 4 for quadruplet
	std::vector<float> params_; // Parameters for the potential
};

} // namespace ARBD
