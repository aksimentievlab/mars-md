#pragma once
#include "Header.h"
#include "IO/FileHandle.h"
#include "Types/Types.h"
#define STRLEN 512
/**
 * @brief The Reservoir class manages particle reservoirs for GCMC - defined spatial regions where
 * particles can be added or removed to maintain a target particle density. It:
 * 1. Reads reservoir configuration files that define:
 *   - Spatial boundaries (x0,y0,z0 to x1,y1,z1)
 *   - Target particle number/density for each reservoir region
 * 2. Manages reservoir regions by:
 *   - Storing origin (r0) and destination (r1) vectors defining box boundaries
 *   - Tracking target particle numbers (num) for each region
 *   - Validating that regions are properly defined (ensuring min/max coordinates)
 * 3. Provides spatial queries to:
 *   - Check if a particle position is inside a reservoir region (inside())
 *   - Get reservoir dimensions and properties
 *   - Calculate volume differences between boundaries
 * 4. Enablinggrand canonical ensemble simulations:
 *   - Particles can be inserted when the current count is below target
 *   - Particles can be deleted when the current count exceeds target
 *   - This maintains constant chemical potential rather than constant particle number
 */

namespace ARBD {
struct Reservoir {
	HOST DEVICE Vector3 start;
	HOST DEVICE Vector3 end;
	HOST DEVICE int target_num;	 // target number of particles in the reservoir
	HOST DEVICE int current_num; // current number of particles in the reservoir
	bool isinside(Vector3 r) const {
		return r.x >= start.x && r.x <= end.x && r.y >= start.y && r.y <= end.y && r.z >= start.z &&
			   r.z <= end.z;
	}
};

class ReservoirManager { // ON host only
  public:
	ReservoirManager() = default;
	ReservoirManager(const char* reservoirFile);
	~ReservoirManager();

	void addReservoir(const Reservoir& reservoir) {
		reservoirs.push_back(reservoir);
	};

	HOST int getTargetMeanNum() const;
	HOST DEVICE int getCurrentMeanNum() const;

	HOST DEVICE bool inside(int i, Vector3 r) {
		return reservoirs[i].isinside(r);
	};

  private:
	std::vector<Reservoir> reservoirs;
	void validateRegions();
};
} // namespace ARBD
