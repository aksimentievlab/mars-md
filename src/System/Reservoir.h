#pragma once
#include "Header.h"
#include "Types/Types.h"
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

class Reservoir {
  public:
	Reservoir();
	~Reservoir();

	static int countReservoirs(const char* reservoirFile);

	HOST DEVICE Vector3 getOrigin(int i) const;
	HOST DEVICE Vector3 getDestination(int i) const;
	HOST DEVICE Vector3 getDifference(int i) const;

	HOST DEVICE float getMeanNumber(int i) const;
	HOST DEVICE int length() const;

	HOST DEVICE bool inside(int i, Vector3 r) const;

  private:
	int reservoir_id;
	Vector3* start;
	Vector3* end;
	float* num;

	void readReservoirs(const char* reservoirFile);
	void validateRegions();
};
} // namespace ARBD
