// Arbd Group Sites
#pragma once

#include "Backend/Buffer.h"

namespace ARBD {

class GroupSite {
  public:
	GroupSite(const DeviceBuffer<int>& particle_ids);
	~GroupSite();
};

} // namespace ARBD
