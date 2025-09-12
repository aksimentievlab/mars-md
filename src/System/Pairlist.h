#pragma once

#include "Backend/Resource.h"
#include "SimSystem.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

class CheckPairlist {
  public:
	CheckPairlist(SimSystem& sys, const ResourceCollection& resources);
	void check_pairlist(SimSystem& sys, const ResourceCollection& resources);

  private:
	SimSystem& sys_;
	ResourceCollection resources_;
};

} // namespace ARBD
