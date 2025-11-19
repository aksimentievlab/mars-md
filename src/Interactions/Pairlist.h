#pragma once

#include "Backend/Resource.h"
#include "SimSystem.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

class CheckPairlist {
  public:
	CheckPairlist(SimSystem& sys);
	void check_pairlist(SimSystem& sys);

  private:
	SimSystem& sys_;
};

} // namespace ARBD
