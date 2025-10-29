#pragma once
#include "../Interactions.h"
#include "Backend/Resource.h"
#include "Constants.h"
#include "Header.h"
#include "IO/TabulatedReader.h"
#include "Objects/ParticleProperties.h"
#include "Types/Types.h"

namespace ARBD {

struct TabulatedNB {
	int p_type_a, p_type_b;
	DEVICE_PTR(float) pot; // Device pointer to potential table
	float step_inv;		   // 1/step_size for interpolation
	unsigned int size;	   // Table size
	float start;		   // Starting value (e.g., 0 for bonds, -PI for dihedrals)
	bool is_periodic;	   // Whether to wrap around (dihedrals)
};

} // namespace ARBD
