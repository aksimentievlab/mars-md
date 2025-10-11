#pragma once
#include "BaseGrid.h"
#include "NanoGridHandle.h"

namespace ARBD {
enum class GridType { Potential, Force, Diffusion, PMF, Histogram };
enum class GridFormat { Dense, Sparse };
enum class InterpolationOrder : int { Linear = 1, Cubic = 3 };

struct GridKey {
	std::string name;
	GridFormat format;
	int functionsid;
	GridKey(const std::string& name, const GridFormat& format) : name(name), format(format) {}
	GridKey(const std::string& name) : name(name) {
		if (name.find(".dx") != std::string::npos) {
			format = GridFormat::Dense;
		} else {
			format = GridFormat::Sparse;
		}
	}
	bool operator==(const GridKey& o) const {
		return name == o.name && format == o.format;
	}
};

} // namespace ARBD
