#pragma once
#include "Types/BaseGrid.h"
#include "Types/Vector3.h"
#include <string>

namespace ARBD {

class ParticleType {
  private:
	void clear();
	void copy(const ParticleType& src);

  public:
	std::string name;
	struct Properties {
		int id;
		int num; // number of particles of this type
		float mass;
		float charge;
		float radius;
		float eps;
		float diffusion;
		float mu; // for Nose-Hoover Langevin dynamics
		int numPartGridFiles;
		float* meanPmf;
		float* pmf_scale;
	};
	BaseGrid<float>** pmfGrid;
	BaseGrid<float>* diffusionGrid;
	BaseGrid<Vector3>* forceGrid;
	using props = Properties;

	ParticleType(const std::string& name) : name(name){};
	ParticleType(const ParticleType& src) {
		copy(src);
	};
	ParticleType& operator=(const ParticleType& src);
	~ParticleType() {
		clear();
	};
};

} // namespace ARBD
