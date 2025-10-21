#include "ParticleProperties.h"

namespace ARBD {

void ParticleType::clear() {
	if (meanPmf != nullptr) {
		delete[] meanPmf;
		meanPmf = nullptr;
	}
	if (pmf_scale != nullptr) {
		delete[] pmf_scale;
		pmf_scale = nullptr;
	}
}

void ParticleType::copy(const ParticleType& src) {
	name = src.name;
	id = src.id;
	num = src.num;
	mass = src.mass;
	charge = src.charge;
	radius = src.radius;
	eps = src.eps;
	diffusion = src.diffusion;
	transDamping = src.transDamping;
	mu = src.mu;
	numPartGridFiles = src.numPartGridFiles;

	// Deep copy arrays
	if (src.meanPmf != nullptr) {
		meanPmf = new float[numPartGridFiles];
		for (int i = 0; i < numPartGridFiles; ++i) {
			meanPmf[i] = src.meanPmf[i];
		}
	} else {
		meanPmf = nullptr;
	}

	if (src.pmf_scale != nullptr) {
		pmf_scale = new float[numPartGridFiles];
		for (int i = 0; i < numPartGridFiles; ++i) {
			pmf_scale[i] = src.pmf_scale[i];
		}
	} else {
		pmf_scale = nullptr;
	}
}

ParticleType::ParticleType(const ParticleType& src) {
	copy(src);
}

ParticleType& ParticleType::operator=(const ParticleType& src) {
	if (this != &src) {
		clear();
		copy(src);
	}
	return *this;
}

ParticleType::~ParticleType() {
	clear();
}

} // namespace ARBD
