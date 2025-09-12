/*********************************************************************
 * @file  Integrator.h
 *
 * @brief Declaration of Integrator class with factory-like method
 * GetIntegrator()
 *
 * Defines hashable Integrator::Conf struct that allows operator
 * resuse. Common CPU/GPU kernels implemented in Integrator/kernels.h
 * with various BD/MD integrators implemented on different backends in
 * Integrator/CUDA.h and Integrator/CPU.h
 *********************************************************************/
#pragma once

#include <cassert>
#include <iostream>
#include <map>

namespace ARBD {
class Integrator {
  public:
	virtual void compute(Patch* patch) = 0;

	// Following relates to lazy initialized factory method
	struct Conf {
		enum Object { Particle, RigidBody };
		enum Algorithm { BD, MD };
		enum Backend { CUDA, SYCL, METAL, CPU };

		Object object_type;
		Algorithm algorithm;
		Backend backend;

		explicit operator int() const {
			return object_type * 16 + algorithm * 4 + backend;
		};
	};

	static Integrator* GetIntegrator(Conf& conf);

  protected:
	static std::map<Conf, Integrator*> _integrators;
};
} // namespace ARBD
