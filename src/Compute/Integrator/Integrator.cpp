#include "Integrator.h"

bool operator<(const Integrator::Conf x, const Integrator::Conf y) { return (int(x) < int(y)); };
std::map<Integrator::Conf, Integrator*> Integrator::_integrators;
	
Integrator* Integrator::GetIntegrator(Conf& conf) {
	// Checks _integrators for a matching configuration, returns one if found, otherwise creates
	if (conf.backend == Conf::Default) {
#ifdef USE_CUDA
	    conf.backend = Conf::CUDA;
#else
	    conf.backend = Conf::CPU;
#endif
	}

	// Insert configuration into map, if it exists 
	auto emplace_result = Integrator::_integrators.emplace(conf, nullptr);
	auto& it = emplace_result.first;
	bool& inserted = emplace_result.second;
	if (inserted) {
	    // Conf not found, so create a new one 
	    Integrator* tmp;

	    switch (conf.object_type) {
	    case Conf::Particle:
		switch (conf.algorithm) {
		case Conf::BD:
		    switch (conf.backend) {
		    case Conf::CUDA:
#ifdef USE_CUDA
			tmp = new BDIntegrateCUDA();
#else
			std::cerr << "WARNING: Integrator::GetIntegrator: "
				  << "CUDA disabled, creating CPU integrator instead" << std::endl;
			tmp = new BDIntegrate();
#endif
			break;
		    case Conf::CPU:
			tmp = new BDIntegrate();
			break;
		    default:
			std::cerr << "Error: Integrator::GetIntegrator: "
				  << "Unrecognized backend; exiting" << std::endl;
			assert(false);
		    }
		    break;
		case Conf::MD:
		    assert(false);
		    break;
		default:
		    std::cerr << "Error: Integrator::GetIntegrator: "
			      << "Unrecognized algorithm type; exiting" << std::endl;
		    assert(false);
		}
		break;
	    case Conf::RigidBody:
		assert(false);
		break;
	    default:
		std::cerr << "Error: Integrator::GetIntegrator: "
			  << "Unrecognized object type; exiting" << std::endl;
		assert(false);
	    }
	    it->second = tmp;
	}
	return it->second;
}
