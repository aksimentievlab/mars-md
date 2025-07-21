#include "Interaction.h"

bool operator<(const LocalInteraction::Conf x, const LocalInteraction::Conf y) { return (int(x) < int(y)); };
std::map<LocalInteraction::Conf, LocalInteraction*> LocalInteraction::_interactions;
	
LocalInteraction* LocalInteraction::GetInteraction(Conf& conf) {
	// Checks _interactions for a matching configuration, returns one if found, otherwise creates
	if (conf.backend == Conf::Default) {
#ifdef USE_CUDA
	    conf.backend = Conf::CUDA;
#else
	    conf.backend = Conf::CPU;
#endif
	}

	// Insert configuration into map, if it exists 
	auto emplace_result = LocalInteraction::_interactions.emplace(conf, nullptr);
	auto& it = emplace_result.first;
	bool& inserted = emplace_result.second;
	if (inserted) {
	    // Conf not found, so create a new one 
 	    LocalInteraction* tmp;

	    switch (conf.object_type) {
	    case Conf::Particle:
		switch (conf.dof) {
		case Conf::Bond:
		    switch (conf.form) {
		    case Conf::Harmonic:
			switch (conf.backend) {
			case Conf::CUDA:
#ifdef USE_CUDA
			    tmp = new LocalBondedCUDA();
#else
			    std::cerr << "WARNING: LocalInteraction::GetInteraction: "
				      << "CUDA disabled, creating CPU interaction instead" << std::endl;
			    tmp = new LocalBonded();
#endif
			    break;
			case Conf::CPU:
			    tmp = new LocalBonded();
			    break;
			default:
			    std::cerr << "Error: LocalInteraction::GetInteraction: "
				      << "Unrecognized backend; exiting" << std::endl;
			}
		    default:
			std::cerr << "Error: LocalInteraction::GetInteraction: "
				  << "Unrecognized form; exiting" << std::endl;
			assert(false);
		    }
		    break;
		default:
		    std::cerr << "Error: Interaction::GetInteraction: "
			      << "Unrecognized algorithm type; exiting" << std::endl;
		    assert(false);
		}
		break;
	    case Conf::RigidBody:
		assert(false);
		break;
	    default:
		std::cerr << "Error: Interaction::GetInteraction: "
			  << "Unrecognized object type; exiting" << std::endl;
		assert(false);
	    }
	    it->second = tmp;
	}
	return it->second;
}
