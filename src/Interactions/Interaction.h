/**
 * @file LocalInteraction.h
 * @brief Defines the LocalInteraction class and its related structures
 */

#pragma once

#include <cassert>
#include <iostream>
#include <map>
#include "Objects/Patch/PatchOp.h"

/**
 * @class LocalInteraction
 * @brief Base class for local interaction
 */
class LocalInteraction : public PatchOp {
public:
    /**
     * @brief Computes interaction for a given patch
     * @param patch Pointer to the patch for which interaction is to be computed
     */
    virtual void compute(Patch* patch) = 0;

    /**
     * @brief Returns the number of patches
     * @return The number of patches
     */
    int num_patches() const { return 1; };

    /**
     * @brief Struct for defining interaction configuration
     */
    struct Conf {
	    enum Object    {Particle, RigidBody};
	    enum DoF {Bond, Angle, Dihedral, Bonded, NeighborhoodPair};
	    enum Form {Harmonic, Tabulated, LennardJones, Coulomb, LJ};
	    enum Backend   { Default, CUDA, CPU };
	
	    Object object_type; ///< The object type
	    DoF dof; ///< The degree of freedom
	    Form form; ///< The form
	    Backend backend; ///< The backend for computing the interaction

	    /**
	     * @brief int Conversion operator for Conf for easy comparison
	     * @return The integer representation of Conf
	     */
	    explicit operator int() const {return object_type*64 + dof*16 + form*4 + backend;};
    };

    /**
     * @brief Factory method for obtaining a LocalInteraction object
     * @param conf The configuration for the LocalInteraction object to be obtained
     * @return A pointer to the LocalInteraction object
     */
    static LocalInteraction* GetInteraction(Conf& conf);

private:
    size_t num_interactions; /**< Stores the number of interactions of this type in the system. */
    
protected:
    /**
     * A map that maps interaction configurations to interaction objects.
     * This map is shared across all instances of LocalInteraction and is used to
     * lazily initialize and retrieve interaction objects based on their configuration.
     */
    static std::map<Conf, LocalInteraction*> _interactions;

};

// class LocalNonbondedInteraction : public BasePatchOp {
// public:
//     virtual void compute(Patch* patch) = 0;
//     int num_patches() const { return 1; };

//     // Following relates to lazy initialized factory method
//     struct Conf {
// 	enum Object    {Particle, RigidBody };
// 	enum Electrostatics {None, Coulomb, DebyeHuckel}
// 	enum Form      {Tabulated, LennardJones, Coulomb, LJ, DebyeHueckel};
// 	enum Backend   { Default, CUDA, CPU };
	
// 	Object object_type;
	
// 	std::string tabulated_file = "";
// 	Algorithm algorithm;
// 	Backend backend;

// 	explicit operator int() const {return object_type*16 + algorithm*4 + backend;};
//     };

//     static Integrator* GetIntegrator(Conf& conf);

// private:
//     size_t num_interactions;
    
// protected:
//     static std::map<Conf, Integrator*> _integrators;

// };

// class LocalBonded : public LocalInteraction {

// private:
//     size_t bondlist;		// Encode bond, angle, dihedral
    
// };

// class LocalBonds : public LocalInteraction {

// private:
//     // static const char* type = "Bond";
//     size_t bondlist;
    
// };


#include "kernels.h"
#include "GPUInteraction.h"
#include "CPUInteraction.h"
