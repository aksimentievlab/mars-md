
#pragma once

// Any necessary includes would go here
#include <cstdint>
#include <limits>
// Use std::int32_t now

// use new instead
/**
 * @file Constants.h
 * @brief Physical and mathematical constants for MARS simulations. Device constants are defined in
 * Header.h.
 * @details Contains all constant values used throughout the MARS project
 */

namespace MARS {
namespace constants {
constexpr float PI = 3.141592653589793f;
constexpr float TWOPI = 2.0f * PI;
constexpr float HALFPI = 0.5f * PI;

// Physical constants
constexpr float COULOMB = 332.0636f;
constexpr float BOLTZMANN = 0.001987191f;
constexpr float PRESSUREFACTOR = 6.95E4f;

// 1 kcal_IT = 4.1868 Joules.
// Scaling for ns timestep and amu mass units.
constexpr float FORCE_CONVERSION_FACTOR = 4.18679994e4;

// Square root of the Joule-Calorie conversion (sqrt(4.1868))
// Required for scaling sqrt(kT/m) noise terms.
constexpr float SQRT_CAL_TO_JOULE = 2.046167337;

// Simulation constants
constexpr float PDBVELFACTOR = 20.45482706f;
constexpr float PDBVELINVFACTOR = 1.0f / PDBVELFACTOR;
constexpr float PNPERKCALMOL = 69.479f;
constexpr float SMALLRAD = 0.0005f;
constexpr float SMALLRAD2 = SMALLRAD * SMALLRAD;

// Legacy RigidBody unit conversions (v1 RigidBody.cu / RigidBodyType.cu).
constexpr float impulse_to_momentum = 4.1867999435271e4f;

constexpr float langevin_damping_unit = 2.3900574e-9f;
// Additional literal factor on the drag term only, from legacy
// RigidBody::addLangevin ("... * 10000"). Applied on top of
// langevin_damping_unit, so the effective drag scale is 2.3900574e-5.
constexpr float langevin_damp_scale = 10000.0f;
constexpr float velocity_scale = 1e4f;
constexpr double ANGSTROM_TO_MICRON = 1.0e-4;
constexpr double MICRON_TO_ANGSTROM = 1.0e4;
constexpr const char* kAttachedSegnameMarker =
	"ATT"; ///< marker in pdb file that will mark the attachement site.

constexpr const char* kCosmeticTypeName =
	"COS"; ///< Type name for template atoms that carry no physics.
} // namespace constants
// namespace constants
/**
 * @brief Unit conversions between SCUFF-EM's reporting units and MARS's.
 *
 * SCUFF meshes are in microns, reports force in nanonewtons and torque in
 * nanonewton*microns (see scuff-em applications/scuff-scatter/OutputModules.cc).
 * MARS uses Angstroms, kcal/mol/Angstrom and kcal/mol.
 */
namespace scuff_units {
/// constants::PNPERKCALMOL is pN per (kcal/mol/Angstrom), so 1 nN = 1000/that.
constexpr double NN_TO_KCAL_PER_MOL_ANGSTROM = 1000.0 / constants::PNPERKCALMOL;
constexpr double NN_MICRON_TO_KCAL_PER_MOL =
	NN_TO_KCAL_PER_MOL_ANGSTROM * constants::MICRON_TO_ANGSTROM;
constexpr double lambda_nm_to_omega_rad_per_sec = 2.0 * constants::PI * 1.0e3;
} // namespace scuff_units

} // namespace MARS
