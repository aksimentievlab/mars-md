#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Interactions/BondedInteraction.h"
#include "Types/Types.h"
#include <vector>

namespace ARBD {

/**
 * @brief Steered Molecular Dynamics (SMD) bond interaction
 *
 * This class implements SMD bonds where bond parameters can change
 * over time according to: r_0 = r_0_0 + v * timestep
 */
class SMDBond {
  public:
	/**
	 * @brief Constructor for SMD bond
	 * @param particle1 First particle index
	 * @param particle2 Second particle index
	 * @param r0_initial Initial bond length
	 * @param spring_constant Spring constant
	 * @param velocity Rate of change of bond length
	 */
	SMDBond(int particle1, int particle2, float r0_initial, float spring_constant, float velocity)
		: ids_{particle1, particle2}, params_{r0_initial, spring_constant, velocity, 0.0f} {
		if (particle1 < 0 || particle2 < 0) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid particle indices for SMD bond: %d, %d",
							particle1,
							particle2);
		}

		if (spring_constant <= 0.0f) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid spring constant for SMD bond: %f",
							spring_constant);
		}

		LOGDEBUG("SMDBond: Created bond between particles %d-%d, r0=%f, k=%f, v=%f",
				 particle1,
				 particle2,
				 r0_initial,
				 spring_constant,
				 velocity);
	}

	/**
	 * @brief Get current bond length at given timestep
	 * @param timestep Current timestep
	 * @return Current bond length
	 */
	float getCurrentBondLength(size_t timestep) const {
		return params_.x + params_.z * timestep; // r0_0 + v * timestep
	}

	/**
	 * @brief Get spring constant
	 * @return Spring constant
	 */
	float getSpringConstant() const {
		return params_.y;
	}

	/**
	 * @brief Get velocity parameter
	 * @return Velocity parameter
	 */
	float getVelocity() const {
		return params_.z;
	}

	/**
	 * @brief Get particle indices
	 * @return Pair of particle indices
	 */
	const int2& getParticles() const {
		return ids_;
	}

	/**
	 * @brief Get all parameters
	 * @return Parameters as float4
	 */
	const float4& getParams() const {
		return params_;
	}

  private:
	int2 ids_;		// Particle indices
	float4 params_; // r0_initial, k, v, unused
};

class SMDAngle {
  public:
	/**
	 * @brief Constructor for SMD dihedral
	 * @param particle1 First particle index
	 * @param particle2 Second particle index
	 * @param particle3 Third particle index
	 * @param particle4 Fourth particle index
	 * @param theta0_initial Initial dihedral angle (in radians)
	 * @param spring_constant Spring constant
	 * @param velocity Rate of change of dihedral angle
	 */
	SMDAngle(int particle1,
			 int particle2,
			 int particle3,
			 float angle0_initial,
			 float spring_constant,
			 float velocity)
		: ids_{particle1, particle2, particle3},
		  params_{angle0_initial, spring_constant, velocity, 0.0f} {

		if (particle1 < 0 || particle2 < 0 || particle3 < 0) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid particle indices for SMD angle: %d, %d, %d",
							particle1,
							particle2,
							particle3);
		}

		if (spring_constant <= 0.0f) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid spring constant for SMD dihedral: %f",
							spring_constant);
		}

		LOGDEBUG(
			"SMDDihedral: Created dihedral between particles %d-%d-%d-%d, theta0=%f, k=%f, v=%f",
			particle1,
			particle2,
			particle3,
			angle0_initial,
			spring_constant,
			velocity);
	}

	/**
	 * @brief Get current dihedral angle at given timestep
	 * @param timestep Current timestep
	 * @return Current dihedral angle
	 */
	float getCurrentAngle(size_t timestep) const {
		return params_.x + params_.z * timestep;
	}

	/**
	 * @brief Get spring constant
	 * @return Spring constant
	 */
	float getSpringConstant() const {
		return params_.y;
	}

	/**
	 * @brief Get velocity parameter
	 * @return Velocity parameter
	 */
	float getVelocity() const {
		return params_.z;
	}

	/**
	 * @brief Get particle indices
	 * @return Four particle indices
	 */
	const int4& getParticles() const {
		return ids_;
	}

	/**
	 * @brief Get all parameters
	 * @return Parameters as float4
	 */
	const float4& getParams() const {
		return params_;
	}

  private:
	int4 ids_;		// Particle indices
	float4 params_; // angle0_initial, k, v, unused
};

/**
 * @brief SMD dihedral interaction
 *
 * This class implements SMD dihedrals where dihedral parameters can change
 * over time according to: theta_0 = theta_0_0 + v * timestep
 */
class SMDDihedral {
  public:
	/**
	 * @brief Constructor for SMD dihedral
	 * @param particle1 First particle index
	 * @param particle2 Second particle index
	 * @param particle3 Third particle index
	 * @param particle4 Fourth particle index
	 * @param theta0_initial Initial dihedral angle (in radians)
	 * @param spring_constant Spring constant
	 * @param velocity Rate of change of dihedral angle
	 */
	SMDDihedral(int particle1,
				int particle2,
				int particle3,
				int particle4,
				float theta0_initial,
				float spring_constant,
				float velocity)
		: ids_{particle1, particle2, particle3, particle4},
		  params_{theta0_initial, spring_constant, velocity, 0.0f} {

		if (particle1 < 0 || particle2 < 0 || particle3 < 0 || particle4 < 0) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid particle indices for SMD dihedral: %d, %d, %d, %d",
							particle1,
							particle2,
							particle3,
							particle4);
		}

		if (spring_constant <= 0.0f) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid spring constant for SMD dihedral: %f",
							spring_constant);
		}

		LOGDEBUG(
			"SMDDihedral: Created dihedral between particles %d-%d-%d-%d, theta0=%f, k=%f, v=%f",
			particle1,
			particle2,
			particle3,
			particle4,
			theta0_initial,
			spring_constant,
			velocity);
	}

	/**
	 * @brief Get current dihedral angle at given timestep
	 * @param timestep Current timestep
	 * @return Current dihedral angle
	 */
	float getCurrentDihedralAngle(size_t timestep) const {
		return params_.x + params_.z * timestep; // theta0_0 + v * timestep
	}

	/**
	 * @brief Get spring constant
	 * @return Spring constant
	 */
	float getSpringConstant() const {
		return params_.y;
	}

	/**
	 * @brief Get velocity parameter
	 * @return Velocity parameter
	 */
	float getVelocity() const {
		return params_.z;
	}

	/**
	 * @brief Get particle indices
	 * @return Four particle indices
	 */
	const int4& getParticles() const {
		return ids_;
	}

	/**
	 * @brief Get all parameters
	 * @return Parameters as float4
	 */
	const float4& getParams() const {
		return params_;
	}

  private:
	int4 ids_;		// Particle indices
	float4 params_; // theta0_initial, k, v, unused
};

/**
 * @brief Manager for SMD interactions
 *
 * This class manages all SMD bonds and dihedrals in the system
 */
class SMDManager {
  public:
	/**
	 * @brief Constructor
	 * @param resources Backend resources for GPU operations
	 */
	SMDManager(std::shared_ptr<Resource> resources) : resources_(resources), enabled_(false) {
		LOGINFO("SMDManager: Initialized");
	}

	/**
	 * @brief Destructor
	 */
	~SMDManager() = default;

	// Non-copyable but movable
	SMDManager(const SMDManager&) = delete;
	SMDManager& operator=(const SMDManager&) = delete;

	SMDManager(SMDManager&&) = default;
	SMDManager& operator=(SMDManager&&) = default;

	/**
	 * @brief Add an SMD bond
	 * @param bond SMD bond to add
	 */
	void addBond(SMDBond&& bond) {
		smd_bonds_.push_back(bond);
		LOGINFO("SMDManager: Added SMD bond between particles %d-%d",
				bond.getParticles().x,
				bond.getParticles().y);
	}

	/**
	 * @brief Add an SMD dihedral
	 * @param dihedral SMD dihedral to add
	 */
	void addDihedral(const SMDDihedral& dihedral) {
		smd_dihedrals_.push_back(dihedral);
		LOGINFO("SMDManager: Added SMD dihedral between particles %d-%d-%d-%d",
				dihedral.getParticles().x,
				dihedral.getParticles().y,
				dihedral.getParticles().z,
				dihedral.getParticles().w);
	}

	/**
	 * @brief Get number of SMD bonds
	 * @return Number of SMD bonds
	 */
	size_t getNumBonds() const {
		return smd_bonds_.size();
	}

	/**
	 * @brief Get number of SMD dihedrals
	 * @return Number of SMD dihedrals
	 */
	size_t getNumDihedrals() const {
		return smd_dihedrals_.size();
	}

	/**
	 * @brief Enable or disable SMD interactions
	 * @param enable Whether to enable SMD
	 */
	void setEnabled(bool enable) {
		enabled_ = enable;
	}

	/**
	 * @brief Check if SMD is enabled
	 * @return true if enabled
	 */
	bool isEnabled() const {
		return enabled_;
	}

	/**
	 * @brief Compute SMD forces and energies
	 * @param positions Current particle positions
	 * @param forces Force array to update
	 * @param energies Energy array to update
	 * @param timestep Current timestep
	 * @param get_energy Whether to compute energies
	 */
	void computeSMDForces(const Vector3* positions,
						  Vector3* forces,
						  float* energies,
						  int timestep,
						  bool get_energy) {
		if (!enabled_)
			return;

		// Compute SMD bond forces
		for (const auto& bond : smd_bonds_) {
			computeBondForce(bond, positions, forces, energies, timestep, get_energy);
		}

		// Compute SMD angle forces
		for (const auto& angle : smd_angles_) {
			computeAngleForce(angle, positions, forces, energies, timestep, get_energy);
		}

		// Compute SMD dihedral forces
		for (const auto& dihedral : smd_dihedrals_) {
			computeDihedralForce(dihedral, positions, forces, energies, timestep, get_energy);
		}
	}

  private:
	std::shared_ptr<Resource> resources_;
	std::vector<SMDBond> smd_bonds_;
	std::vector<SMDAngle> smd_angles_;
	std::vector<SMDDihedral> smd_dihedrals_;
	bool enabled_;

	/**
	 * @brief Compute force for a single SMD bond
	 */
	void computeBondForce(const SMDBond& bond,
						  const Vector3* positions,
						  Vector3* forces,
						  float* energies,
						  int timestep,
						  bool get_energy) {
		// Get current bond parameters
		float r0 = bond.getCurrentBondLength(timestep);
		float k = bond.getSpringConstant();
		const int2& particles = bond.getParticles();

		// Calculate current bond vector and length
		Vector3 rvec = positions[particles.y] - positions[particles.x];
		float r = rvec.length();
		float dr = r - r0;

		// Calculate force magnitude
		float force_magnitude = -k * dr;

		// Calculate force vector
		Vector3 force_vector = (force_magnitude / r) * rvec;

		// Apply forces
		forces[particles.x] -= force_vector;
		forces[particles.y] += force_vector;

		// Calculate energy if requested
		if (get_energy) {
			float energy = 0.5f * k * dr * dr;
			energies[particles.x] += energy;
			energies[particles.y] += energy;
		}
	}

	void computeAngleForce(const SMDAngle& angle,
						   const Vector3* positions,
						   Vector3* forces,
						   float* energies,
						   int timestep,
						   bool get_energy) {
		// Get current angle parameters
		float angle0 = angle.getCurrentAngle(timestep);
		float k = angle.getSpringConstant();
		const int3& particles = angle.getParticles();

		// Calculate current angle vector and length
		Vector3 ab = positions[particles.y] - positions[particles.x];
		Vector3 bc = positions[particles.z] - positions[particles.y];
		Vector3 ac = positions[particles.z] - positions[particles.x];
		// todo:compute angle
		// todo: apply forces
		// todo: calculate energy if requested
		// todo: add energy to energies array
	}
	/**
	 * @brief Compute force for a single SMD dihedral
	 */
	void computeDihedralForce(const SMDDihedral& dihedral,
							  const Vector3* positions,
							  Vector3* forces,
							  float* energies,
							  int timestep,
							  bool get_energy) {
		// Get current dihedral parameters
		float theta0 = dihedral.getCurrentDihedralAngle(timestep);
		float k = dihedral.getSpringConstant();
		int4 particles = dihedral.getParticles();

		// Calculate dihedral angle and forces
		// This is a simplified implementation - full dihedral calculation
		// would involve proper angle calculation and force distribution

		// TODO: Implement full dihedral force calculation
		LOGDEBUG("SMDDihedral: Computing dihedral force for particles %d-%d-%d-%d",
				 particles.x,
				 particles.y,
				 particles.z,
				 particles.w);
	}
};

} // namespace ARBD
