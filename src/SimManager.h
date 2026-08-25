#pragma once

/**
 * @file SimManager.h
 * @brief Declares the simulation loop and output coordinator. Runtime control only
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @version 0.1
 * @date 2025-09-09
 */

#include <fstream>
#include <future>
#include <iostream>

#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "IO/ConfigParser.h"
#include "IO/DcdWriter.h"
#include "IO/PsfPdbIO.h"
#include "IO/TrajectoryWriter.h"
#include "IO/WKFUtils.h"
#include "MARSException.h"
#include "MARSLogger.h"
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/ReorderManager.h"
#include "System/RigidBodyManager.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"

namespace MARS {

/**
 * @brief Coordinates simulation, device execution, synchronization, and output.
 * @note Initialization must precede simulation and structure-file output.
 * @todo Expose replica configuration and implement interactive molecular dynamics.
 */
class SimManager {
  public:
	/** @brief Constructs a manager for a simulation system.
	 * @param sys Simulation system containing configuration and global objects.
	 */
	SimManager(SimSystem& sys);

	/** @brief Constructs and configures a simulation manager.
	 * @param sys Simulation system containing configuration and global objects.
	 * @param parser Configuration parser.
	 */
	SimManager(SimSystem& sys, const ConfigParser& parser);

	/** @brief Initializes decomposition, device data, rigid bodies, and output.
	 * @note Initial data must be staged before initialization.
	 */
	void init();

	/** @brief Loads simulation configuration into the owned system state.
	 * @param parser Configuration parser.
	 */
	void load_config(const ConfigParser& parser);

	/** @brief Stages initial particle data for initialization.
	 * @param particles Host-side particle data.
	 * @note Particle type names are resolved during init().
	 */
	void set_initial_particles(std::vector<ParticleIO> particles) {
		pending_initial_particles_ = std::move(particles);
	}

	/** @brief Stages bonded interactions for initialization.
	 * @param bonded_interactions Host-side bonded interactions.
	 */
	void set_bonded_interactions(const BondedInteractions& bonded_interactions) {
		pending_bonded_interactions_ = bonded_interactions;
	}

	/** @brief Stages initial rigid-body data for initialization.
	 * @param bodies Host-side rigid-body data.
	 * @note Rigid-body type IDs must already be resolved.
	 */
	void set_initial_rigid_bodies(std::vector<RigidBodyIO> bodies) {
		pending_initial_rigid_bodies_ = std::move(bodies);
	}

	/** @brief Returns the rigid-body manager when configured.
	 * @return A read-only manager pointer, or nullptr when none exists.
	 */
	const RigidBodyManager* get_rigid_body_manager() {
		return rigid_body_manager_.get();
	};

	void run();

	/** @brief Writes a PSF for the simulation trajectory.
	 * @param path Output path, or an empty string for the configured output name.
	 * @note Call after init(); atom order matches DCD output.
	 */
	void write_psf(const std::string& path = "");

	/** @brief Writes a PDB snapshot of the current simulation positions.
	 * @param path Output path, or an empty string for the configured output name.
	 * @note Call after init(); atom order matches PSF and DCD output.
	 */
	void write_pdb(const std::string& path = "");

	/** @brief Returns elapsed wall-clock time since construction.
	 * @return Total elapsed time in seconds.
	 */
	float get_total_time() const;

	/** @brief Returns accumulated output time.
	 * @return Output time in seconds.
	 */
	float get_io_time() const;

	/** @brief Returns accumulated energy-output time.
	 * @return Energy-output time in seconds.
	 */
	float get_energy_time() const;

  private:
	SimSystem& sys_;
	SystemState sys_state_;
	std::vector<ParticleIO> pending_initial_particles_{};
	std::vector<RigidBodyIO> pending_initial_rigid_bodies_{};
	BondedInteractions pending_bonded_interactions_{};
	std::unique_ptr<RigidBodyManager> rigid_body_manager_;
#ifdef ENABLE_ZORDER_REORDER
	std::unique_ptr<ParticleReorderManager> reorder_mgr_;
#endif
	size_t current_step_{0};
	std::vector<std::unique_ptr<DeviceParticleTypes>> device_particle_types_cache_;
	wkfmsgtimer timer0_, timerS_, timerE_, timerP_;
	std::unique_ptr<TrajectoryWriter> traj_writer_;
	std::unique_ptr<DcdWriter> dcd_writer_;
	std::unique_ptr<PsfPdbStructure> structure_view_;
	bool dcd_header_written_{false};
	std::unique_ptr<DcdWriter> momentum_dcd_writer_;
	bool momentum_dcd_header_written_{false};
	std::ofstream rb_traj_file_;
	std::ofstream energy_file_;
	std::ofstream rb_energy_file_;
	bool has_momentum_output_{false};
	bool has_rigid_bodies_{false};
	std::future<void> pending_restart_write_;
	void* clientsock_{nullptr};
	bool imd_on_{false};

	/** @brief Initializes configured output writers.
	 */
	void initialize_output_writers();

	/** @brief Initializes interactive molecular dynamics support.
	 * @param port Listening port.
	 */
	void initialize_imd(int port);

	/** @brief Executes force calculation for all patches.
	 * @param step Current simulation step.
	 */
	void execute_force_calculation(size_t step);

	/** @brief Executes particle and rigid-body integration.
	 * @param step Current simulation step.
	 */
	void execute_integration(size_t step);

	/** @brief Synchronizes state across multiple resources.
	 * @todo Implement halo exchange through PatchManager.
	 */
	void synchronize_multi_resource();

	/** @brief Performs trajectory, energy, and restart output.
	 * @param step Current simulation step.
	 */
	void handle_output(size_t step);

	/** @brief Completes deferred momentum updates before output.
	 * @param step Current simulation step.
	 * @note Applies the deferred half-kick when required.
	 */
	void settle_momenta_for_output(size_t step);

	/** @brief Gathers particle data from patches into SystemState.
	 * @param need_energy Whether force and energy fields are required.
	 */
	void gather_particle_data_from_patches(bool need_energy = false);

	/** @brief Gathers rigid-body data from device storage.
	 * @note Does nothing when no rigid bodies are configured.
	 */
	void gather_rigid_body_data();

	/** @brief Writes one position trajectory frame.
	 * @param step Current simulation step.
	 */
	void write_dcd_frame(size_t step);

	/** @brief Writes one momentum trajectory frame.
	 * @param step Current simulation step.
	 * @note Requires a particle gather from the same output tick.
	 */
	void write_momentum_dcd_frame(size_t step);

	/** @brief Writes one rigid-body trajectory frame.
	 * @param step Current simulation step.
	 */
	void write_rb_traj_frame(size_t step);

	/** @brief Builds the cached structure atom and bond view.
	 */
	void build_structure_view();

	/** @brief Refreshes cached structure coordinates.
	 */
	void refresh_structure_positions();

	/** @brief Writes energy reports and refreshes restart files.
	 * @param step Current simulation step.
	 */
	void write_energy_output(size_t step);

	/** @brief Starts an asynchronous restart-file write.
	 * @note Waits for the previous write before starting another.
	 */
	void write_restart_files();

	/** @brief Waits for an asynchronous restart-file write.
	 */
	void wait_for_pending_restart_write();

	/** @brief Reports progress for the current simulation interval.
	 * @param current_step Current simulation step.
	 * @param total_steps Total configured steps.
	 * @param report_period Steps represented by the interval.
	 */
	void report_progress(size_t current_step, size_t total_steps, size_t report_period);

	/** @brief Reports final timing statistics.
	 * @param elapsed_time Total elapsed time in seconds.
	 * @param total_steps Completed simulation steps.
	 */
	void report_performance(float elapsed_time, size_t total_steps);

	/** @brief Handles interactive molecular dynamics commands.
	 * @todo Implement command polling and state updates.
	 */
	void handle_imd_commands();

	/** @brief Loads or generates initial conditions.
	 */
	void load_initial_conditions();

	/** @brief Generates initial particle positions and types.
	 * @param positions Destination positions.
	 * @param types Destination type IDs.
	 */
	void generate_initial_particles(std::vector<Vector3>& positions, std::vector<int>& types);

	/** @brief Generates initial momenta from a Boltzmann distribution.
	 * @param v_com Desired center-of-mass velocity.
	 * @note Grid temperatures are unsupported.
	 * @todo Route generated momenta into the initial-state storage.
	 */
	void generate_initial_momentum(const Vector3& v_com);

	/** @brief Writes final position and momentum restart files.
	 */
	void write_final_restart();

	/** @brief Performs configured particle reactions.
	 * @todo Implement reaction, compaction, topology, and neighbor-list updates.
	 */
	void perform_reactions();
};

} // namespace MARS
