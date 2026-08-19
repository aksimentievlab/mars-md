#pragma once

/**
 * @file SimManager.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation manager class. Manages the simulation loop and the parallelization.
 * @version 0.1
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 */

#include <fstream>
#include <future>
#include <iostream>

#include "ARBDException.h"
#include "ARBDLogger.h"
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
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/ReorderManager.h"
#include "System/RigidBodyManager.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"

// Q: what is our parallel heirarchy?
// A: Serial/openMP, MPI-only, Single-GPU, or NVSHMEM

// 1 Patch per MPI rank or GPU
// Patches should work independently with syncronization mediated by SimManager
// Patch to Patch data exchange should not require explicit scheduling by SimManager

namespace ARBD {

/**
 * @brief Simulation manager - handles the main simulation loop and runtime operations
 *
 * SimManager is responsible for:
 * - Managing the simulation loop
 * - Coordinating kernel execution across resources
 * - Handling output
 * - Managing inter-patch communication and synchronization
 * - Performance monitoring and reporting
 */
class SimManager {
  public:
	/**
	 * @brief Construct simulation manager
	 * @param sys Simulation system containing configuration and global objects
	 */
	SimManager(SimSystem& sys);
	/**
	 * @brief Construct simulation manager
	 * @param sys Simulation system containing configuration and global objects
	 * @param parser Configuration parser
	 */
	SimManager(SimSystem& sys, const ConfigParser& parser);

	/**
	 * @brief Initialize simulation manager
	 * Sets up decomposition, output writers, IMD, and initial conditions
	 */
	void init();

	/**
	 * @brief Initialize simulation manager configuration
	 * @param parser Configuration parser
	 */
	void load_config(const ConfigParser& parser);

	/**
	 * @brief Provide initial particle data to be loaded during init()
	 *
	 * Particles are cached and only converted (type_name -> type_id) inside
	 * init(), after particle type IDs have been assigned - calling
	 * SystemState::set_init_particle_data() any earlier would look up type
	 * names in an empty/unbuilt map.
	 * @param particles Initial particle data (host-side)
	 */
	void set_initial_particles(std::vector<ParticleIO> particles) {
		pending_initial_particles_ = std::move(particles);
	}

	/**
	 * @brief Provide parsed bonded interactions (bonds/angles/dihedrals/
	 * exclusions) to be loaded during init()
	 *
	 * The single-argument SimManager(SimSystem&) constructor - the one the
	 * main arbd executable actually uses - never calls load_config(), so
	 * without this the ConfigParser's parsed bonds never reach SystemState
	 * and calculate_bonded_forces() would silently have nothing to compute.
	 * @param bonded_interactions Parsed bonds/angles/dihedrals/exclusions (host-side)
	 */
	void set_bonded_interactions(const BondedInteractions& bonded_interactions) {
		pending_bonded_interactions_ = bonded_interactions;
	}

	/**
	 * @brief Provide initial rigid-body data to be loaded during init()
	 *
	 * Mirrors set_initial_particles() - cached until init() so it isn't
	 * needed until after rigid-body type IDs are assigned. Unlike particles,
	 * no name->id resolution happens here (ConfigParser already resolves
	 * RigidBodyIO::type_id at parse time), so this is just cached, not
	 * converted.
	 * @param bodies Initial rigid-body data (host-side)
	 */
	void set_initial_rigid_bodies(std::vector<RigidBodyIO> bodies) {
		pending_initial_rigid_bodies_ = std::move(bodies);
	}

	/**
	 * @brief Get the rigid-body manager, or nullptr if the system has no rigid body types
	 */
	const RigidBodyManager* get_rigid_body_manager() const {
		return rigid_body_manager_.get();
	}

	/**
	 * @brief Run the main simulation loop
	 * Executes the complete simulation with force calculation, integration, and I/O
	 */
	void run();

	/**
	 * @brief Write a PSF describing the trajectory this run produces.
	 *
	 * Atom order matches write_dcd_frame() exactly:
	 *   [regular particles][attached particles][cosmetic atoms]
	 * so the PSF and DCD load together in VMD. The first two blocks are the
	 * global particle array (ConfigParser appends attached particles after all
	 * regular ones); the tail is the rigid bodies' visualization-only template
	 * atoms, instance-major then template order.
	 *
	 * Bonds are the real bonded topology plus, per rigid-body instance, its
	 * template's own bonds remapped into this numbering - the latter purely
	 * cosmetic, so a body draws as a molecule rather than a cloud of dots.
	 *
	 * Call after init(); the structure is fixed for the run.
	 * @param path Output file; defaults to "<outputName>.psf".
	 */
	void write_psf(const std::string& path = "");

	/**
	 * @brief Write a PDB snapshot of the current positions.
	 *
	 * Same atom order and topology as write_psf(); usable as the companion
	 * structure file for the DCD, or as a coordinate snapshot on its own.
	 * @param path Output file; defaults to "<outputName>.pdb".
	 */
	void write_pdb(const std::string& path = "");

	/**
	 * @brief Get timing information
	 */
	float get_total_time() const {
		return static_cast<float>(wkf_timer_timenow(timer0_.timer));
	}
	float get_io_time() const {
		return wkf_timer_time(timerS_.timer);
	}
	float get_energy_time() const {
		return wkf_timer_time(timerE_.timer);
	}

  private:
	//================================================================================
	// Core Components
	//================================================================================
	SimSystem& sys_; // System owns PatchManager, accessible via sys_.get_patch_manager()
	SystemState sys_state_;

	// Initial particle data, cached until init() (needs particle type IDs assigned first)
	std::vector<ParticleIO> pending_initial_particles_{};
	// Initial rigid-body data, cached until init() (see set_initial_rigid_bodies)
	std::vector<RigidBodyIO> pending_initial_rigid_bodies_{};
	// Bonded interactions, cached until init() (see set_bonded_interactions)
	BondedInteractions pending_bonded_interactions_{};

	// Owns all rigid-body device state once the system has rigid body types;
	// null otherwise. Constructed in init(), after grids are on-device.
	std::unique_ptr<RigidBodyManager> rigid_body_manager_;

#ifdef ENABLE_ZORDER_REORDER
	// Intra-patch Morton reorder policy; constructed lazily once reorder_period > 0.
	std::unique_ptr<ParticleReorderManager> reorder_mgr_;
#endif

	// Random number generation
	size_t current_step_{0}; ///< Current simulation step (used for RNG counter)

	// Per-resource device particle-type tables, built once on first use.
	// Particle types are static for the whole run, so re-allocating and
	// re-uploading them every step (as execute_force_calculation used to)
	// wastes a device malloc/free + H2D copy per patch per step.
	std::vector<std::unique_ptr<DeviceParticleTypes>> device_particle_types_cache_;
	//================================================================================
	// Timing and Performance
	//================================================================================
	wkfmsgtimer timer0_, timerS_, timerE_, timerP_;

	//================================================================================
	// I/O and Output Management
	//================================================================================
	std::unique_ptr<TrajectoryWriter> traj_writer_;
	std::unique_ptr<DcdWriter> dcd_writer_;
	std::unique_ptr<PsfPdbStructure> structure_view_;
	bool dcd_header_written_{false};

	// Momentum trajectory (Langevin dynamics only, mirrors legacy ARBD's
	// "<name>.0.momentum.dcd" - the "0" is a legacy replica-index artifact
	// kept for output compatibility, not an actual replica count).
	std::unique_ptr<DcdWriter> momentum_dcd_writer_;
	bool momentum_dcd_header_written_{false};

	// Legacy plaintext rigid-body trajectory (see write_rb_traj_frame). Held
	// open for the run and flushed per frame, mirroring v1's trajFile.
	std::ofstream rb_traj_file_;

	/// Energy logs, held open for the run (see write_energy_output).
	std::ofstream energy_file_;
	std::ofstream rb_energy_file_;

	bool has_momentum_output_{false};
	bool has_rigid_bodies_{false};

	// Background restart-file writer. std::async's future blocks in its own
	// destructor until the task finishes, so this also guarantees a pending
	// write is never abandoned even if SimManager is destroyed unexpectedly.
	std::future<void> pending_restart_write_;

	//================================================================================
	// IMD (Interactive Molecular Dynamics) Support
	//================================================================================
	void* clientsock_{nullptr};
	bool imd_on_{false};

	//================================================================================
	// Initialization Methods
	//================================================================================

	/**
	 * @brief Initialize output writers based on configuration
	 */
	void initialize_output_writers();

	/**
	 * @brief Initialize IMD if requested
	 * @param port IMD port number
	 */
	void initialize_imd(int port);

	//================================================================================
	// Simulation Loop Components
	//================================================================================
	/**
	 * @brief Execute force calculation phase for all resources
	 * @param step Current simulation step
	 */
	void execute_force_calculation(size_t step);

	/**
	 * @brief Execute integration phase for all resources
	 * @param step Current simulation step
	 */
	void execute_integration(size_t step);

	/**
	 * @brief Synchronize multi-resource simulations (halo exchange)
	 */
	void synchronize_multi_resource();

	/**
	 * @brief Handle output operations (trajectory, energy, restart)
	 * @param step Current simulation step
	 */
	void handle_output(size_t step);

	/**
	 * @brief Bring momenta up to the current step before anything reads them
	 *
	 * BAOAB's closing half-kick is deferred to the next step, so between steps
	 * momentum trails position by half a kick. Call this before writing the
	 * momentum trajectory, the restart files, or kinetic energy. Re-evaluates
	 * forces at the current positions (the kick needs them) and is a no-op for
	 * integrators that defer nothing.
	 */
	void settle_momenta_for_output(size_t step);

	//================================================================================
	// Output Methods
	//================================================================================

	/**
	 * @brief Gather particle data from all patches into SystemState
	 * Collects positions, momenta, and IDs from all patches and assembles
	 * them into global arrays in SystemState for output.
	 * @param need_energy Also gather force/energy - only write_energy_output
	 *        reads these; DCD/restart writers don't, so they skip this
	 *        field-copy by default (see Patch::copy_particles_to_host).
	 */
	void gather_particle_data_from_patches(bool need_energy = false);

	/**
	 * @brief Refresh SystemState's rigid-body SoA from the device.
	 * @note No-op when there are no rigid bodies.
	 */
	void gather_rigid_body_data();

	/**
	 * @brief Write a single DCD trajectory frame
	 * @param step Current simulation step
	 */
	void write_dcd_frame(size_t step);

	/**
	 * @brief Write a single momentum DCD trajectory frame
	 * @note Reuses the SystemState gathered by write_dcd_frame() this tick -
	 *       must be called only after write_dcd_frame() within the same output step.
	 */
	void write_momentum_dcd_frame(size_t step);

	/**
	 * @brief Write one frame of the legacy plaintext rigid-body trajectory.
	 *
	 * Byte-format-compatible with ARBD v1's "<name>.0.rb-traj" so runs can be
	 * diffed directly against a v1 reference. One line per rigid body per
	 * output step:
	 *   step  <type>#<i>  pos(3)  orientation(9, row-major)  momentum(3)  angular_momentum(3)
	 * The last six columns are labelled vel/angVel by v1's own legend, but v1
	 * writes raw momenta there (RigidBody::getVelocity() returns `momentum`,
	 * its division by mass commented out), so this does the same.
	 */
	void write_rb_traj_frame(size_t step);

	/**
	 * @brief Populate structure_view_ with the trajectory's atoms and bonds.
	 *
	 * Built once and cached: the atom set is fixed for the run. Only the
	 * positions are refreshed (by write_pdb) on later calls.
	 */
	void build_structure_view();

	/// Refresh structure_view_'s coordinates from the current state.
	void refresh_structure_positions();

	/**
	 * @brief Compute and append kinetic/potential energy for this step
	 * Writes "<name>.energy.dat", and "<name>.rb_energy.dat" if the system
	 * has rigid body types, then refreshes the restart files.
	 */
	void write_energy_output(size_t step);

	/**
	 * @brief Overwrite the position (and, for Langevin dynamics, momentum)
	 * restart files with the current SystemState.
	 *
	 * The actual `to_chars` formatting + `fwrite` (the dominant cost of the
	 * output-path stall, measured at ~47% of it) runs on a background thread
	 * via `pending_restart_write_`, so the main simulation loop can go on
	 * launching the next step's kernels instead of blocking the GPU queue
	 * on host-side file I/O. Waits for any previous still-running write
	 * before dispatching a new one, so at most one write is ever in flight
	 * and two writes never race on the same file.
	 */
	void write_restart_files();

	/**
	 * @brief Block until any in-flight background restart write has
	 * finished. Must be called before the process can safely exit (the
	 * final restart write has to actually be on disk).
	 */
	void wait_for_pending_restart_write();
	/**
	 * @brief Report simulation progress
	 * @param current_step Current step
	 * @param total_steps Total steps
	 */
	void report_progress(size_t current_step, size_t total_steps, size_t report_period);

	/**
	 * @brief Report final performance statistics
	 * @param elapsed_time Total simulation time
	 * @param total_steps Total steps completed
	 */
	void report_performance(float elapsed_time, size_t total_steps);

	//================================================================================
	// IMD Methods
	//================================================================================

	/**
	 * @brief Handle IMD commands and communication
	 */
	void handle_imd_commands();

	//================================================================================
	// I/O Methods (SimManager handles file I/O, not SimSystem)
	//================================================================================

	/**
	 * @brief Load initial conditions from files or generate them
	 */
	void load_initial_conditions();

	/**
	 * @brief Generate initial particle positions and types
	 * @param positions Output vector for particle positions
	 * @param types Output vector for particle types
	 */
	void generate_initial_particles(std::vector<Vector3>& positions, std::vector<int>& types);
	/**
	 * @brief Generate initial particle momentum and types according to Boltzmann distribution
	 * @param momentum Output vector for particle momentum
	 */
	void generate_initial_momentum(const Vector3& v_com);
	/**
	 * @brief Write final restart file
	 */
	void write_final_restart();
	void load_restart_data(const std::string& filename);

	/**
	 * @brief Perform particle reactions
	 */
	void perform_reactions();
};

} // namespace ARBD
