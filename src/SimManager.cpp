#include "SimManager.h"
#include "PatchOperation/ZOrderKernels/ZOrderSort.h"
#include "System/PatchManager.h"
#include <charconv>
#include <cstdio>
#include <fstream>
#include <random>
#include <unordered_map>

namespace ARBD {

namespace {
void append_restart_line(std::string& buf, int type_id, const Vector3& v) {
	char tmp[64];
	auto append = [&](auto value) {
		auto res = std::to_chars(tmp, tmp + sizeof(tmp), value);
		buf.append(tmp, res.ptr);
	};
	append(type_id);
	buf.push_back(' ');
	append(v.x);
	buf.push_back(' ');
	append(v.y);
	buf.push_back(' ');
	append(v.z);
	buf.push_back('\n');
}

std::string rigid_body_segname(const std::string& type_name) {
	std::string s = type_name.substr(0, type_name.find('.'));
	if (s.size() > 4) {
		s.resize(4);
	}
	return s.empty() ? std::string("RB") : s;
}
} // namespace

SimManager::SimManager(SimSystem& sys) : sys_(sys), sys_state_(sys) {
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
	timerP_.timer = wkf_timer_create();
}

SimManager::SimManager(SimSystem& sys, const ConfigParser& parser) : sys_(sys), sys_state_(sys) {
	load_config(parser);
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
	timerP_.timer = wkf_timer_create();
}

void SimManager::load_config(const ConfigParser& parser) {
	sys_.set_temperature(parser.get_sim_system().get_temperature());
	sys_.set_cutoff(parser.get_sim_system().get_cutoff());
	sys_.set_timestep(parser.get_sim_system().get_timestep());
	sys_.set_num_steps(parser.get_sim_system().get_num_steps());
	sys_.set_neighbor_list_rebuild_period(
		parser.get_sim_system().get_neighbor_list_rebuild_period());
	sys_.set_reorder_period(parser.get_sim_system().get_reorder_period());
	sys_.set_output_period(parser.get_sim_system().get_output_period());
	sys_.set_energy_output_period(parser.get_sim_system().get_energy_output_period());
	sys_.set_output_name(parser.get_sim_system().get_output_name());
	sys_.set_output_format(parser.get_sim_system().get_output_format());
	sys_.set_decomposer_type(parser.get_sim_system().get_decomposer_type());
	sys_.set_long_range_method(parser.get_sim_system().get_long_range_method());
	sys_.set_particle_integrator_type(parser.get_sim_system().get_particle_algorithm());
	sys_.set_rigid_body_integrator_type(parser.get_sim_system().get_rigid_body_algorithm());
	sys_.set_particle_types(parser.get_sim_system().get_particle_types());
	sys_.set_rigid_body_types(parser.get_sim_system().get_rigid_body_types());
	sys_.set_rb_update_period(parser.get_sim_system().get_rb_update_period());
	sys_.set_base_seed(parser.get_sim_system().get_base_seed());

	pending_initial_particles_ = parser.get_init_particles();
	pending_initial_rigid_bodies_ = parser.get_init_rigid_bodies();
	sys_state_.update_bonded_interactions(parser.get_init_bonded_interactions());
}

void SimManager::init() {
	LOGINFO("SimManager: Initializing simulation");

	initialize_output_writers();

	if (!sys_.get_decomposer()) {
		LOGINFO("SimManager: Setting up default spatial decomposer");
		sys_.set_decomposer_type(sys_.get_decomposer_type());
	}

	sys_.assign_particle_type_ids();
	LOGINFO("SimManager: Particle type IDs assigned");
	sys_.assign_rigid_body_type_ids();
	LOGINFO("SimManager: Rigid body type IDs assigned");
	sys_.build_name_to_id_maps();
	LOGINFO("SimManager: Name to ID maps built");

	if (!pending_initial_particles_.empty()) {
		std::unordered_map<int, int> uid_to_index;
		for (size_t i = 0; i < pending_initial_particles_.size(); ++i) {
			auto& particle = pending_initial_particles_[i];
			particle.id = static_cast<int>(i);
			if (particle.uid >= 0) {
				uid_to_index[particle.uid] = static_cast<int>(i);
			}
		}
		if (!uid_to_index.empty()) {
			pending_bonded_interactions_.resolve_particle_uids([&](int uid) {
				auto it = uid_to_index.find(uid);
				if (it == uid_to_index.end()) {
					throw_value_error("SimManager: bonded interaction references a particle that "
									  "was not staged (uid %d)",
									  uid);
				}
				return it->second;
			});
		}

		sys_state_.set_init_particle_data(pending_initial_particles_);
		LOGINFO("SimManager: Loaded {} initial particles into system state",
				pending_initial_particles_.size());
	}

	if (!pending_initial_rigid_bodies_.empty()) {
		sys_state_.set_init_rigid_body_data(pending_initial_rigid_bodies_);
		LOGINFO("SimManager: Loaded {} initial rigid bodies into system state",
				pending_initial_rigid_bodies_.size());
	}

	if (pending_bonded_interactions_.get_num_bonds() > 0 ||
		pending_bonded_interactions_.get_num_angles() > 0 ||
		pending_bonded_interactions_.get_num_dihedrals() > 0 ||
		!pending_bonded_interactions_.get_exclusions().empty() ||
		!pending_bonded_interactions_.get_restraints().empty()) {
		sys_state_.update_bonded_interactions(pending_bonded_interactions_);
		LOGINFO("SimManager: Loaded {} bonds, {} angles, {} dihedrals, {} exclusions, {} "
				"restraints into system state",
				pending_bonded_interactions_.get_num_bonds(),
				pending_bonded_interactions_.get_num_angles(),
				pending_bonded_interactions_.get_num_dihedrals(),
				pending_bonded_interactions_.get_exclusions().size(),
				pending_bonded_interactions_.get_restraints().size());
	}

	LOGINFO("SimManager: Performing domain decomposition");
	sys_.decompose_system(sys_state_);

	if (!sys_.has_patch_manager()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Domain decomposition failed to create PatchManager");
	}
	LOGINFO("SimManager: Domain decomposition complete");

	sys_.get_patch_manager()->distribute_particles_from_state(sys_state_);
	LOGINFO("SimManager: Initial particles distributed to patches");

	sys_.get_grid_manager().build_device_arrays();
	LOGINFO("SimManager: Grids transferred to all resources");
	sys_.get_tables_registry().build_device_arrays();
	LOGINFO("SimManager: Tables transferred to all resources");
	sys_.get_nonbonded_interactions().prepare_device_data();
	LOGINFO("SimManager: Nonbonded interactions transferred to all resources");

	if (!sys_.get_rigid_body_types().empty()) {
		rigid_body_manager_ = std::make_unique<RigidBodyManager>(sys_.get_resources());
		rigid_body_manager_->initialize(
			sys_.get_rigid_body_types(),
			sys_state_.get_global_rigid_bodies(),
			[this](int id) { return sys_.get_grid_manager().get_grid_format(id); },
			sys_.get_rb_update_period());
		{
			const size_t rb_count = pending_initial_rigid_bodies_.size();
			std::vector<Vector3> constant_force(rb_count);
			std::vector<Vector3> constant_torque(rb_count);
			for (size_t i = 0; i < rb_count; ++i) {
				constant_force[i] = pending_initial_rigid_bodies_[i].external_force;
				constant_torque[i] = pending_initial_rigid_bodies_[i].external_torque;
			}
			rigid_body_manager_->set_external_loads(constant_force, constant_torque);
		}
		rigid_body_manager_->prepare_grid_grid_dispatch(sys_.get_grid_manager(), 0);
		rigid_body_manager_->prepare_particle_grid_dispatch(sys_.get_rigid_body_types(),
															sys_state_.get_num_particles());
		rigid_body_manager_->prepare_attached_particles(
			sys_.get_rigid_body_types(),
			pending_initial_rigid_bodies_,
			sys_.get_patch_manager() ? sys_.get_patch_manager()->get_patches().size() : 1);
		rigid_body_manager_->prepare_cosmetic_atoms(sys_.get_rigid_body_types(),
													pending_initial_rigid_bodies_);
		LOGINFO("SimManager: Rigid body manager initialized with {} bodies",
				sys_state_.get_num_rigid_bodies());

		const IntegratorType rb_algorithm = sys_.get_rigid_body_algorithm();
		if (rb_algorithm == IntegratorType::VelocityVerlet) {
			LOGWARN("SimManager: VelocityVerlet rigid-body dynamics is not implemented; "
					"falling back to Langevin/DLM");
		}
		for (const RigidBodyType& rb_type : sys_.get_rigid_body_types()) {
			rb_type.check_damping(rb_algorithm);
		}
	}

	try {
		write_psf();
		write_pdb();
	} catch (const std::exception& e) {
		LOGWARN("SimManager: could not write structure files ({}) - the simulation will still run, "
				"but the trajectory will have no matching PSF/PDB",
				e.what());
	}

	LOGINFO("SimManager: Initialization completed");
}

void SimManager::run() {
	LOGINFO("SimManager: Starting simulation loop");

	const size_t num_steps = sys_.get_num_steps();
	const size_t output_period = static_cast<size_t>(sys_.get_output_period());
	const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());
	const auto& resources = sys_.get_resources();

	LOGINFO("SimManager: Running {} steps with {} resources", num_steps, resources.size());

	const size_t progress_period = energy_output_period > 0 ? energy_output_period : 1000;
	const int num_replicas = 1;

	std::printf("Configuration: %zu particles | %d replicas\n",
				sys_state_.get_num_particles(),
				num_replicas);
	std::fflush(stdout);

	wkf_timer_start(timer0_.timer);
	wkf_timer_start(timerP_.timer);

	const bool split_dlm = rigid_body_manager_ && rigid_body_manager_->size() > 0 &&
						   sys_.get_rigid_body_algorithm() != IntegratorType::Brownian;
	const PeriodicBox& sim_box = sys_.get_boundary_conditions();
	const float dt = sys_.get_timestep();

	if (split_dlm) {
		execute_force_calculation(0);
	}

	for (size_t step = 1; step <= num_steps; ++step) {
		if (split_dlm) {
			rigid_body_manager_->integrate_drift(dt, sim_box); //.wait();
			sys_state_.invalidate_rigid_bodies();
		}

		execute_force_calculation(step);

		if (split_dlm) {
			rigid_body_manager_->integrate_kick(dt, sim_box); //.wait();
			sys_state_.invalidate_rigid_bodies();
		}

		execute_integration(step);

		if (resources.size() > 1) {
			synchronize_multi_resource();
		}

		handle_output(step);

		if (imd_on_ && clientsock_) {
			handle_imd_commands();
		}

		if (step % progress_period == 0) {
			report_progress(step, num_steps, progress_period);
		}
	}

	std::printf("\n");
	std::fflush(stdout);

	wkf_timer_stop(timer0_.timer);
	const float elapsed = wkf_timer_time(timer0_.timer);

	report_performance(elapsed, num_steps);
	settle_momenta_for_output(num_steps);
	write_final_restart();

	if (imd_on_ && clientsock_) {
	}

	LOGINFO("SimManager: Simulation completed successfully");
}

void SimManager::initialize_output_writers() {
	std::string output_name = sys_.get_output_name();

	dcd_writer_ = std::make_unique<DcdWriter>(output_name + ".dcd");
	LOGINFO("SimManager: Initialized DCD writer for '{}.dcd'", output_name);

	has_momentum_output_ = (sys_.get_particle_algorithm() == IntegratorType::Langevin);
	if (has_momentum_output_) {
		momentum_dcd_writer_ = std::make_unique<DcdWriter>(output_name + ".0.momentum.dcd");
		LOGINFO("SimManager: Initialized momentum DCD writer for '{}.0.momentum.dcd'", output_name);
	}

	has_rigid_bodies_ = !sys_.get_rigid_body_types().empty();
}

void SimManager::initialize_imd(int port) {
	LOGINFO("SimManager: IMD initialization (port {}) not yet implemented", port);
}

void SimManager::execute_force_calculation(size_t step) {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for force calculation");
	}

#ifdef ENABLE_ZORDER_REORDER
	if (sys_.get_reorder_period() > 0) {
		auto& reorder_patches = patch_mgr->get_patches();
		if (!reorder_patches.empty() && reorder_patches.front()) {
			if (!reorder_mgr_) {
				reorder_mgr_ = std::make_unique<ParticleReorderManager>(sys_.get_reorder_period());
			}
			auto& reorder_patch = reorder_patches.front();
			if (reorder_mgr_->maybe_reorder(step, *reorder_patch) && rigid_body_manager_) {
				rigid_body_manager_->remap_attached_particle_indices(
					reorder_patch->reorder_sorter());
			}
		}
	}
#endif

	if (rigid_body_manager_) {
		rigid_body_manager_->bodies().clear_forces();

		if (rigid_body_manager_->has_attached_particles()) {
			auto& patches = patch_mgr->get_patches();
			if (!patches.empty()) {
				rigid_body_manager_
					->sync_attached_particle_positions(patches.front()->get_particles().view())
					.wait();
			}
		}
	}

	const auto& resources = sys_.get_resources();
	const auto& grid_manager = sys_.get_grid_manager();

	if (device_particle_types_cache_.size() < resources.size()) {
		device_particle_types_cache_.resize(resources.size());
	}

	for (auto& patch : patch_mgr->get_patches()) {
		size_t resource_idx = 0;
		for (size_t i = 0; i < resources.size(); ++i) {
			if (resources[i] == patch->get_resource()) {
				resource_idx = i;
				break;
			}
		}

		if (!device_particle_types_cache_[resource_idx]) {
			device_particle_types_cache_[resource_idx] =
				std::make_unique<DeviceParticleTypes>(sys_.get_particle_types(),
													  patch->get_resource());
		}
		DeviceParticleTypes& particle_types = *device_particle_types_cache_[resource_idx];

		const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());
		const bool compute_energy = energy_output_period > 0 && step % energy_output_period == 0;
		Event evt = patch->calculate_nonbonded_forces(
			sys_.get_nonbonded_interactions(),
			sys_state_.get_bonded_interactions(),
			particle_types,
			grid_manager.get_device_grid_views(resource_idx),
			sys_.get_tables_registry(),
			resource_idx,
			static_cast<float>(sys_.get_pairlist_cutoff()),
			static_cast<float>(sys_.get_cutoff()),
			step,
			static_cast<size_t>(sys_.get_neighbor_list_rebuild_period()),
			Vector3{0.0, 0.0, 0.0},
			0,
			compute_energy);

		Event bonded_evt = patch->calculate_bonded_forces(sys_state_.get_bonded_interactions(),
														  particle_types,
														  sys_.get_tables_registry(),
														  resource_idx,
														  compute_energy);

		// bonded_evt.wait();
		// evt.wait();
	}

	if (rigid_body_manager_) {
		if (rigid_body_manager_->has_attached_particles()) {
			auto& patches = patch_mgr->get_patches();
			if (!patches.empty()) {
				rigid_body_manager_->reduce_attached_particle_forces(
					std::as_const(patches.front()->get_particles()).view());
				//.wait();
			}
		}

		Event grid_evt =
			rigid_body_manager_->compute_grid_grid_forces(grid_manager,
														  0,
														  step,
														  static_cast<float>(sys_.get_cutoff()));

		auto& patches = patch_mgr->get_patches();
		if (!patches.empty()) {
			Event particle_rb_evt = rigid_body_manager_->compute_particle_rb_forces(
				grid_manager,
				0,
				patches.front()->get_particles().view());
			particle_rb_evt.wait();
		}
		grid_evt.wait();

		if (sys_.get_rigid_body_algorithm() != IntegratorType::Brownian) {
			const Temperature& temperature = sys_.get_temperature_struct();
			rigid_body_manager_->add_langevin_forces(sys_.get_timestep(),
													 temperature.kT,
													 sys_.get_base_seed(),
													 step);
			//.wait();
		}
	}

	if (step == 1) {
		LOGINFO("SimManager: PMF/grid and pairwise nonbonded force kernels launched");
	}
}

void SimManager::execute_integration(size_t step) {
	const IntegratorType particle_algorithm = sys_.get_particle_algorithm();
	const IntegratorType rigidbody_algorithm = sys_.get_rigid_body_algorithm();
	const float timestep = sys_.get_timestep();
	const Temperature& temperature = sys_.get_temperature_struct();

	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for integration");
	}

	for (auto& patch : patch_mgr->get_patches()) {
		Event evt = patch->integrate_motion(timestep, temperature, particle_algorithm, step);
		evt.wait();
	}

	if (rigid_body_manager_) {
		if (sys_.get_rigid_body_algorithm() == IntegratorType::Brownian) {
			const Temperature& temperature = sys_.get_temperature_struct();
			rigid_body_manager_
				->integrate_brownian(timestep,
									 temperature.kT,
									 sys_.get_base_seed(),
									 step,
									 sys_.get_boundary_conditions())
				.wait();
		}
		sys_state_.invalidate_rigid_bodies();
	}

	current_step_ = step;
}

void SimManager::synchronize_multi_resource() {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		LOGWARN("SimManager: Cannot synchronize - PatchManager not available");
		return;
	}

	static bool logged_once = false;
	if (!logged_once) {
		LOGINFO("SimManager: Multi-resource synchronization not yet implemented");
		logged_once = true;
	}
}

void SimManager::settle_momenta_for_output(size_t step) {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		return;
	}

	bool any_pending = false;
	for (auto& patch : patch_mgr->get_patches()) {
		any_pending = any_pending || patch->has_deferred_kick();
	}
	if (!any_pending) {
		return;
	}

	execute_force_calculation(step);
	const float timestep = sys_.get_timestep();
	for (auto& patch : patch_mgr->get_patches()) {
		patch->finish_deferred_kick(timestep).wait();
	}
}

void SimManager::handle_output(size_t step) {
	const size_t output_period = static_cast<size_t>(sys_.get_output_period());
	const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());

	const bool trajectory_due = output_period > 0 && step % output_period == 0;
	const bool energy_due = energy_output_period > 0 && step % energy_output_period == 0;

	if ((trajectory_due && momentum_dcd_writer_) || energy_due) {
		settle_momenta_for_output(step);
	}

	if (output_period > 0 && step % output_period == 0) {
		wkf_timer_start(timerS_.timer);

		if (dcd_writer_) {
			write_dcd_frame(step);
			if (momentum_dcd_writer_) {
				write_momentum_dcd_frame(step);
			}
		} else if (traj_writer_) {
		}

		write_rb_traj_frame(step);

		wkf_timer_stop(timerS_.timer);
	}

	if (energy_output_period > 0 && step % energy_output_period == 0) {
		wkf_timer_start(timerE_.timer);
		write_energy_output(step);
		wkf_timer_stop(timerE_.timer);
	}
}

void SimManager::gather_particle_data_from_patches(bool need_energy) {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	sys_state_.clear_global_arrays();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for gathering particle data");
	}
	patch_mgr->gather_particles_to_state(sys_state_, need_energy);

	sys_state_.mark_synced();
}

void SimManager::gather_rigid_body_data() {
	if (!rigid_body_manager_ || rigid_body_manager_->size() == 0) {
		return;
	}
	rigid_body_manager_->gather_to_host(sys_state_.mutable_rigid_bodies());
	sys_state_.mark_rigid_bodies_synced();
}

void SimManager::write_dcd_frame(size_t step) {
	gather_particle_data_from_patches();
	if (sys_state_.prepare_for_dcd_output()) {
		const auto& positions = sys_state_.get_global_positions();
		std::vector<Vector3> frame(positions.begin(), positions.end());
		if (rigid_body_manager_ && rigid_body_manager_->num_cosmetic_atoms() > 0) {
			rigid_body_manager_->compute_cosmetic_positions().wait();
			rigid_body_manager_->copy_cosmetic_positions_to_host(frame);
		}

		const auto& periodicity = sys_.get_boundary_conditions().get_periodicity();
		const bool with_unitcell = periodicity[0] || periodicity[1] || periodicity[2];

		if (!dcd_header_written_) {
			const int nsavc = std::max(1, static_cast<int>(sys_.get_output_period()));
			dcd_writer_->writeHeader(static_cast<int>(frame.size()),
									 1,
									 nsavc,
									 nsavc,
									 0,
									 sys_.get_timestep(),
									 with_unitcell);
			dcd_header_written_ = true;
			LOGINFO("SimManager: DCD has {} atom(s) = {} particle(s) + {} cosmetic",
					frame.size(),
					positions.size(),
					frame.size() - positions.size());
		}

		if (with_unitcell) {
			const Vector3 box = sys_.get_boundary_conditions().get_box_size();
			const std::vector<double> unitcell{box.x, 0.0, box.y, 0.0, 0.0, box.z};
			dcd_writer_->writeStep(frame, unitcell);
		} else {
			dcd_writer_->writeStep(frame);
		}
	}
}

void SimManager::build_structure_view() {
	if (structure_view_) {
		return;
	}
	structure_view_ = std::make_unique<PsfPdbStructure>();
	PsfPdbStructure& s = *structure_view_;

	const Vector3 box = sys_.get_boundary_conditions().get_box_size();
	s.box_dimensions = box;
	s.has_cryst1 = true;

	const HostParticleData& particles = sys_state_.get_global_particles();
	const auto& ptypes = sys_.get_particle_types();
	const size_t num_particles = particles.size();

	s.atoms.reserve(num_particles);
	for (size_t i = 0; i < num_particles; ++i) {
		const int tid = particles.type_id[i];
		const bool known = tid >= 0 && static_cast<size_t>(tid) < ptypes.size();

		PdbAtomRecord a;
		a.serial = static_cast<int>(i) + 1;
		a.name = known ? ptypes[tid].name : "X";
		a.resname = a.name;
		a.type_name = a.name;
		a.mass = known ? ptypes[tid].mass : 1.0f;
		a.charge = known ? ptypes[tid].charge : 0.0f;
		a.resid = static_cast<int>(i) + 1;
		a.chain = "A";
		const bool is_attached =
			i < particles.attached_rigid_body_id.size() && particles.attached_rigid_body_id[i] >= 0;
		a.segname = is_attached ? "ATT" : "SYS";
		if (is_attached) {
			a.beta = static_cast<float>(particles.attached_rigid_body_id[i]);
		}
		a.position = i < particles.pos.size() ? particles.pos[i] : Vector3(0.0f);
		s.atoms.push_back(std::move(a));
	}

	for (const Bond& b : sys_state_.get_bonded_interactions().get_bonds()) {
		s.bonds.emplace_back(b.ind1, b.ind2);
	}

	const auto& rtypes = sys_.get_rigid_body_types();
	for (const RigidBodyIO& rb : pending_initial_rigid_bodies_) {
		if (rb.type_id < 0 || static_cast<size_t>(rb.type_id) >= rtypes.size()) {
			continue;
		}
		const RigidBodyType& type = rtypes[rb.type_id];
		std::vector<int> template_to_global(type.template_particles.size(), -1);

		for (size_t t = 0; t < type.template_particles.size(); ++t) {
			const CosmeticParticle& c = type.template_particles[t];
			if (c.attached_particle_index >= 0) {
				template_to_global[t] = rb.attached_start + c.attached_particle_index;
				continue;
			}
			template_to_global[t] = static_cast<int>(s.atoms.size());

			PdbAtomRecord a;
			a.serial = static_cast<int>(s.atoms.size()) + 1;
			a.name = c.name.empty() ? c.resname : c.name;
			a.resname = c.resname;
			a.type_name = c.type_name.empty() ? c.resname : c.type_name;
			a.resid = c.resid;
			a.chain = "R";
			a.segname = rigid_body_segname(type.name);
			a.beta = static_cast<float>(rb.id);
			a.mass = 0.0f;
			a.charge = 0.0f;
			a.position = rb.orientation * c.body_frame_position + rb.position;
			s.atoms.push_back(std::move(a));
		}

		for (const int2& b : type.template_bonds) {
			if (b.x < 0 || b.y < 0 || static_cast<size_t>(b.x) >= template_to_global.size() ||
				static_cast<size_t>(b.y) >= template_to_global.size()) {
				continue;
			}
			const int g1 = template_to_global[b.x];
			const int g2 = template_to_global[b.y];
			if (g1 >= 0 && g2 >= 0) {
				s.bonds.emplace_back(g1, g2);
			}
		}
	}

	if (s.atoms.empty()) {
		structure_view_.reset();
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"SimManager: no particles or rigid-body template atoms to write - was "
						"init() called, and does the system actually contain anything?");
	}
}

void SimManager::refresh_structure_positions() {
	if (!structure_view_) {
		return;
	}
	PsfPdbStructure& s = *structure_view_;

	gather_particle_data_from_patches();
	std::vector<Vector3> frame;
	if (sys_state_.prepare_for_dcd_output()) {
		const auto& positions = sys_state_.get_global_positions();
		frame.assign(positions.begin(), positions.end());
	}
	if (rigid_body_manager_ && rigid_body_manager_->num_cosmetic_atoms() > 0) {
		rigid_body_manager_->compute_cosmetic_positions().wait();
		rigid_body_manager_->copy_cosmetic_positions_to_host(frame);
	}

	const size_t n = std::min(frame.size(), s.atoms.size());
	for (size_t i = 0; i < n; ++i) {
		s.atoms[i].position = frame[i];
	}
	if (frame.size() != s.atoms.size()) {
		LOGWARN("SimManager: structure has {} atom(s) but the current frame has {} - "
				"only the overlap was refreshed",
				s.atoms.size(),
				frame.size());
	}
}

void SimManager::write_psf(const std::string& path) {
	build_structure_view();
	const std::string out = path.empty() ? sys_.get_output_name() + ".psf" : path;
	structure_view_->write_psf(out);
	LOGINFO("SimManager: wrote PSF '{}' ({} atoms, {} bonds)",
			out,
			structure_view_->atoms.size(),
			structure_view_->bonds.size());
}

void SimManager::write_pdb(const std::string& path) {
	build_structure_view();
	refresh_structure_positions();
	const std::string out = path.empty() ? sys_.get_output_name() + ".pdb" : path;
	structure_view_->write_pdb(out, sys_.get_boundary_conditions().get_box_size());
	LOGINFO("SimManager: wrote PDB '{}' ({} atoms)", out, structure_view_->atoms.size());
}

void SimManager::write_rb_traj_frame(size_t step) {
	if (!rigid_body_manager_ || rigid_body_manager_->size() == 0) {
		return;
	}

	const idx_t count = rigid_body_manager_->size();
	gather_rigid_body_data();
	const HostRigidBodyData& rb = sys_state_.get_global_rigid_bodies();

	if (!rb_traj_file_.is_open()) {
		const std::string path = sys_.get_output_name() + ".0.rb-traj";
		rb_traj_file_.open(path);
		if (!rb_traj_file_) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"SimManager: could not open rigid-body trajectory file '%s'",
							path.c_str());
		}
		rb_traj_file_ << "# RigidBody trajectory file\n";
		rb_traj_file_ << "#$LABELS step RigidBodyKey"
						 " posX  posY  posZ"
						 " rotXX rotXY rotXZ"
						 " rotYX rotYY rotYZ"
						 " rotZX rotZY rotZZ"
						 " velX  velY  velZ"
						 " angVelX angVelY angVelZ\n";
		LOGINFO("SimManager: writing rigid-body trajectory to '{}'", path);
	}

	const auto& types = sys_.get_rigid_body_types();
	std::unordered_map<int, int> seen_per_type;

	for (idx_t i = 0; i < count; ++i) {
		const int type_id = rb.type_id[i];
		const int instance = seen_per_type[type_id]++;
		const std::string type_name = (type_id >= 0 && static_cast<size_t>(type_id) < types.size())
										  ? types[type_id].name
										  : std::string("RB");

		const Vector3& p = rb.position[i];
		const Matrix3& o = rb.orientation[i];
		const Vector3& mom = rb.momentum[i];
		const Vector3& ang = rb.angular_momentum[i];

		rb_traj_file_ << std::setprecision(6) << step << " " << type_name << "#" << instance << " "
					  << p.x << " " << p.y << " " << p.z;
		rb_traj_file_ << std::setprecision(10) << " " << o.ex().x << " " << o.ey().x << " "
					  << o.ez().x << " " << o.ex().y << " " << o.ey().y << " " << o.ez().y << " "
					  << o.ex().z << " " << o.ey().z << " " << o.ez().z << " " << mom.x << " "
					  << mom.y << " " << mom.z << " " << ang.x << " " << ang.y << " " << ang.z
					  << "\n";
	}
	rb_traj_file_.flush();
}

void SimManager::write_momentum_dcd_frame(size_t step) {
	(void)step;
	if (!sys_state_.is_state_synced()) {
		return;
	}
	const auto& momentum = sys_state_.get_global_momentum();

	const auto& periodicity = sys_.get_boundary_conditions().get_periodicity();
	const bool with_unitcell = periodicity[0] || periodicity[1] || periodicity[2];

	if (!momentum_dcd_header_written_) {
		const int nsavc = std::max(1, static_cast<int>(sys_.get_output_period()));
		momentum_dcd_writer_->writeHeader(static_cast<int>(momentum.size()),
										  1,
										  nsavc,
										  nsavc,
										  0,
										  sys_.get_timestep(),
										  with_unitcell);
		momentum_dcd_header_written_ = true;
	}

	if (with_unitcell) {
		const Vector3 box = sys_.get_boundary_conditions().get_box_size();
		const std::vector<double> unitcell{box.x, 0.0, box.y, 0.0, 0.0, box.z};
		momentum_dcd_writer_->writeStep(momentum, unitcell);
	} else {
		momentum_dcd_writer_->writeStep(momentum);
	}
}

//================================================================================
// Progress and Performance Reporting
//================================================================================

void SimManager::report_progress(size_t current_step, size_t total_steps, size_t report_period) {
	wkf_timer_stop(timerP_.timer);
	const float interval_elapsed = static_cast<float>(wkf_timer_time(timerP_.timer));
	wkf_timer_start(timerP_.timer);

	const float percent =
		(100.0f * static_cast<float>(current_step)) / static_cast<float>(total_steps);
	const float ms_per_step = interval_elapsed * 1000.0f / static_cast<float>(report_period);
	const int num_replicas = 1; // TODO: expose replicas from SimSystem config
	const float ns_per_day =
		static_cast<float>(num_replicas) * sys_.get_timestep() / ms_per_step * 86400000.0f;

	std::printf("\rStep %zu [%.2f%% complete | %.3f ms/step | %.3f ns/day]",
				current_step,
				percent,
				ms_per_step,
				ns_per_day);
	std::fflush(stdout);
}

void SimManager::report_performance(float elapsed_time, size_t total_steps) {
	const float steps_per_second = static_cast<float>(total_steps) / elapsed_time;
	const float ms_per_step = elapsed_time * 1000.0f / static_cast<float>(total_steps);
	const float io_time = wkf_timer_time(timerS_.timer);
	const float energy_time = wkf_timer_time(timerE_.timer);
	const float compute_time = elapsed_time - io_time - energy_time;
	std::cout << "=========================================" << std::endl;
	std::cout << "SimManager: Performance Summary: " << std::endl;
	std::cout << "  Total time:        " << elapsed_time << " s" << std::endl;
	std::cout << "  Compute time:      " << compute_time << " s ("
			  << compute_time / elapsed_time * 100 << "%)" << std::endl;
	std::cout << "  I/O time:          " << io_time << " s (" << io_time / elapsed_time * 100
			  << "%)" << std::endl;
	std::cout << "  Energy time:       " << energy_time << " s ("
			  << energy_time / elapsed_time * 100 << "%)" << std::endl;
	std::cout << "  ms/Step:      " << steps_per_second << std::endl;
	std::cout << "  ns/day (est):      "
			  << (steps_per_second * sys_.get_timestep() * 86400.0f) / 1e6f << std::endl;
	std::cout << "=========================================" << std::endl;
}

//================================================================================
// IMD Handling
//================================================================================

void SimManager::handle_imd_commands() {
	// TODO: Implement IMD command handling when needed
	// Check for incoming IMD commands
	// Update forces/positions based on user interaction
}

//================================================================================
// Initial Conditions
//================================================================================
/**
 * @brief
 *
 * @param positions
 * @param types
 */
void SimManager::generate_initial_particles(std::vector<Vector3>& positions,
											std::vector<int>& types) {
	const Vector3 box_size = sys_.get_box_size();

	const size_t num_particles = sys_state_.get_num_particles();

	positions.reserve(num_particles);
	types.reserve(num_particles);

	for (size_t i = 0; i < num_particles; ++i) {
		positions.emplace_back(box_size.x * (float)rand() / float(RAND_MAX),
							   box_size.y * (float)rand() / float(RAND_MAX),
							   box_size.z * (float)rand() / float(RAND_MAX));
		types.push_back(0); // All type 0 for now
	}

	LOGINFO("SimManager: Generated {} particles", num_particles);
}

/**
 * @brief Generate initial particle momentum and types according to Boltzmann distribution
 * @todo Make sure this is correct
 * @param v_com
 * @note V1 Configuration::Boltzmann(const Vector3& v_com, int N)
 */
void SimManager::generate_initial_momentum(const Vector3& v_com) {
	const Temperature& temperature = sys_.get_temperature_struct();
	float kT = 1.0f;
	if (temperature.format == Temperature::Format::Grid) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Grid temperature not supported for initial momentum generation");
	} else {
		kT = temperature.kT; // Fix: don't redeclare, just assign
	}

	const size_t num_particles = sys_state_.get_num_particles();
	const auto& particle_types = sys_.get_particle_types();
	std::vector<Vector3> momentum(num_particles);

	// Constants for unit conversion
	// SQRT_CAL_TO_JOULE = 2.046167337e4 (from Constants.h)

	// Initialize random number generator for host-side generation
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<double> gaussian(0.0, 1.0);

	// Generate momenta from Maxwell-Boltzmann distribution
	// p = sqrt(kT * m) * random_gaussian
	Vector3 total_momentum(0.0, 0.0, 0.0);

	for (size_t i = 0; i < num_particles; ++i) {
		int typ = particle_types[i].id;
		double M = particle_types[typ].mass;
		double sigma = sqrt(kT * M) * constants::SQRT_CAL_TO_JOULE;

		Vector3 tmp(gaussian(gen) * sigma, gaussian(gen) * sigma, gaussian(gen) * sigma);

		momentum[i] = tmp;
		total_momentum += tmp;
	}

	// Remove center of mass momentum to ensure zero net momentum
	if (num_particles > 1) {
		Vector3 p_com = total_momentum / static_cast<double>(num_particles);
		for (size_t i = 0; i < num_particles; ++i) {
			int typ = particle_types[i].id;
			double M = particle_types[typ].mass;
			momentum[i] = momentum[i] - p_com + M * v_com;
		}
	}

	LOGINFO("SimManager: Generated initial momenta for {} particles at kT={}", num_particles, kT);
}

//================================================================================
// Energy Output
//================================================================================

void SimManager::write_energy_output(size_t step) {
	(void)step;
	gather_particle_data_from_patches(/*need_energy=*/true);

	const auto& particles = sys_state_.get_global_particles();
	const size_t n = particles.size();

	double potential_energy = 0.0;
	for (float e : particles.energy) {
		potential_energy += e;
	}

	// Kinetic energy: momenta are generated/integrated in units scaled by
	// SQRT_CAL_TO_JOULE (see generate_initial_momentum), so 0.5*p^2/m must be
	// divided by SQRT_CAL_TO_JOULE^2 to recover kcal/mol before expressing it
	// as a multiple of kT - matching legacy ARBD's energy.dat convention.
	const auto& particle_types = sys_.get_particle_types();
	const Temperature& temperature = sys_.get_temperature_struct();
	const float kT = temperature.kT;
	double kinetic_energy_kcal = 0.0;
	for (size_t i = 0; i < n; ++i) {
		const int typ = particles.type_id[i];
		const double mass = particle_types[typ].mass;
		kinetic_energy_kcal += 0.5 * particles.mom[i].length2() / mass;
	}
	kinetic_energy_kcal /= (constants::SQRT_CAL_TO_JOULE * constants::SQRT_CAL_TO_JOULE);
	const double kinetic_energy_kT = kT > 0.0f ? kinetic_energy_kcal / kT : 0.0;

	if (!energy_file_.is_open()) {
		energy_file_.open(sys_.get_output_name() + ".energy.dat");
	}
	if (energy_file_.is_open()) {
		energy_file_ << "Kinetic Energy: " << kinetic_energy_kT << " (kT) " << std::endl;
		energy_file_ << "Potential Energy: " << potential_energy << " (kcal/mol) " << std::endl;
	} else {
		LOGWARN("SimManager: Failed to open '{}' for energy output",
				sys_.get_output_name() + ".energy.dat");
	}

	if (has_rigid_bodies_) {
		// TODO: rigid-body kinetic/potential energy is not yet computed by
		// SimManager; write zeros so downstream tooling still gets the file,
		// matching legacy ARBD's rb_energy.dat format.
		if (!rb_energy_file_.is_open()) {
			rb_energy_file_.open(sys_.get_output_name() + ".rb_energy.dat");
		}
		if (rb_energy_file_.is_open()) {
			rb_energy_file_ << "Kinetic Energy 0 (kT)" << std::endl;
			rb_energy_file_ << "Potential Energy 0 (kcal/mol)" << std::endl;
		} else {
			LOGWARN("SimManager: Failed to open '{}' for rigid body energy output",
					sys_.get_output_name() + ".rb_energy.dat");
		}
	}

	write_restart_files();
}

//================================================================================
// Restart Files
//================================================================================

void SimManager::wait_for_pending_restart_write() {
	if (pending_restart_write_.valid()) {
		pending_restart_write_.get();
	}
}

void SimManager::write_restart_files() {
	wait_for_pending_restart_write();

	const auto& particles = sys_state_.get_global_particles();
	std::vector<int> type_id = particles.type_id;
	std::vector<Vector3> pos = particles.pos;
	std::vector<Vector3> mom = has_momentum_output_ ? particles.mom : std::vector<Vector3>{};
	const std::string output_name = sys_.get_output_name();
	const bool write_momentum = has_momentum_output_;

	pending_restart_write_ = std::async(
		std::launch::async,
		[type_id = std::move(type_id),
		 pos = std::move(pos),
		 mom = std::move(mom),
		 output_name,
		 write_momentum]() {
			const std::string restart_filename = output_name + ".restart";
			FILE* out = std::fopen(restart_filename.c_str(), "w");
			if (out) {
				std::string buf;
				buf.reserve(pos.size() * 32);
				for (size_t i = 0; i < pos.size(); ++i) {
					append_restart_line(buf, type_id[i], pos[i]);
				}
				std::fwrite(buf.data(), 1, buf.size(), out);
				std::fclose(out);
			} else {
				LOGWARN("SimManager: Failed to open '{}' for restart output", restart_filename);
			}

			// Momentum restart only applies to Langevin dynamics
			// The "0" in the filename mirrors legacy ARBD's on-disk naming.
			if (write_momentum) {
				const std::string momentum_restart_filename = output_name + ".0.momentum.restart";
				FILE* mout = std::fopen(momentum_restart_filename.c_str(), "w");
				if (mout) {
					std::string buf;
					buf.reserve(mom.size() * 32);
					for (size_t i = 0; i < mom.size(); ++i) {
						append_restart_line(buf, type_id[i], mom[i]);
					}
					std::fwrite(buf.data(), 1, buf.size(), mout);
					std::fclose(mout);
				} else {
					LOGWARN("SimManager: Failed to open '{}' for momentum restart output",
							momentum_restart_filename);
				}
			}
		});
}

void SimManager::write_final_restart() {
	gather_particle_data_from_patches();
	write_restart_files();
	wait_for_pending_restart_write();
	LOGINFO("SimManager: Wrote final restart files for '{}'", sys_.get_output_name());
}

//================================================================================
// Particle Reactions
//================================================================================
/*
void SimManager::perform_reactions() {
	Patch& patch = sys_.get_patch_manager()->get_local_patch();

	// 1. Run Reaction Kernel
	//    Sets FLAG_DEAD on some particles.
	//    Creates new particles in a temporary "Birth Buffer".

	// 2. Remove Dead Particles
	//    Compacts DeviceParticle array.
	//    Generates 'permutation_map'.
	auto perm_map = patch.compact_particles();

	// 3. Update Interactions (Fix Indices)
	if (patch.has_topology_changes()) {
		for (auto& interaction : interactions_) {
			interaction->update_topology(perm_map);
		}
	}

	// 4. Add New Particles
	//    Appends from "Birth Buffer" to end of DeviceParticle.
	//    (No index shifting for existing particles, so safe).
	patch.append_from_buffer(birth_buffer);

	// 5. Rebuild Neighbor Lists
	//    Mandatory after moving particles.
	neighbor_list_.force_rebuild();
}
*/
} // namespace ARBD
