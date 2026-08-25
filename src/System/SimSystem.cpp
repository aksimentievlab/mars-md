#include "SimSystem.h"
#include "System/PatchDecomposer/ZOrder.h"
#include "System/PatchManager.h"
#include "System/SystemState.h"

namespace MARS {

void SimSystem::validate_physical_parameters() const {

	if (temperature_.value <= mars_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Temperature must be greater than 0");
	}

	if (cutoff_ <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Cutoff must be greater than 0");
	}
	if (sim_box_.get_box_size().x <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length x must be positive (got {})",
						sim_box_.get_box_size().x);
	}
	if (sim_box_.get_box_size().y <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length y must be positive (got {})",
						sim_box_.get_box_size().y);
	}
	if (sim_box_.get_box_size().z <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length z must be positive (got {})",
						sim_box_.get_box_size().z);
	}
};

void SimSystem::validate_method_parameters() const {

	if (steps_.timestep <= mars_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Timestep must be positive (got {})",
						steps_.timestep);
	}
	if (steps_.steps <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Number of steps must be positive (got {})",
						steps_.steps);
	}
	if (neighbor_list_rebuild_period <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Neighbor list rebuild period must be positive (got {})",
						neighbor_list_rebuild_period);
	}
};

void SimSystem::validate_output_parameters() const {

	if (output_period_ <= mars_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Output period must be positive (got {})",
						output_period_);
	}
	if (energy_output_period_ <= mars_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Energy output period must be positive (got {})",
						energy_output_period_);
	}
	if (output_name.empty()) {
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Output name cannot be empty");
	}
};

//================================================================================
// Factory Function
//================================================================================

std::unique_ptr<PatchDecomposer> create_patch_decomposer(DecomposerType type) {
	switch (type) {
	case DecomposerType::Spatial:
		return std::make_unique<SpatialPatchDecomposer>();

	case DecomposerType::RecursiveBisection:
		return std::make_unique<RecursiveBisectionPatchDecomposer>();

	case DecomposerType::Geometric:
		return std::make_unique<GeometricPatchDecomposer>();

	case DecomposerType::ZOrder:
		return std::make_unique<ZOrderDecomposer>();

	default:
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Unsupported decomposer type");
	}
}

//================================================================================
// Single Resource Optimization
//================================================================================

void SimSystem::create_single_patch_manager(SystemState& state) {
	// Create simplified patch manager for single resource systems
	auto patch_manager = std::make_unique<PatchManager>(*this);

	// Get cutoff parameters
	const Length cutoff = get_cutoff();
	const Length pairlist_cutoff = get_pairlist_cutoff();
	const Length halo_width = std::max(cutoff, pairlist_cutoff);

	// Create a minimal decomposition plan for single patch
	DecompositionPlan plan;
	plan.grid_dimensions = {1, 1, 1}; // Single patch
	plan.periodicity = {sim_box_.get_periodicity()[0],
						sim_box_.get_periodicity()[1],
						sim_box_.get_periodicity()[2]};

	// Single patch covers entire system
	Vector3 system_size = get_box_size();
	plan.patch_min_bounds.push_back(Vector3(0.0f, 0.0f, 0.0f));
	plan.patch_max_bounds.push_back(system_size);
	plan.system_min = Vector3(0.0f, 0.0f, 0.0f);
	plan.system_max = system_size;

	// Assign the single resource
	plan.patch_resources.push_back(resources_[0]);

	// No particle pre-assignment needed for single patch
	plan.patch_particle_indices.resize(1);
	plan.system_box = sim_box_;
	plan.set_periodic_box();

	// Initialize statistics
	plan.statistics.total_particles = state.get_num_particles();
	plan.statistics.particles_per_patch = {state.get_num_particles()};
	plan.statistics.load_balance_factor = 1.0f;	 // Perfect balance with single patch
	plan.statistics.communication_volume = 0.0f; // No communication needed

	// Initialize PatchManager with simplified plan
	patch_manager->initialize_from_plan(plan, estimated_particles_);

	// Store the patch manager
	patch_manager_ = std::move(patch_manager);

	LOGINFO("SimSystem: Single patch created for {} particles on single resource",
			state.get_num_particles());
}

} // namespace MARS
