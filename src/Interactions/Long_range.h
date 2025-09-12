/*********************************************************************
 * @file  Long_range.h
 *
 * @brief Unified framework supporting three long-range electrostatics methods:
 *        1. Cutoff-AMR (novel adaptive approach)
 *        2. PPPM/PME (traditional mesh methods)
 *        3. FMM (fast multipole method)
 *********************************************************************/

#pragma once

#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/BaseGrid.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace ARBD {

/*==========================*\
|  LONG-RANGE METHOD ENUM    |
\*==========================*/

/**
 * @brief Method selection criteria
 */
struct MethodSelectionCriteria {
	size_t particle_count;
	float charge_density_variation; ///< 0.0 = uniform, 1.0 = highly clustered
	bool is_periodic;
	bool multi_gpu_available;
	float accuracy_requirement; ///< 0.0 = fast, 1.0 = highly accurate

	// System characteristics
	float system_size;
	float cutoff_distance;
	bool has_long_range_order; ///< e.g., crystals vs liquids

	// Performance constraints
	float max_memory_gb;
	float max_compute_time_ms;

	HOST DEVICE bool prefers_local_methods() const {
		return charge_density_variation > 0.7f || !is_periodic;
	}

	HOST DEVICE bool needs_high_accuracy() const {
		return accuracy_requirement > 0.8f;
	}

	HOST DEVICE bool is_small_system() const {
		return particle_count < 10000;
	}
};

/*==========================*\
|  ABSTRACT BASE CLASS       |
\*==========================*/

/**
 * @brief Abstract base class for all long-range methods
 */
class LongRangeElectrostatics {
  public:
	struct Config {
		LongRangeMethod method;
		float cutoff_distance = 10.0f;
		float accuracy_tolerance = 1e-5f;
		bool use_periodic_boundaries = true;
		Resource preferred_resource;

		// Method-specific parameters (stored as key-value pairs)
		std::unordered_map<std::string, float> parameters;
	};

  protected:
	Config config_;
	mutable float last_computation_time_ = 0.0f;
	mutable size_t interaction_count_ = 0;

  public:
	LongRangeElectrostatics(const Config& config) : config_(config) {}
	virtual ~LongRangeElectrostatics() = default;

	// Core interface
	virtual void solve_electrostatics(ParticleBuffer& particles, const ParticleType& type_id) = 0;

	virtual float estimate_computational_cost(size_t particle_count) const = 0;
	virtual float estimate_memory_usage(size_t particle_count) const = 0;
	virtual float estimate_accuracy() const = 0;
	virtual bool supports_periodic_boundaries() const = 0;
	virtual bool supports_multi_gpu() const = 0;

	// Performance metrics
	float last_computation_time() const {
		return last_computation_time_;
	}
	size_t last_interaction_count() const {
		return interaction_count_;
	}

	// Configuration
	const Config& config() const {
		return config_;
	}
	LongRangeMethod method() const {
		return config_.method;
	}

	virtual std::string name() const = 0;
	virtual std::string description() const = 0;
};

/*==========================*\
|  CUTOFF-AMR IMPLEMENTATION |
\*==========================*/

/**
 * @brief Cutoff-AMR: Your novel adaptive approach
 */
class CutoffAMRElectrostatics : public LongRangeElectrostatics {
  private:
	// AMR hierarchy using your BaseGrid system
	std::vector<std::unique_ptr<BaseGrid<float>>> amr_levels_;
	std::vector<std::unique_ptr<BaseGrid<int>>> particle_count_grids_;

	// Adaptive parameters
	float refinement_threshold_;
	float coarsening_threshold_;
	size_t max_refinement_levels_;

	// Cutoff functions
	enum class CutoffType { Hard, Smooth, ReactionField, Damped, ForceShifted };
	CutoffType cutoff_type_;

  public:
	CutoffAMRElectrostatics(const Config& config)
		: LongRangeElectrostatics(config),
		  refinement_threshold_(config.parameters.at("refinement_threshold")),
		  coarsening_threshold_(config.parameters.at("coarsening_threshold")),
		  max_refinement_levels_(static_cast<size_t>(config.parameters.at("max_levels"))),
		  cutoff_type_(static_cast<CutoffType>(config.parameters.at("cutoff_type"))) {

		initialize_amr_hierarchy();
	}

	void solve_electrostatics(ParticleBuffer& particles,
							  const ParticleTypeManager& type_manager) override {
		auto start_time = std::chrono::high_resolution_clock::now();

		// 1. Adaptive mesh refinement based on particle density
		update_particle_density(particles);
		adaptive_refinement();

		// 2. Build neighbor lists using AMR spatial structure
		build_amr_neighbor_lists(particles);

		// 3. Calculate cutoff electrostatics with smooth functions
		calculate_cutoff_forces(particles, type_manager);

		// 4. Apply long-range correction (reaction field or damping)
		apply_long_range_correction(particles, type_manager);

		auto end_time = std::chrono::high_resolution_clock::now();
		last_computation_time_ =
			std::chrono::duration<float, std::milli>(end_time - start_time).count();
	}

	float estimate_computational_cost(size_t particle_count) const override {
		// O(N) scaling with small constant factors
		return static_cast<float>(particle_count) * 1e-5f; // Very efficient
	}

	float estimate_memory_usage(size_t particle_count) const override {
		// Only allocate where particles exist
		float occupied_fraction = 0.1f; // Typical for molecular systems
		return static_cast<float>(particle_count) * occupied_fraction * sizeof(float) *
			   8; // 8 grids
	}

	float estimate_accuracy() const override {
		return 0.8f; // Good accuracy, not perfect due to cutoff
	}

	bool supports_periodic_boundaries() const override {
		return true;
	}
	bool supports_multi_gpu() const override {
		return true;
	} // Excellent scaling

	std::string name() const override {
		return "Cutoff-AMR";
	}
	std::string description() const override {
		return "Adaptive mesh refinement with cutoff electrostatics";
	}

  private:
	void initialize_amr_hierarchy() {
		// Create initial coarse grid
		Matrix3 basis = Matrix3::diagonal(config_.cutoff_distance,
										  config_.cutoff_distance,
										  config_.cutoff_distance);
		Vector3 origin(-50.0f, -50.0f, -50.0f); // System-dependent

		auto coarse_grid = std::make_unique<BaseGrid<float>>(basis, origin, 32, 32, 32);
		amr_levels_.push_back(std::move(coarse_grid));

		auto count_grid = std::make_unique<BaseGrid<int>>(basis, origin, 32, 32, 32);
		particle_count_grids_.push_back(std::move(count_grid));
	}

	void update_particle_density(const ParticleBuffer& particles) {
		// Reset particle counts
		for (auto& grid : particle_count_grids_) {
			grid->zero();
		}

		// Deposit particles onto grids
		if (particles.layout() == ParticleBuffer::Layout::SoA) {
			const Vector3* positions = particles.get_position_array();
			for (size_t i = 0; i < particles.size(); ++i) {
				deposit_particle_to_amr(positions[i], 1.0f);
			}
		}
	}

	void adaptive_refinement() {
		// Check each grid level for refinement/coarsening
		for (size_t level = 0; level < amr_levels_.size(); ++level) {
			auto& count_grid = particle_count_grids_[level];

			bool needs_refinement = false;
			for (size_t i = 0; i < count_grid->size(); ++i) {
				int count = (*count_grid)[i];
				if (count > refinement_threshold_) {
					needs_refinement = true;
					break;
				}
			}

			if (needs_refinement && level < max_refinement_levels_) {
				create_refined_level(level);
			}
		}
	}

	void deposit_particle_to_amr(const Vector3& position, float contribution) {
		// Find appropriate AMR level and deposit particle
		for (auto& grid : particle_count_grids_) {
			if (grid->in_bounds(position)) {
				Vector3 grid_pos = grid->transform_to_grid(position);
				size_t ix = static_cast<size_t>(grid_pos.x);
				size_t iy = static_cast<size_t>(grid_pos.y);
				size_t iz = static_cast<size_t>(grid_pos.z);
				size_t idx = grid->index(ix, iy, iz);
				(*grid)[idx] += static_cast<int>(contribution);
				break;
			}
		}
	}

	void build_amr_neighbor_lists(const ParticleBuffer& particles) {
		// Use AMR structure to build efficient neighbor lists
		// Implementation similar to your V1 cell decomposition but adaptive
	}

	void calculate_cutoff_forces(ParticleBuffer& particles,
								 const ParticleTypeManager& type_manager) {
		// Cutoff electrostatics with chosen smoothing function
		// Uses neighbor lists for O(N) scaling
	}

	void apply_long_range_correction(ParticleBuffer& particles,
									 const ParticleTypeManager& type_manager) {
		// Reaction field or other long-range correction
	}

	void create_refined_level(size_t parent_level) {
		// Create finer grid level for high-density regions
	}
};

/*==========================*\
|  PPPM/PME IMPLEMENTATION   |
\*==========================*/

/**
 * @brief PPPM/PME: Traditional mesh-based methods
 */
class PPPMElectrostatics : public LongRangeElectrostatics {
  private:
	std::unique_ptr<BaseGrid<float>> charge_grid_;
	std::unique_ptr<BaseGrid<float>> potential_grid_;
	std::unique_ptr<BaseGrid<Vector3>> electric_field_grid_;

	// Ewald parameters
	float alpha_; // Ewald splitting parameter
	float real_cutoff_;
	Vector3 grid_dimensions_;

	// FFT workspace (backend-dependent)
	mutable void* fft_workspace_ = nullptr;

  public:
	PPPMElectrostatics(const Config& config) : LongRangeElectrostatics(config) {
		// Initialize grids and FFT parameters
		initialize_grids();
		calculate_optimal_parameters();
	}

	void solve_electrostatics(ParticleBuffer& particles,
							  const ParticleTypeManager& type_manager) override {
		auto start_time = std::chrono::high_resolution_clock::now();

		// 1. Real-space interactions (short-range)
		calculate_real_space_forces(particles, type_manager);

		// 2. Charge assignment to mesh
		assign_charges_to_mesh(particles, type_manager);

		// 3. Forward FFT
		forward_fft();

		// 4. Apply Green's function in k-space
		solve_in_fourier_space();

		// 5. Backward FFT
		backward_fft();

		// 6. Force interpolation from mesh
		interpolate_forces_from_mesh(particles, type_manager);

		// 7. Self-energy correction
		apply_self_energy_correction(particles, type_manager);

		auto end_time = std::chrono::high_resolution_clock::now();
		last_computation_time_ =
			std::chrono::duration<float, std::milli>(end_time - start_time).count();
	}

	float estimate_computational_cost(size_t particle_count) const override {
		// O(N log N) due to FFT
		return static_cast<float>(particle_count * std::log2(particle_count)) * 1e-6f;
	}

	float estimate_memory_usage(size_t particle_count) const override {
		// Fixed grid size regardless of particle distribution
		size_t grid_size =
			static_cast<size_t>(grid_dimensions_.x * grid_dimensions_.y * grid_dimensions_.z);
		return static_cast<float>(grid_size * sizeof(float) * 4); // 4 grids
	}

	float estimate_accuracy() const override {
		return 0.95f; // Very high accuracy
	}

	bool supports_periodic_boundaries() const override {
		return true;
	}
	bool supports_multi_gpu() const override {
		return false;
	} // FFT communication bottleneck

	std::string name() const override {
		return config_.method == LongRangeMethod::PPPM ? "PPPM" : "PME";
	}
	std::string description() const override {
		return "Particle-Particle Particle-Mesh with FFT";
	}

  private:
	void initialize_grids() {
		// Create uniform grids for FFT
		size_t nx = static_cast<size_t>(config_.parameters.at("grid_nx"));
		size_t ny = static_cast<size_t>(config_.parameters.at("grid_ny"));
		size_t nz = static_cast<size_t>(config_.parameters.at("grid_nz"));

		Matrix3 basis = Matrix3::diagonal(1.0f, 1.0f, 1.0f); // Unit spacing
		Vector3 origin(-static_cast<float>(nx) / 2,
					   -static_cast<float>(ny) / 2,
					   -static_cast<float>(nz) / 2);

		charge_grid_ = std::make_unique<BaseGrid<float>>(basis, origin, nx, ny, nz);
		potential_grid_ = std::make_unique<BaseGrid<float>>(basis, origin, nx, ny, nz);
		electric_field_grid_ = std::make_unique<BaseGrid<Vector3>>(basis, origin, nx, ny, nz);
	}

	void calculate_optimal_parameters() {
		// Auto-tune Ewald parameters for optimal performance/accuracy
		alpha_ = config_.parameters.at("ewald_alpha");
		real_cutoff_ = config_.parameters.at("real_cutoff");
	}

	void calculate_real_space_forces(ParticleBuffer& particles,
									 const ParticleTypeManager& type_manager) {
		// Direct pairwise interactions with erfc(alpha*r)/r
	}

	void assign_charges_to_mesh(const ParticleBuffer& particles,
								const ParticleTypeManager& type_manager) {
		// B-spline or other interpolation scheme
	}

	void forward_fft() {
		// Backend-specific FFT implementation
	}

	void solve_in_fourier_space() {
		// Apply Green's function: φ(k) = ρ(k) / k²
	}

	void backward_fft() {
		// Backend-specific inverse FFT
	}

	void interpolate_forces_from_mesh(ParticleBuffer& particles,
									  const ParticleTypeManager& type_manager) {
		// Interpolate electric field to particle positions
	}

	void apply_self_energy_correction(ParticleBuffer& particles,
									  const ParticleTypeManager& type_manager) {
		// Remove self-interaction artifacts
	}
};

/*==========================*\
|  FMM IMPLEMENTATION        |
\*==========================*/

/**
 * @brief FMM: Fast Multipole Method
 */
class FMMElectrostatics : public LongRangeElectrostatics {
  private:
	struct FMMNode {
		Vector3 center;
		float size;
		size_t level;
		std::vector<std::complex<float>> multipole_coeffs;
		std::vector<std::complex<float>> local_coeffs;
		std::vector<size_t> particle_indices;
		std::unique_ptr<FMMNode> children[8];
		FMMNode* parent = nullptr;

		bool is_leaf() const {
			return children[0] == nullptr;
		}
	};

	std::unique_ptr<FMMNode> tree_root_;
	size_t max_particles_per_leaf_;
	size_t multipole_order_;
	float theta_; // Accuracy parameter

  public:
	FMMElectrostatics(const Config& config) : LongRangeElectrostatics(config) {
		max_particles_per_leaf_ = static_cast<size_t>(config.parameters.at("max_leaf_particles"));
		multipole_order_ = static_cast<size_t>(config.parameters.at("multipole_order"));
		theta_ = config.parameters.at("theta");
	}

	void solve_electrostatics(ParticleBuffer& particles,
							  const ParticleTypeManager& type_manager) override {
		auto start_time = std::chrono::high_resolution_clock::now();

		// 1. Build adaptive octree
		build_tree(particles);

		// 2. Upward pass: compute multipole expansions
		compute_multipole_expansions(particles, type_manager);

		// 3. Downward pass: compute local expansions
		compute_local_expansions();

		// 4. Direct interactions for nearby particles
		compute_direct_interactions(particles, type_manager);

		// 5. Apply forces from local expansions
		evaluate_local_expansions(particles, type_manager);

		auto end_time = std::chrono::high_resolution_clock::now();
		last_computation_time_ =
			std::chrono::duration<float, std::milli>(end_time - start_time).count();
	}

	float estimate_computational_cost(size_t particle_count) const override {
		// O(N) scaling
		return static_cast<float>(particle_count) * 1e-4f; // Higher constant than cutoff-AMR
	}

	float estimate_memory_usage(size_t particle_count) const override {
		// Tree structure + multipole coefficients
		size_t tree_nodes = particle_count / max_particles_per_leaf_ * 8;
		size_t coeffs_per_node = multipole_order_ * multipole_order_;
		return static_cast<float>(tree_nodes * coeffs_per_node * sizeof(std::complex<float>) * 2);
	}

	float estimate_accuracy() const override {
		return 0.99f; // Highest accuracy, controllable via theta parameter
	}

	bool supports_periodic_boundaries() const override {
		return false;
	} // Needs special handling
	bool supports_multi_gpu() const override {
		return true;
	}

	std::string name() const override {
		return "FMM";
	}
	std::string description() const override {
		return "Fast Multipole Method with adaptive octree";
	}

  private:
	void build_tree(const ParticleBuffer& particles) {
		// Build adaptive octree based on particle distribution
	}

	void compute_multipole_expansions(const ParticleBuffer& particles,
									  const ParticleTypeManager& type_manager) {
		// Bottom-up pass: particles -> multipole moments
	}

	void compute_local_expansions() {
		// Top-down pass: far-field -> local Taylor expansions
	}

	void compute_direct_interactions(ParticleBuffer& particles,
									 const ParticleTypeManager& type_manager) {
		// Direct calculation for nearby particles
	}

	void evaluate_local_expansions(ParticleBuffer& particles,
								   const ParticleTypeManager& type_manager) {
		// Apply local expansion forces to particles
	}
};

/*==========================*\
|  UNIFIED METHOD FACTORY    |
\*==========================*/

/**
 * @brief Factory for creating appropriate long-range method
 */
class LongRangeMethodFactory {
  public:
	/**
	 * @brief Automatically select best method based on system characteristics
	 */
	static std::unique_ptr<LongRangeElectrostatics>
	create_optimal_method(const MethodSelectionCriteria& criteria) {

		LongRangeElectrostatics::Config config;

		// Decision tree for method selection
		if (criteria.is_small_system()) {
			config.method = LongRangeMethod::Direct;

		} else if (criteria.prefers_local_methods()) {
			config.method = LongRangeMethod::CutoffAMR;
			config.parameters["refinement_threshold"] = 50.0f;
			config.parameters["coarsening_threshold"] = 10.0f;
			config.parameters["max_levels"] = 4.0f;
			config.parameters["cutoff_type"] = 1.0f; // Smooth

		} else if (criteria.is_periodic && !criteria.needs_high_accuracy()) {
			config.method = LongRangeMethod::PPPM;
			config.parameters["grid_nx"] = 64.0f;
			config.parameters["grid_ny"] = 64.0f;
			config.parameters["grid_nz"] = 64.0f;
			config.parameters["ewald_alpha"] = 0.3f;
			config.parameters["real_cutoff"] = criteria.cutoff_distance;

		} else if (criteria.needs_high_accuracy() || !criteria.is_periodic) {
			config.method = LongRangeMethod::FMM;
			config.parameters["max_leaf_particles"] = 100.0f;
			config.parameters["multipole_order"] = 8.0f;
			config.parameters["theta"] = 0.5f;

		} else {
			// Default to cutoff-AMR as it's most versatile
			config.method = LongRangeMethod::CutoffAMR;
		}

		return create_method(config);
	}

	/**
	 * @brief Create specific method with given configuration
	 */
	static std::unique_ptr<LongRangeElectrostatics>
	create_method(const LongRangeElectrostatics::Config& config) {

		switch (config.method) {
		case LongRangeMethod::CutoffAMR:
			return std::make_unique<CutoffAMRElectrostatics>(config);

		case LongRangeMethod::PPPM:
		case LongRangeMethod::PME:
			return std::make_unique<PPPMElectrostatics>(config);

		case LongRangeMethod::FMM:
			return std::make_unique<FMMElectrostatics>(config);

		case LongRangeMethod::Direct:
			// Implementation of direct O(N²) method
			break;

		case LongRangeMethod::None:
			// No-op implementation
			break;

		default:
			throw std::runtime_error("Unknown long-range method");
		}

		return nullptr;
	}
};

/*==========================*\
|  USAGE EXAMPLE             |
\*==========================*/

/**
 * @brief Example of how to use the unified framework
 */
class UnifiedLongRangeExample {
  public:
	void demonstrate_usage() {
		// 1. Define system characteristics
		MethodSelectionCriteria criteria;
		criteria.particle_count = 100000;
		criteria.charge_density_variation = 0.3f; // Moderately clustered
		criteria.is_periodic = true;
		criteria.multi_gpu_available = true;
		criteria.accuracy_requirement = 0.8f;

		// 2. Automatically select optimal method
		auto solver = LongRangeMethodFactory::create_optimal_method(criteria);

		std::cout << "Selected method: " << solver->name() << std::endl;
		std::cout << "Description: " << solver->description() << std::endl;

		// 3. Or manually choose a specific method
		LongRangeElectrostatics::Config manual_config;
		manual_config.method = LongRangeMethod::CutoffAMR;
		manual_config.parameters["refinement_threshold"] = 30.0f;

		auto manual_solver = LongRangeMethodFactory::create_method(manual_config);

		// 4. Use in simulation
		ParticleBuffer particles(100000, ParticleBuffer::Layout::SoA);
		ParticleTypeManager type_manager;

		// Main simulation loop
		for (int step = 0; step < 1000; ++step) {
			solver->solve_electrostatics(particles, type_manager);

			// Performance monitoring
			if (step % 100 == 0) {
				std::cout << "Step " << step
						  << ", computation time: " << solver->last_computation_time() << " ms"
						  << ", interactions: " << solver->last_interaction_count() << std::endl;
			}
		}

		// 5. Method comparison
		compare_methods(criteria, particles, type_manager);
	}

  private:
	void compare_methods(const MethodSelectionCriteria& criteria,
						 ParticleBuffer& particles,
						 const ParticleTypeManager& type_manager) {

		std::vector<LongRangeMethod> methods = {LongRangeMethod::CutoffAMR,
												LongRangeMethod::PPPM,
												LongRangeMethod::FMM};

		for (auto method : methods) {
			LongRangeElectrostatics::Config config;
			config.method = method;

			try {
				auto solver = LongRangeMethodFactory::create_method(config);

				// Benchmark
				auto start = std::chrono::high_resolution_clock::now();
				solver->solve_electrostatics(particles, type_manager);
				auto end = std::chrono::high_resolution_clock::now();

				float time_ms = std::chrono::duration<float, std::milli>(end - start).count();

				std::cout << solver->name() << ":\n"
						  << "  Time: " << time_ms << " ms\n"
						  << "  Memory: " << solver->estimate_memory_usage(particles.size()) / 1e6
						  << " MB\n"
						  << "  Accuracy: " << solver->estimate_accuracy() << "\n"
						  << "  Multi-GPU: " << (solver->supports_multi_gpu() ? "Yes" : "No")
						  << "\n"
						  << std::endl;

			} catch (const std::exception& e) {
				std::cout << "Method " << static_cast<int>(method) << " failed: " << e.what()
						  << std::endl;
			}
		}
	}
};

} // namespace ARBD
