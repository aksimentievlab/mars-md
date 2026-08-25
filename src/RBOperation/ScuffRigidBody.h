#pragma once

#include "libscuff.h"

#include "MARSException.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Constants.h"
#include "Header.h"
#include "Objects/DeviceRigidBody.h"
#include "Objects/RigidBodyProperties.h"
#include "Types/Types.h"

#include <complex>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace MARS {

/**
 * @todo Change the wrong column major matrix to row major.
 * @brief Host-side SoA for the plasmonic subset of the rigid bodies.
 *
 * Entry @c i couples slot @c plasmonic_particle_id[i] of @c global_rigidBody_data
 * (which is also its device SoA slot) to SCUFF surface index @c surface_tag[i].
 * Only the subset is stored here; the arrays are independent of the global
 * rigid-body data and are never resized through it.
 */
struct ScuffRigidBody {
	std::vector<int> plasmonic_particle_id;
	std::vector<int> surface_tag;
	std::vector<Vector3> position;
	std::vector<Matrix3> orientation;
	std::vector<Vector3> plasmonic_force;
	std::vector<Vector3> plasmonic_torque;
	HostRigidBodyData& global_rigidBody_data;

	explicit ScuffRigidBody(HostRigidBodyData& global) : global_rigidBody_data(global) {}

	idx_t size() const {
		return static_cast<idx_t>(plasmonic_particle_id.size());
	}

	void resize(idx_t n) {
		plasmonic_particle_id.resize(n);
		surface_tag.resize(n);
		position.resize(n);
		orientation.resize(n);
		plasmonic_force.resize(n, Vector3(0.0f));
		plasmonic_torque.resize(n, Vector3(0.0f));
	}

	void clear() {
		plasmonic_particle_id.clear();
		surface_tag.clear();
		position.clear();
		orientation.clear();
		plasmonic_force.clear();
		plasmonic_torque.clear();
	}

	void reserve(idx_t n) {
		plasmonic_particle_id.reserve(n);
		surface_tag.reserve(n);
		position.reserve(n);
		orientation.reserve(n);
		plasmonic_force.reserve(n);
		plasmonic_torque.reserve(n);
	}

	/// Zeroes the force/torque outputs while keeping the entry count intact.
	void clear_force() {
		plasmonic_force.assign(plasmonic_force.size(), Vector3(0.0f));
		plasmonic_torque.assign(plasmonic_torque.size(), Vector3(0.0f));
	}

	/**
	 * @param rigid_body_index Slot in @c global_rigidBody_data / the device SoA.
	 * @param scuff_surface_index Surface index in the loaded .scuffgeo geometry.
	 */
	void add(int rigid_body_index, int scuff_surface_index) {
		plasmonic_particle_id.push_back(rigid_body_index);
		surface_tag.push_back(scuff_surface_index);
		position.emplace_back(0.0f);
		orientation.emplace_back();
		plasmonic_force.emplace_back(0.0f);
		plasmonic_torque.emplace_back(0.0f);
	}

	/**
	 * @brief Pulls the current pose of every plasmonic body out of the global SoA.
	 * @note Reads whatever the last gather left; the caller is responsible for
	 *       refreshing it (SimManager::gather_rigid_body_data).
	 */
	void update_from_bd() {
		const idx_t n = size();
		if (position.size() < n) {
			resize(n);
		}
		for (idx_t i = 0; i < n; ++i) {
			const idx_t src = static_cast<idx_t>(plasmonic_particle_id[i]);
			if (src >= global_rigidBody_data.size()) {
				MARS_Exception(ExceptionType::ValueError,
							   "ScuffRigidBody entry %zu references rigid body %zu of %zu",
							   i,
							   src,
							   global_rigidBody_data.size());
			}
			position[i] = global_rigidBody_data.position[src];
			orientation[i] = global_rigidBody_data.orientation[src];
		}
	}
};

/**
 * @brief Wraps one SCUFF-EM BEM solve per call into MARS force/torque units.
 *
 * The geometry file supplies the mesh and materials; each surface must be
 * meshed about its own origin and left unplaced, because the transform written
 * here sets the absolute pose rather than a displacement from a previously
 * placed position. check_geo_format() enforces that at load time.
 */
class ScuffForceCalculator {
  public:
	/**
	 * @param geo_file Path to the .scuffgeo geometry.
	 * @param omega Angular frequency in SCUFF units (3e14 rad/sec = c / 1 micron).
	 */
	ScuffForceCalculator(const std::string& geo_file, double omega)
		: geometry_(geo_file.c_str()), geo_file_(geo_file), omega_(omega) {
		check_geo_format();
		incident_field_ = std::make_unique<PlaneWave>();
		bem_matrix_.reset(geometry_.AllocateBEMMatrix(false, false));
		rhs_vector_.reset(geometry_.AllocateRHSVector(false));
		kn_vector_.reset(geometry_.AllocateRHSVector(false));
	}

	ScuffForceCalculator(const ScuffForceCalculator&) = delete;
	ScuffForceCalculator& operator=(const ScuffForceCalculator&) = delete;
	~ScuffForceCalculator() = default;

	/**
	 * @brief Rejects a geometry whose OBJECT/SURFACE blocks place their own mesh.
	 *
	 * A DISPLACED/ROTATED line inside OBJECT...ENDOBJECT is parsed into
	 * RWGSurface::OTGT, applied to the vertices at birth and never un-applied
	 * (RWGSurface.cc:427). UnTransform() rewinds only RWGSurface::GT, so the
	 * pose written by update_all_surface_transforms() would compose on top of
	 * that placement instead of replacing it, silently offsetting every body.
	 *
	 * @throws Exception listing every offending surface.
	 */
	void check_geo_format() const {
		std::string placed;
		for (int ns = 0; ns < geometry_.NumSurfaces; ++ns) {
			const scuff::RWGSurface* surface = geometry_.Surfaces[ns];
			if (surface->OTGT == nullptr) {
				continue;
			}
			if (!placed.empty()) {
				placed += ", ";
			}
			placed += surface->Label ? surface->Label : "<unlabeled>";
		}
		if (!placed.empty()) {
			MARS_Exception(ExceptionType::ValueError,
						   "scuff geometry '%s' places surfaces itself: %s. Remove the "
						   "DISPLACED/ROTATED lines from those OBJECT/SURFACE blocks - the "
						   "rigid body pose supplies the absolute placement.",
						   geo_file_,
						   placed);
		}
	}

	int num_surfaces() const {
		return geometry_.NumSurfaces;
	}

	void set_omega(double omega) {
		omega_ = omega;
	}

	/// Overrides the default +x-polarized wave travelling along +z.
	void set_plane_wave(const std::complex<double> E0[3], const double n_hat[3]) {
		auto pw = std::make_unique<PlaneWave>();
		cdouble e0[3] = {E0[0], E0[1], E0[2]};
		double n[3] = {n_hat[0], n_hat[1], n_hat[2]};
		pw->SetE0(e0);
		pw->SetnHat(n);
		incident_field_ = std::move(pw);
	}

	/// Returns every surface to the pose it had when the geometry was read.
	void reset_transforms() {
		geometry_.UnTransform();
	}

	/**
	 * @brief Places each tagged surface at its rigid body's current pose.
	 *
	 * Equivalent to RWGGeometry::Transform(GTComplex*) but addressed by surface
	 * index instead of surface label, so no GTComplex/label plumbing has to be
	 * built and torn down per step.
	 */
	void update_all_surface_transforms(const ScuffRigidBody& scuff_rb) {
		geometry_.UnTransform();
		std::memset(geometry_.SurfaceMoved, 0, geometry_.NumSurfaces * sizeof(int));

		const idx_t n = scuff_rb.size();
		for (idx_t i = 0; i < n; ++i) {
			scuff::RWGSurface* surface = get_surface(scuff_rb.surface_tag[i]);

			// GTransformation maps X -> M*X + DX with M indexed row-first,
			// while Matrix3 stores columns; M[r][c] is column c's r-th
			// component.
			scuff::GTransformation gt;
			const Matrix3& rot = scuff_rb.orientation[i];
			gt.M[0][0] = static_cast<double>(rot.ex().x);
			gt.M[1][0] = static_cast<double>(rot.ex().y);
			gt.M[2][0] = static_cast<double>(rot.ex().z);
			gt.M[0][1] = static_cast<double>(rot.ey().x);
			gt.M[1][1] = static_cast<double>(rot.ey().y);
			gt.M[2][1] = static_cast<double>(rot.ey().z);
			gt.M[0][2] = static_cast<double>(rot.ez().x);
			gt.M[1][2] = static_cast<double>(rot.ez().y);
			gt.M[2][2] = static_cast<double>(rot.ez().z);

			const Vector3& pos = scuff_rb.position[i];
			gt.DX[0] = static_cast<double>(pos.x) * constants::ANGSTROM_TO_MICRON;
			gt.DX[1] = static_cast<double>(pos.y) * constants::ANGSTROM_TO_MICRON;
			gt.DX[2] = static_cast<double>(pos.z) * constants::ANGSTROM_TO_MICRON;

			surface->Transform(&gt);
			geometry_.SurfaceMoved[surface->Index] = 1;
		}
	}

	/**
	 * @brief Poses the geometry, solves the BEM system and fills the force SoA.
	 *
	 * Follows scuff-scatter's ordering: transform first, then assemble, since
	 * both the BEM matrix and the RHS depend on the transformed mesh.
	 */
	void compute_forces(ScuffRigidBody& scuff_rb) {
		update_all_surface_transforms(scuff_rb);
		solve();
		read_back_pft(scuff_rb);
	}

  private:
	scuff::RWGSurface* get_surface(int surface_index) {
		if (surface_index < 0 || surface_index >= geometry_.NumSurfaces) {
			MARS_Exception(ExceptionType::ValueError,
						   "SCUFF surface index %d out of range [0,%d)",
						   surface_index,
						   geometry_.NumSurfaces);
		}
		return geometry_.Surfaces[surface_index];
	}

	void solve() {
		geometry_.AssembleBEMMatrix(omega_, bem_matrix_.get());
		if (bem_matrix_->LUFactorize() != 0) {
			MARS_Exception(ExceptionType::RuntimeError, "SCUFF BEM matrix is singular");
		}
		geometry_.AssembleRHSVector(omega_, incident_field_.get(), kn_vector_.get());
		// GetPFTMatrix's EMT method needs the un-solved RHS alongside the
		// solved surface currents, so keep a copy before LUSolve overwrites it.
		rhs_vector_->Copy(kn_vector_.get());
		if (bem_matrix_->LUSolve(kn_vector_.get()) != 0) {
			MARS_Exception(ExceptionType::RuntimeError, "SCUFF BEM solve failed");
		}
	}

	void read_back_pft(ScuffRigidBody& scuff_rb) {
		scuff::PFTOptions options;
		scuff::InitPFTOptions(&options);
		options.IF = incident_field_.get();
		options.RHSVector = rhs_vector_.get();

		// GetPFTMatrix reallocates when handed a wrong-sized matrix, so hand
		// over ownership and take back whatever it returns.
		pft_matrix_.reset(
			geometry_.GetPFTMatrix(kn_vector_.get(), omega_, &options, pft_matrix_.release()));

		const idx_t n = scuff_rb.size();
		if (scuff_rb.plasmonic_force.size() < n) {
			scuff_rb.resize(n);
		}
		for (idx_t i = 0; i < n; ++i) {
			double pft[NUMPFT] = {0.0};
			pft_matrix_->GetEntriesD(scuff_rb.surface_tag[i], ":", pft);

			scuff_rb.plasmonic_force[i] = Vector3(
				static_cast<mars_real>(pft[PFT_XFORCE] * scuff_units::NN_TO_KCAL_PER_MOL_ANGSTROM),
				static_cast<mars_real>(pft[PFT_YFORCE] * scuff_units::NN_TO_KCAL_PER_MOL_ANGSTROM),
				static_cast<mars_real>(pft[PFT_ZFORCE] * scuff_units::NN_TO_KCAL_PER_MOL_ANGSTROM));
			scuff_rb.plasmonic_torque[i] = Vector3(
				static_cast<mars_real>(pft[PFT_XTORQUE] * scuff_units::NN_MICRON_TO_KCAL_PER_MOL),
				static_cast<mars_real>(pft[PFT_YTORQUE] * scuff_units::NN_MICRON_TO_KCAL_PER_MOL),
				static_cast<mars_real>(pft[PFT_ZTORQUE] * scuff_units::NN_MICRON_TO_KCAL_PER_MOL));
		}
	}

	scuff::RWGGeometry geometry_;
	std::string geo_file_;
	std::unique_ptr<IncField> incident_field_;
	cdouble omega_;
	std::unique_ptr<HMatrix> bem_matrix_;
	std::unique_ptr<HVector> rhs_vector_;
	std::unique_ptr<HVector> kn_vector_;
	std::unique_ptr<HMatrix> pft_matrix_;
};

} // namespace MARS
