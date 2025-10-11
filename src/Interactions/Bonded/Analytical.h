#pragma once
#include "Header.h"
#include "Types/Types.h"

namespace ARBD {
class Register_Potential {
  public:
	// Returns a unique integer ID for the potential name
	int register_potential(const std::string& name) {
		if (m_name_to_id.find(name) == m_name_to_id.end()) {
			int new_id = m_name_to_id.size();
			m_name_to_id[name] = new_id;
		}
		return m_name_to_id[name];
	}

	int get_id(const std::string& name) const {
		return m_name_to_id.at(name);
	}

  private:
	std::unordered_map<std::string, int> m_name_to_id;
};

enum class BondType { Harmonic = 0, Morse = 1 }; // for precompiled type_id reference.

struct AnalyticalBondFunctor {
	HOST DEVICE void
	operator()(idx_t i,		// Thread ID, corresponds to the i-th bond IN THE CURRENT BATCH
			   int type_id, // The integer ID for this batch of bonds (e.g., 0 for Harmonic)
			   int offset,
			   // --- Pointers to DEVICE SoA Data ---
			   DEVICE_PTR(Vector3) positions,
			   DEVICE_PTR(Vector3) forces,
			   DEVICE_PTR(int2) particle_indices, // Bond connectivity for ALL bond
			   // --- Pointers to Parameter Arrays for Each Potential Type ---
			   DEVICE_PTR(const float2) harmonic_params, // { k, r0 }
			   DEVICE_PTR(const Vector3) morse_params	 // { D0, a, r0 }
			   // Add more parameter array pointers here for new C++ potentials
	) const {
		// 1. Get Particle Indices for this specific bond
		// Note: We use the thread ID 'i' directly as the index into the
		// parameter arrays, but we must use an offset for the DEVICE particle_indices array.
		// This offset is handled by the caller when launching the kernel.

		const int p1_idx = particle_indices[i + offset].x;
		const int p2_idx = particle_indices[i + offset].y;

		// 2. Fetch Particle Positions
		const Vector3 pos1 = positions[p1_idx];
		const Vector3 pos2 = positions[p2_idx];

		// 3. Calculate distance and direction
		const Vector3 r_ij = pos2 - pos1;
		const float distance = r_ij.length();
		if (distance < 1e-6f) { // Avoid division by zero
			return;
		}
		const Vector3 direction = r_ij / distance;

		float force_magnitude = 0.0f;

		switch (type_id) {
		case 0: {								   // harmonic
			const float k = harmonic_params[i].x;  // Spring constant
			const float r0 = harmonic_params[i].y; // Equilibrium distance

			force_magnitude = -k * (distance - r0);
			break;
		}

		case 1: {								// morse
			const float D0 = morse_params[i].x; // Dissociation energy
			const float a = morse_params[i].y;	// Controls the width of the potential
			const float r0 = morse_params[i].z; // Equilibrium distance

			// Force Law: F = 2 * D0 * a * exp(-a * (r - r0)) * [1 - exp(-a * (r - r0))]
			const float exp_term = expf(-a * (distance - r0));
			force_magnitude = 2.0f * D0 * a * exp_term * (1.0f - exp_term);
			break;
		}
		}

		// 5. Calculate the final force vector
		const Vector3 force_vec = direction * force_magnitude;

		// 6. Atomically Add Forces to the DEVICE Force Array
		ATOMIC_ADD(&forces[p1_idx], -force_vec);
		ATOMIC_ADD(&forces[p2_idx], force_vec);
	}
};
} // namespace ARBD
