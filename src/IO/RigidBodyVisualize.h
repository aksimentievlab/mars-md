#pragma once

#include "../ARBDException.h"
#include "../Objects/ParticleProperties.h"
#include "../Objects/RigidBodyProperties.h"
#include "../Types/Types.h"
#include "Constants.h"
#include "PsfPdbIO.h"
#include <algorithm>

namespace ARBD {

class RigidBodyPdbPsfReader {
  public:
	static void load(RigidBodyType& rt,
					 const std::string& pdb_path,
					 const std::string& psf_path,
					 const Vector3& reference_point,
					 const std::vector<ParticleType>& known_particle_types,
					 const std::string& attached_marker = constants::kAttachedSegnameMarker) {

		const PsfPdbStructure structure = PsfPdbStructure::from_psf_pdb(psf_path, pdb_path);

		rt.template_particles.reserve(structure.atoms.size());
		for (const PdbAtomRecord& atom : structure.atoms) {
			const bool known =
				std::any_of(known_particle_types.begin(),
							known_particle_types.end(),
							[&](const ParticleType& pt) { return pt.name == atom.resname; });
			if (!known) {
				throw Exception(ExceptionType::ValueError,
								SourceLocation(),
								"RigidBodyTemplateReader: resname '%s' (resid %d) in type '%s' "
								"has no matching predeclared particle type",
								atom.resname.c_str(),
								atom.resid,
								rt.name.c_str());
			}

			CosmeticParticle cosmetic{};
			cosmetic.name = atom.name;
			cosmetic.resname = atom.resname;
			cosmetic.segname = atom.segname;
			cosmetic.type_name = atom.type_name;
			cosmetic.resid = atom.resid;
			cosmetic.body_frame_position = atom.position - reference_point;

			if (cosmetic.segname == attached_marker) {
				ParticleIO p{};
				p.type_name = cosmetic.resname;
				p.position = cosmetic.body_frame_position;
				rt.attached_particle.push_back(p);
				// Index into this type's attached_particle vector - type-local,
				// so it is well defined here. The *global* particle index is
				// not: that depends on how many instances precede this one, and
				// is resolved later as rb.attached_start + this value. Stays -1
				// for template atoms that are cosmetic only, which is what marks
				// them for position-by-transform in the trajectory writer.
				cosmetic.attached_particle_index =
					static_cast<int>(rt.attached_particle.size()) - 1;
			}

			rt.template_particles.push_back(cosmetic);
		}

		rt.template_bonds = structure.bonds;
	};
};
} // namespace ARBD
