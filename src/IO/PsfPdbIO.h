#pragma once

#include "../Types/Types.h"
#include <string>
#include <vector>

namespace ARBD {

struct PdbAtomRecord {
	int serial = 0;
	std::string name;
	std::string resname;
	std::string chain;
	std::string segname; // For Rigidbodies-- segname "ATT" reservers for attached particles.
	int resid = 0;
	Vector3 position{};
	float occupancy = 1.0f;
	float beta = 0.0f;
	float charge = 0.0f;
	float mass = 1.0f;
	std::string type_name;
	bool fixed = false;
	float radius = -1.0f; // negative => derive from mass in PQR writer
};

// Owns merged PSF/PDB atom records and connectivity. Safe to default-construct,
// copy, and mutate in application code; factories only populate from disk.
struct PsfPdbStructure {
	Vector3 box_dimensions{};
	bool has_cryst1 = false;
	std::vector<PdbAtomRecord> atoms;
	std::vector<int2> bonds;
	std::vector<int3> angles;
	std::vector<int4> dihedrals;
	std::vector<int4> impropers;

	std::string pdb_path;
	std::string psf_path;

	static PsfPdbStructure from_pdb(const std::string& pdb_path);
	static PsfPdbStructure from_psf_pdb(const std::string& psf_path, const std::string& pdb_path);

	bool has_topology() const {
		return !psf_path.empty();
	}

	void write_pdb(const std::string& path, bool beta_from_fixed = false) const;
	void write_pdb(const std::string& path,
				   const Vector3& dimensions,
				   bool beta_from_fixed = false) const;
	void write_psf(const std::string& path) const;
	void write_pqr(const std::string& path) const;
	void write_pqr(const std::string& path, const Vector3& dimensions) const;

  private:
	static PsfPdbStructure read_pdb_file(const std::string& path);
	static void read_psf_file(PsfPdbStructure& structure, const std::string& path);
	static void merge_pdb_coordinates(PsfPdbStructure& structure, const PsfPdbStructure& pdb);

	Vector3 output_box(const Vector3& dimensions) const;
};

} // namespace ARBD
