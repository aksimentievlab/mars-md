#include "PsfPdbIO.h"

#include "../MARSException.h"
#include "FileHandle.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace MARS {
namespace {

constexpr int kPdbRecordLength = 80;
constexpr int kPsfRecordLength = 256;

void trim_inplace(std::string& s) {
	while (!s.empty() && s.front() == ' ')
		s.erase(s.begin());
	while (!s.empty() && s.back() == ' ')
		s.pop_back();
}

std::string substr_field(const char* line, int start, int len) {
	std::string out(line + start, static_cast<size_t>(len));
	trim_inplace(out);
	return out;
}

float parse_float_field(const char* line, int start, int len) {
	const std::string field = substr_field(line, start, len);
	return field.empty() ? 0.0f : static_cast<float>(std::atof(field.c_str()));
}

int parse_int_field(const char* line, int start, int len) {
	const std::string field = substr_field(line, start, len);
	return field.empty() ? 0 : std::atoi(field.c_str());
}

bool is_atom_record(const char* line) {
	return std::strncmp(line, "ATOM ", 5) == 0 || std::strncmp(line, "HETATM", 6) == 0;
}

PdbAtomRecord parse_pdb_atom_line(const char* line) {
	PdbAtomRecord atom;
	atom.serial = parse_int_field(line, 6, 5);
	atom.name = substr_field(line, 12, 4);
	atom.resname = substr_field(line, 17, 3);
	atom.chain = substr_field(line, 21, 1);
	if (atom.chain.empty())
		atom.chain = "A";
	atom.resid = parse_int_field(line, 22, 4);
	atom.position.x = parse_float_field(line, 30, 8);
	atom.position.y = parse_float_field(line, 38, 8);
	atom.position.z = parse_float_field(line, 46, 8);
	atom.occupancy = parse_float_field(line, 54, 6);
	atom.beta = parse_float_field(line, 60, 6);
	if (static_cast<int>(std::strlen(line)) >= 76)
		atom.segname = substr_field(line, 72, 4);
	return atom;
}

void parse_pdb_cryst1(const char* line, Vector3& box) {
	box.x = parse_float_field(line, 6, 9);
	box.y = parse_float_field(line, 15, 9);
	box.z = parse_float_field(line, 24, 9);
}

int count_digits_before_decimal(float value) {
	const float magnitude = std::fabs(value);
	if (magnitude < 1e-6f)
		return 1;
	return static_cast<int>(std::floor(std::log10(magnitude))) + 1;
}

std::string format_pdb_coordinate(float value) {
	const int digits = count_digits_before_decimal(value);
	if (digits > 7) {
		throw Exception(ExceptionType::NotImplementedError,
						SourceLocation(),
						"arbdmodel is unable to write a PDB with such large dimensions; "
						"consider whether shifting your object or scaling dimensions can be "
						"used as a workaround");
	}
	const int decimals = 7 - digits;
	char buffer[32];
	std::snprintf(buffer, sizeof(buffer), "%*.*f", 8, decimals, static_cast<double>(value));
	return buffer;
}

std::string format_pdb_serial(int serial) {
	if (serial >= 100000)
		return " *****";
	char buffer[16];
	std::snprintf(buffer, sizeof(buffer), "%6d", serial);
	return buffer;
}

std::string center_field(const std::string& value, size_t width) {
	if (value.size() >= width)
		return value.substr(0, width);
	const size_t padding = width - value.size();
	const size_t left = padding / 2;
	return std::string(left, ' ') + value + std::string(padding - left, ' ');
}

char next_chain_id(char c) {
	if (c >= 'Z')
		return '0';
	if (c < 'A' && c >= '9')
		return 'A';
	return static_cast<char>(c + 1);
}

bool valid_chain_id(char c) {
	return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

char assign_chain_id(char requested, std::unordered_set<char>& used_chains) {
	char c = static_cast<char>(std::toupper(static_cast<unsigned char>(requested)));
	while (!valid_chain_id(c) || used_chains.count(c) != 0)
		c = next_chain_id(c);
	used_chains.insert(c);
	if (used_chains.size() >= 36)
		used_chains = {c};
	return c;
}

int psf_start_block(FILE* file, const char* blockname) {
	char inbuf[kPsfRecordLength + 2];
	int count = -1;
	while (std::fgets(inbuf, kPsfRecordLength + 1, file)) {
		if (std::strstr(inbuf, blockname) != nullptr) {
			count = std::atoi(inbuf);
			break;
		}
	}
	if (count < 0) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"PsfPdbStructure: failed to find '%s' section",
						blockname);
	}
	return count;
}

int psf_atoifw(char** ptr, int field_width) {
	char* op = *ptr;
	int value = 0;
	int consumed = 0;
	std::sscanf(op, "%d%n", &value, &consumed);
	if (consumed == field_width) {
		*ptr += consumed;
	} else if (consumed < field_width) {
		while (consumed < field_width && op[consumed] == ' ')
			++consumed;
		*ptr += consumed;
	} else if (consumed < 2 * field_width) {
		*ptr += consumed;
	} else {
		const char saved = op[field_width];
		op[field_width] = '\0';
		value = std::atoi(op);
		op[field_width] = saved;
		*ptr += field_width;
	}
	return value;
}

bool read_psf_bonds(FILE* file,
					int count,
					bool namd_fmt,
					bool charmm_ext,
					std::vector<int2>& bonds) {
	bonds.reserve(static_cast<size_t>(count));
	const int field_width = charmm_ext ? 10 : 8;
	char inbuf[kPsfRecordLength + 2];
	char* bondptr = nullptr;

	for (int i = 0; i < count; ++i) {
		int from = 0;
		int to = 0;
		if (namd_fmt) {
			if (std::fscanf(file, "%d %d", &from, &to) != 2)
				return false;
		} else {
			if ((i % 4) == 0) {
				if (!std::fgets(inbuf, kPsfRecordLength + 1, file))
					return false;
				bondptr = inbuf;
			}
			from = psf_atoifw(&bondptr, field_width);
			to = psf_atoifw(&bondptr, field_width);
			if (from < 1 || to < 1)
				return false;
		}
		bonds.emplace_back(from - 1, to - 1);
	}
	return true;
}

bool read_psf_triplets(FILE* file,
					   int count,
					   int per_line,
					   bool charmm_ext,
					   std::vector<int3>& entries) {
	entries.reserve(static_cast<size_t>(count));
	const int field_width = charmm_ext ? 10 : 8;
	char inbuf[kPsfRecordLength + 2];
	char* ptr = nullptr;

	for (int i = 0; i < count; ++i) {
		if ((i % per_line) == 0) {
			if (!std::fgets(inbuf, kPsfRecordLength + 1, file))
				return false;
			ptr = inbuf;
		}
		const int a = psf_atoifw(&ptr, field_width);
		const int b = psf_atoifw(&ptr, field_width);
		const int c = psf_atoifw(&ptr, field_width);
		if (a < 1 || b < 1 || c < 1)
			return false;
		entries.emplace_back(a - 1, b - 1, c - 1);
	}
	return true;
}

bool read_psf_quads(FILE* file,
					int count,
					int per_line,
					bool charmm_ext,
					std::vector<int4>& entries) {
	entries.reserve(static_cast<size_t>(count));
	const int field_width = charmm_ext ? 10 : 8;
	char inbuf[kPsfRecordLength + 2];
	char* ptr = nullptr;

	for (int i = 0; i < count; ++i) {
		if ((i % per_line) == 0) {
			if (!std::fgets(inbuf, kPsfRecordLength + 1, file))
				return false;
			ptr = inbuf;
		}
		const int a = psf_atoifw(&ptr, field_width);
		const int b = psf_atoifw(&ptr, field_width);
		const int c = psf_atoifw(&ptr, field_width);
		const int d = psf_atoifw(&ptr, field_width);
		if (a < 1 || b < 1 || c < 1 || d < 1)
			return false;
		entries.emplace_back(a - 1, b - 1, c - 1, d - 1);
	}
	return true;
}

PdbAtomRecord parse_psf_atom_line(const char* line, bool namd_fmt, bool charmm_ext) {
	PdbAtomRecord atom;
	char segname[16] = {};
	char residstr[16] = {};
	char resname[16] = {};
	char name[16] = {};
	char type_name[16] = {};
	int serial = 0;

	if (namd_fmt) {
		const int parsed = std::sscanf(line,
									   "%d %15s %15s %15s %15s %15s %f %f",
									   &serial,
									   segname,
									   residstr,
									   resname,
									   name,
									   type_name,
									   &atom.charge,
									   &atom.mass);
		if (parsed < 8) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"PsfPdbStructure: failed to parse NAMD atom line: '%s'",
							line);
		}
		atom.resid = std::atoi(residstr);
		atom.serial = serial;
	} else if (charmm_ext) {
		atom.segname = substr_field(line, 11, 7);
		atom.resname = substr_field(line, 29, 7);
		atom.name = substr_field(line, 38, 7);
		atom.type_name = substr_field(line, 47, 6);
		if (!atom.type_name.empty() && !std::isdigit(static_cast<unsigned char>(atom.type_name[0])))
			atom.type_name = substr_field(line, 47, 4);
		std::sscanf(line + 20, "%d", &atom.resid);
		atom.charge = parse_float_field(line, 52, 14);
		atom.mass = parse_float_field(line, 66, 14);
		atom.serial = parse_int_field(line, 0, 10);
	} else {
		const int parsed = std::sscanf(line,
									   "%d %15s %15s %15s %15s %15s %f %f",
									   &serial,
									   segname,
									   residstr,
									   resname,
									   name,
									   type_name,
									   &atom.charge,
									   &atom.mass);
		if (parsed < 8) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"PsfPdbStructure: failed to parse atom line: '%s'",
							line);
		}
		atom.resid = std::atoi(residstr);
		atom.serial = serial;
	}

	if (namd_fmt || !charmm_ext) {
		atom.segname = segname;
		atom.resname = resname;
		atom.name = name;
		atom.type_name = type_name;
	}
	return atom;
}

void write_psf_bonds(FILE* file, const std::vector<int2>& bonds) {
	int counter = 0;
	for (const int2& bond : bonds) {
		std::fprintf(file, "%8d%8d", bond.x + 1, bond.y + 1);
		++counter;
		if (counter == 4) {
			std::fputc('\n', file);
			counter = 0;
		} else {
			std::fputc(' ', file);
		}
	}
	if (counter == 0)
		std::fputc('\n', file);
	else
		std::fputs("\n\n", file);
}

void write_psf_angles(FILE* file, const std::vector<int3>& angles) {
	int counter = 0;
	for (const int3& angle : angles) {
		std::fprintf(file, "%8d%8d%8d", angle.x + 1, angle.y + 1, angle.z + 1);
		++counter;
		if (counter == 3) {
			std::fputc('\n', file);
			counter = 0;
		} else {
			std::fputc(' ', file);
		}
	}
	if (counter == 0)
		std::fputc('\n', file);
	else
		std::fputs("\n\n", file);
}

void write_psf_quartet_group(FILE* file, const std::vector<int4>& entries, int per_line) {
	int counter = 0;
	for (const int4& entry : entries) {
		std::fprintf(file, "%8d%8d%8d%8d", entry.x + 1, entry.y + 1, entry.z + 1, entry.t + 1);
		++counter;
		if (counter == per_line) {
			std::fputc('\n', file);
			counter = 0;
		} else {
			std::fputc(' ', file);
		}
	}
	if (counter == 0)
		std::fputc('\n', file);
	else
		std::fputs("\n\n", file);
}

void write_pdb_atoms(FILE* file,
					 const std::vector<PdbAtomRecord>& atoms,
					 const Vector3& dimensions,
					 bool beta_from_fixed) {
	std::fprintf(file,
				 "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n",
				 static_cast<double>(dimensions.x),
				 static_cast<double>(dimensions.y),
				 static_cast<double>(dimensions.z));

	std::unordered_set<char> used_chains;
	const PdbAtomRecord* last_atom = nullptr;
	char assigned_chain = 'A';

	for (const PdbAtomRecord& atom : atoms) {
		int resid = atom.resid;
		if (resid > 9999)
			resid = ((resid - 1) % 9999) + 1;

		float beta = atom.beta;
		if (beta_from_fixed)
			beta = atom.fixed ? 1.0f : 0.0f;

		const char requested_chain =
			atom.chain.empty()
				? 'A'
				: static_cast<char>(std::toupper(static_cast<unsigned char>(atom.chain[0])));
		if (last_atom == nullptr || resid < last_atom->resid ||
			requested_chain !=
				static_cast<char>(std::toupper(static_cast<unsigned char>(last_atom->chain[0])))) {
			assigned_chain = assign_chain_id(requested_chain, used_chains);
		}

		const std::string serial = format_pdb_serial(atom.serial > 0 ? atom.serial : atom.resid);
		const std::string name = center_field(atom.name, 4);
		std::string resname = atom.resname;
		if (resname.size() > 3)
			resname = resname.substr(0, 3);

		char resid_buf[8];
		std::snprintf(resid_buf, sizeof(resid_buf), "%4d", resid);

		const std::string x = format_pdb_coordinate(atom.position.x);
		const std::string y = format_pdb_coordinate(atom.position.y);
		const std::string z = format_pdb_coordinate(atom.position.z);

		std::string segname = atom.segname;
		if (segname.size() > 4)
			segname = segname.substr(0, 4);

		std::fprintf(file,
					 "ATOM %6.6s %4.4s %-3.3s %c%-5.5s   %8.8s%8.8s%8.8s%6.2f%6.2f      %-4.4s\n",
					 serial.c_str(),
					 name.c_str(),
					 resname.c_str(),
					 assigned_chain,
					 resid_buf,
					 x.c_str(),
					 y.c_str(),
					 z.c_str(),
					 static_cast<double>(atom.occupancy),
					 static_cast<double>(beta),
					 segname.c_str());
		last_atom = &atom;
	}
}

void write_pqr_atoms(FILE* file,
					 const std::vector<PdbAtomRecord>& atoms,
					 const Vector3& dimensions) {
	std::fprintf(file,
				 "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n",
				 static_cast<double>(dimensions.x),
				 static_cast<double>(dimensions.y),
				 static_cast<double>(dimensions.z));

	for (const PdbAtomRecord& atom : atoms) {
		const int serial = atom.serial > 0 ? atom.serial : atom.resid;
		const std::string serial_str = format_pdb_serial(serial);
		const std::string name = center_field(atom.name, 4);
		std::string resname = atom.resname;
		if (resname.size() > 3)
			resname = resname.substr(0, 3);
		const char chain =
			atom.chain.empty()
				? 'A'
				: static_cast<char>(std::toupper(static_cast<unsigned char>(atom.chain[0])));

		char resid_buf[8];
		std::snprintf(resid_buf, sizeof(resid_buf), "%-4d", atom.resid);

		float radius = atom.radius;
		if (radius < 0.0f)
			radius = 2.0f * std::pow(atom.mass / 16.0f, 0.333333f);

		std::fprintf(file,
					 "ATOM %6.6s %4.4s %-3.3s %c %5.5s   %.6f %.6f %.6f %.4f %.6f\n",
					 serial_str.c_str(),
					 name.c_str(),
					 resname.c_str(),
					 chain,
					 resid_buf,
					 static_cast<double>(atom.position.x),
					 static_cast<double>(atom.position.y),
					 static_cast<double>(atom.position.z),
					 static_cast<double>(atom.charge),
					 static_cast<double>(radius));
	}
}

void write_psf_topology(FILE* file, const PsfPdbStructure& structure) {
	std::fputs("PSF NAMD\n\n", file);
	std::fprintf(file, "%8d !NTITLE\n\n", 0);
	std::fprintf(file, "%8zu !NATOM\n", structure.atoms.size());

	for (size_t i = 0; i < structure.atoms.size(); ++i) {
		const PdbAtomRecord& atom = structure.atoms[i];
		char resid_buf[16];
		std::snprintf(resid_buf, sizeof(resid_buf), "%d  ", atom.resid);
		std::fprintf(file,
					 "%8zu %-7.7s %-10.10s %-7.7s %-7.7s %-7.7s %g %g\n",
					 i + 1,
					 atom.segname.c_str(),
					 resid_buf,
					 atom.resname.c_str(),
					 atom.name.c_str(),
					 atom.type_name.c_str(),
					 static_cast<double>(atom.charge),
					 static_cast<double>(atom.mass));
	}
	std::fputc('\n', file);

	std::fprintf(file, "%8zu !NBOND\n", structure.bonds.size());
	if (!structure.bonds.empty())
		write_psf_bonds(file, structure.bonds);
	else
		std::fputc('\n', file);

	std::fprintf(file, "%8zu !NTHETA\n", structure.angles.size());
	if (!structure.angles.empty())
		write_psf_angles(file, structure.angles);
	else
		std::fputc('\n', file);

	std::fprintf(file, "%8zu !NPHI\n", structure.dihedrals.size());
	if (!structure.dihedrals.empty())
		write_psf_quartet_group(file, structure.dihedrals, 2);
	else
		std::fputc('\n', file);

	std::fprintf(file, "%8zu !NIMPHI\n", structure.impropers.size());
	if (!structure.impropers.empty())
		write_psf_quartet_group(file, structure.impropers, 2);
	else
		std::fputc('\n', file);

	std::fprintf(file, "\n%8d !NDON: donors\n\n\n", 0);
	std::fprintf(file, "\n%8d !NACC: acceptors\n\n\n", 0);
	std::fputs("\n       0 !NNB\n\n", file);

	const size_t natoms = structure.atoms.size();
	for (size_t i = 0; i < natoms / 8; ++i)
		std::fputs("      0       0       0       0       0       0       0       0\n", file);
	for (size_t i = 0; i < natoms - 8 * (natoms / 8); ++i)
		std::fputs("      0", file);
	std::fputs("\n\n       1       0 !NGRP\n\n", file);
}

} // namespace

PsfPdbStructure PsfPdbStructure::read_pdb_file(const std::string& path) {
	PsfPdbStructure structure;
	structure.pdb_path = path;
	char line[kPdbRecordLength + 3] = {};
	FileHandle file(path.c_str(), "r");

	while (std::fgets(line, sizeof(line), file.get())) {
		if (std::strncmp(line, "CRYST1", 6) == 0) {
			parse_pdb_cryst1(line, structure.box_dimensions);
			structure.has_cryst1 = true;
		} else if (is_atom_record(line)) {
			structure.atoms.push_back(parse_pdb_atom_line(line));
		}
	}
	return structure;
}

void PsfPdbStructure::read_psf_file(PsfPdbStructure& structure, const std::string& path) {
	char inbuf[kPsfRecordLength * 8 + 2] = {};
	bool namd_fmt = false;
	bool charmm_ext = false;
	int num_atoms = -1;

	FileHandle file(path.c_str(), "r");
	FILE* fp = file.get();

	while (std::fgets(inbuf, sizeof(inbuf), fp)) {
		if (std::strstr(inbuf, "REMARKS") != nullptr)
			continue;
		if (std::strstr(inbuf, "PSF") != nullptr) {
			if (std::strstr(inbuf, "NAMD") != nullptr)
				namd_fmt = true;
			if (std::strstr(inbuf, "EXT") != nullptr)
				charmm_ext = true;
		} else if (std::strstr(inbuf, "NATOM") != nullptr) {
			num_atoms = std::atoi(inbuf);
			break;
		}
	}

	if (num_atoms < 0) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"PsfPdbStructure: failed to find !NATOM section in '%s'",
						path.c_str());
	}

	structure.atoms.clear();
	structure.atoms.reserve(static_cast<size_t>(num_atoms));
	for (int i = 0; i < num_atoms; ++i) {
		if (!std::fgets(inbuf, kPsfRecordLength + 1, fp)) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"PsfPdbStructure: unexpected EOF reading atoms in '%s'",
							path.c_str());
		}
		structure.atoms.push_back(parse_psf_atom_line(inbuf, namd_fmt, charmm_ext));
	}

	const int num_bonds = psf_start_block(fp, "NBOND");
	if (num_bonds > 0 && !read_psf_bonds(fp, num_bonds, namd_fmt, charmm_ext, structure.bonds)) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"PsfPdbStructure: failed to read bonds in '%s'",
						path.c_str());
	}

	const int num_angles = psf_start_block(fp, "NTHETA");
	if (num_angles > 0 && !read_psf_triplets(fp, num_angles, 3, charmm_ext, structure.angles)) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"PsfPdbStructure: failed to read angles in '%s'",
						path.c_str());
	}

	const int num_dihedrals = psf_start_block(fp, "NPHI");
	if (num_dihedrals > 0 &&
		!read_psf_quads(fp, num_dihedrals, 2, charmm_ext, structure.dihedrals)) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"PsfPdbStructure: failed to read dihedrals in '%s'",
						path.c_str());
	}

	const int num_impropers = psf_start_block(fp, "NIMPHI");
	if (num_impropers > 0 &&
		!read_psf_quads(fp, num_impropers, 2, charmm_ext, structure.impropers)) {
		throw Exception(ExceptionType::FileIoError,
						SourceLocation(),
						"PsfPdbStructure: failed to read impropers in '%s'",
						path.c_str());
	}
}

void PsfPdbStructure::merge_pdb_coordinates(PsfPdbStructure& structure,
											const PsfPdbStructure& pdb) {
	if (structure.atoms.size() != pdb.atoms.size()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"PsfPdbStructure: PSF atom count %zu != PDB atom count %zu",
						structure.atoms.size(),
						pdb.atoms.size());
	}

	for (size_t i = 0; i < structure.atoms.size(); ++i) {
		structure.atoms[i].position = pdb.atoms[i].position;
		structure.atoms[i].occupancy = pdb.atoms[i].occupancy;
		structure.atoms[i].beta = pdb.atoms[i].beta;
		if (structure.atoms[i].chain.empty())
			structure.atoms[i].chain = pdb.atoms[i].chain;
		if (structure.atoms[i].segname.empty())
			structure.atoms[i].segname = pdb.atoms[i].segname;
	}

	structure.box_dimensions = pdb.box_dimensions;
	structure.has_cryst1 = pdb.has_cryst1;
}

PsfPdbStructure PsfPdbStructure::from_pdb(const std::string& pdb_path) {
	return read_pdb_file(pdb_path);
}

PsfPdbStructure PsfPdbStructure::from_psf_pdb(const std::string& psf_path,
											  const std::string& pdb_path) {
	PsfPdbStructure structure;
	structure.psf_path = psf_path;
	structure.pdb_path = pdb_path;
	read_psf_file(structure, psf_path);
	merge_pdb_coordinates(structure, read_pdb_file(pdb_path));
	return structure;
}

Vector3 PsfPdbStructure::output_box(const Vector3& dimensions) const {
	if (dimensions.x != 0.0f || dimensions.y != 0.0f || dimensions.z != 0.0f)
		return dimensions;
	return box_dimensions;
}

void PsfPdbStructure::write_pdb(const std::string& path, bool beta_from_fixed) const {
	write_pdb(path, box_dimensions, beta_from_fixed);
}

void PsfPdbStructure::write_pdb(const std::string& path,
								const Vector3& dimensions,
								bool beta_from_fixed) const {
	FileHandle file(path.c_str(), "w");
	write_pdb_atoms(file.get(), atoms, output_box(dimensions), beta_from_fixed);
}

void PsfPdbStructure::write_psf(const std::string& path) const {
	FileHandle file(path.c_str(), "w");
	write_psf_topology(file.get(), *this);
}

void PsfPdbStructure::write_pqr(const std::string& path) const {
	write_pqr(path, box_dimensions);
}

void PsfPdbStructure::write_pqr(const std::string& path, const Vector3& dimensions) const {
	FileHandle file(path.c_str(), "w");
	write_pqr_atoms(file.get(), atoms, output_box(dimensions));
}

} // namespace MARS
