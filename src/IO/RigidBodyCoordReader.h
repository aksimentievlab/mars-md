#pragma once

#include "../ARBDException.h"
#include "../Objects/RigidBodyProperties.h"
#include "../Types/Types.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ARBD {

/**
 * @brief Reader for legacy `inputRBCoordinates` files.
 *
 * One line per rigid-body instance, consumed in declaration order across every
 * type (type-major, instance-minor) - a single file covers the whole system,
 * which is why this is a top-level config key rather than a per-`rigidBody`
 * block one. Port of legacy RigidBodyController::loadRBCoordinates.
 *
 * Line format (whitespace separated):
 * @code
 *   posX posY posZ  r00 r01 r02 r10 r11 r12 r20 r21 r22  [pX pY pZ  LX LY LZ]
 * @endcode
 * Rotation is row-major (Matrix3 stores columns, so it is transposed on load).
 * The trailing six momentum / angular-momentum values are optional: legacy
 * randomizes them from a Boltzmann distribution when absent, whereas here the
 * instance simply keeps whatever it already had (its config-block value, or
 * zero) - the Langevin integrator equilibrates either way.
 *
 * Lines beginning with '#' and blank lines are skipped, matching legacy.
 */
class RigidBodyCoordReader {
  public:
	static void load(std::vector<RigidBodyIO>& bodies, const std::string& path) {
		std::ifstream file(path);
		if (!file) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"RigidBodyCoordReader: could not open '%s'",
							path.c_str());
		}

		size_t instance = 0;
		std::string line;
		size_t line_number = 0;
		while (std::getline(file, line)) {
			++line_number;
			if (line.empty() || line[0] == '#')
				continue;

			std::istringstream iss(line);
			std::vector<float> v;
			float value = 0.0f;
			while (iss >> value)
				v.push_back(value);
			if (v.empty())
				continue;

			if (instance >= bodies.size()) {
				throw Exception(ExceptionType::ValueError,
								SourceLocation(),
								"RigidBodyCoordReader: '%s' line %zu supplies coordinates for "
								"rigid body %zu, but only %zu were declared",
								path.c_str(),
								line_number,
								instance,
								bodies.size());
			}
			if (v.size() < 12) {
				throw Exception(ExceptionType::ValueError,
								SourceLocation(),
								"RigidBodyCoordReader: '%s' line %zu has %zu values, expected at "
								"least 12 (3 position + 9 row-major orientation)",
								path.c_str(),
								line_number,
								v.size());
			}

			RigidBodyIO& rb = bodies[instance];
			rb.position = Vector3(v[0], v[1], v[2]);
			// Input is row-major; Matrix3's three-vector constructor takes
			// columns, so index down the columns rather than across the rows.
			rb.orientation =
				Matrix3(Vector3(v[3], v[6], v[9]), Vector3(v[4], v[7], v[10]), Vector3(v[5], v[8], v[11]));
			rb.has_orientation = true;

			if (v.size() >= 18) {
				rb.momentum = Vector3(v[12], v[13], v[14]);
				rb.angular_momentum = Vector3(v[15], v[16], v[17]);
			}

			++instance;
		}

		if (instance != bodies.size()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"RigidBodyCoordReader: '%s' supplied %zu coordinate line(s) but %zu "
							"rigid body instance(s) were declared",
							path.c_str(),
							instance,
							bodies.size());
		}
	}
};

} // namespace ARBD
