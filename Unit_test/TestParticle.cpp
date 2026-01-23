
#include "Object_gen.h"
#include "Objects/ParticleProperties.h"
#include "Types/Types.h"
#include <random>
#include <vector>
using namespace ARBD;

HostParticleData create_test_particles(int count, const std::string& pattern, float box_size) {
	std::vector<ParticleRead> particles;
	particles.reserve(count);

	// Initialize random number generator for random patterns
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0f, box_size);

	for (int i = 0; i < count; ++i) {
		ParticleRead p;
		p.type_name = "A"; // Alternate between two particle types
		p.id = i;
		if (pattern == "linear") {
			// Linear arrangement along x-axis
			float spacing = box_size / (count + 1);
			p.position = Vector3_t<float>{(i + 1) * spacing, box_size * 0.5f, box_size * 0.5f};
		} else if (pattern == "grid") {
			// 3D grid arrangement
			int side = std::ceil(std::cbrt(count));
			float spacing = box_size / (side + 1);
			int x = i % side;
			int y = (i / side) % side;
			int z = i / (side * side);
			p.position = Vector3_t<float>{(x + 1) * spacing, (y + 1) * spacing, (z + 1) * spacing};
		} else if (pattern == "random") {
			// Random positions within box
			p.position = Vector3_t<float>{dist(gen), dist(gen), dist(gen)};
		} else if (pattern == "sphere") {
			// Spherical arrangement
			float radius = box_size * 0.3f;
			float theta = 2.0f * M_PI * i / count;
			float phi = M_PI * (i + 0.5f) / count;
			p.position =
				Vector3_t<float>{box_size * 0.5f + radius * std::sin(phi) * std::cos(theta),
								 box_size * 0.5f + radius * std::sin(phi) * std::sin(theta),
								 box_size * 0.5f + radius * std::cos(phi)};
		} else {
			// Default to linear if pattern not recognized
			float spacing = box_size / (count + 1);
			p.position = Vector3_t<float>{(i + 1) * spacing, box_size * 0.5f, box_size * 0.5f};
		}

		particles.push_back(p);
	}

	HostParticleData host_data;
	host_data.resize(particles.size());
	for (size_t i = 0; i < particles.size(); ++i) {
		host_data.global_id[i] = particles[i].id;
		host_data.type_id[i] = 0;
		host_data.pos[i] = particles[i].position;
		host_data.mom[i] = particles[i].momentum;
	}
	return host_data;
}
std::vector<ParticleType> create_test_particle_types(int type_count, int particles_per_type) {
	std::vector<ParticleType> particle_types;
	particle_types.reserve(type_count);

	for (int i = 0; i < type_count; ++i) {
		ParticleType type(std::string("TestType") + std::to_string(i));
		type.mass = 100.0f + i * 50.0f; // Increasing mass for different types
		type.num = particles_per_type;

		// Set damping coefficients (example values)
		type.diffusion = Vector3(100.0f, 100.0f, 100.0f);

		particle_types.push_back(type);
	}

	return particle_types;
}
