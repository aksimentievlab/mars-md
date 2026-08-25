#include "../catch_boiler.h"
#include "Backend/Resource.h"
#include "Object_gen.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/ParticleProperties.h"
using namespace MARS;

TEST_CASE("Device Particle Copy From Host To Device", "[device][particles]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	DeviceParticle particles(100, res);

	// Create particle types (constructor automatically calls copy_from_host)
	std::vector<ParticleType> particle_types_vec = create_test_particle_types(2, 50);
	DeviceParticleTypes particle_types(particle_types_vec, res);

	HostParticleData host_data = create_test_particles(100, "linear", 100.0f);
	host_data.type_id = std::vector<int>(100, 0);

	particles.copy_from_host(host_data, 100);

	REQUIRE(particles.size() == 100);
	HostParticleData result;
	particles.copy_to_host(result, 100);
	REQUIRE(result.pos[20] == host_data.pos[20]);
}
