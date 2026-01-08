#include "../Object_gen.h"
#include "../catch_boiler.h"
#include "Backend/Resource.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/ParticleProperties.h"
using namespace ARBD;

TEST_CASE("Device Particle Copy From Host To Device", "[device][particles]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	std::vector<ParticleType> test_particle_types = create_test_particle_types(2, 50);
	DeviceParticle particles(100, res);
	HostParticleData host_data = create_test_particles(100, "linear", 100.0f);

	particles.copy_from_host(host_data, 100);

	REQUIRE(particles.size() == 100);
	HostParticleData result;
	particles.copy_to_host(result, 100);
	REQUIRE(result.pos[20] == host_data.pos[20]);
}
