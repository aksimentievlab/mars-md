#include "Reservoir.h"
namespace ARBD {

ReservoirManager::ReservoirManager(const char* reservoirFile) {
	char line[STRLEN];
	FileHandle file(reservoirFile, "r");
	float x0, y0, z0, x1, y1, z1, n;
	int result = fscanf(file.get(), "%f %f %f %f %f %f %f", &x0, &y0, &z0, &x1, &y1, &z1, &n);
	if (result == 7) { // Check if all 7 values were successfully read
		reservoirs.push_back(Reservoir(Vector3(x0, y0, z0), Vector3(x1, y1, z1), n));
	}

	while (fscanf(file.get(), "%f %f %f %f %f %f %f", &x0, &y0, &z0, &x1, &y1, &z1, &n) == 7) {
		reservoirs.push_back(Reservoir(Vector3(x0, y0, z0), Vector3(x1, y1, z1), n));
	}
	validateRegions();
}

ReservoirManager::~ReservoirManager() {}

void ReservoirManager::validateRegions() {
	for (auto& reservoir : reservoirs) {
		Vector3 a = reservoir.start;
		Vector3 b = reservoir.end;

		if (a.x > b.x) {
			reservoir.start.x = b.x;
			reservoir.end.x = a.x;
		}
		if (a.y > b.y) {
			reservoir.start.y = b.y;
			reservoir.end.y = a.y;
		}
		if (a.z > b.z) {
			reservoir.start.z = b.z;
			reservoir.end.z = a.z;
		}
	}
}

int ReservoirManager::getTargetMeanNum() const {
	std::vector<int> target_nums = std::vector<int>(reservoirs.size());
	for (int i = 0; i < reservoirs.size(); i++) {
		target_nums[i] = reservoirs[i].target_num;
	}
	int sum = 0;
	for (int i = 0; i < reservoirs.size(); i++) {
		sum += target_nums[i];
	}
	return sum / reservoirs.size();
}

int ReservoirManager::getCurrentMeanNum() const {
	std::vector<int> current_nums = std::vector<int>(reservoirs.size());
	for (int i = 0; i < reservoirs.size(); i++) {
		current_nums[i] = reservoirs[i].current_num;
	}
	int sum = 0;
	for (int i = 0; i < reservoirs.size(); i++) {
		sum += current_nums[i];
	}
	return sum / reservoirs.size();
}

} // namespace ARBD
