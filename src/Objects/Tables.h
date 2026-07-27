#pragma once
#include "Header.h"
#include "IO/FileHandle.h"
#include "Interactions/NonBondedInteraction.h"
#include "Types/Types.h"
#include <string>
#include <unordered_map>

namespace ARBD {
/**
 * @brief Registry to load and cache tabulated potentials by name
 *
 * Host-side singleton that manages tabulated potential tables for bonds, angles, and dihedrals.
 * Provides caching and indexing functionality for efficient lookup during configuration parsing.
 */

enum class TabulatedType { NonBondedPair, Bond, Angle, Dihedral, Default };

struct Table {
	std::string name;
	TabulatedType type;
	bool is_periodic{false};
	arbd_real step_size{0.0};
	arbd_real start{0.0};

	std::vector<arbd_real> X;
	std::vector<arbd_real> Y;
	Table(TabulatedType type) : type(type) {
		if (type == TabulatedType::Dihedral) {
			is_periodic = true;
			start = -constants::PI;
		}
		check_same_step_size();
	}
	bool check_same_step_size() {
		if (X.empty() || X.size() < 2) {
			LOGWARN("Table: Not enough values to check step size");
			return false;
		}
		step_size = X[1] - X[0];
		start = X[0];
		for (size_t i = 2; i < X.size(); ++i) {
			if (std::abs(X[i] - X[i - 1]) - step_size > 1e-4) {
				LOGWARN("Table: Step size is not the same for all values at index {}: {}", i, X[i]);
				return false;
			}
		}
		return true;
	}

	void read_file(std::string_view file_name, std::string name_ = "") {
		if (name_.empty()) {
			this->name = std::filesystem::path(file_name).stem().string();
		} else {
			this->name = name_;
		}
		FileHandle file(file_name.data(), "r");
		FILE* fp = file.get();

		char* line = nullptr;
		size_t len = 0;
		ssize_t read;
		size_t lineNumber = 0;

		while ((read = getline(&line, &len, fp)) != -1) {
			++lineNumber;
			std::string lineStr(line, read);

			// Skip empty lines and comments
			if (lineStr.empty() || lineStr[0] == '#') {
				continue;
			}

			std::istringstream iss(lineStr);
			arbd_real x, y;

			if (iss >> x >> y) {
				X.push_back(x);
				Y.push_back(y);
			} else {
				LOGWARN("Tables: Failed to parse line {}: {}", lineNumber, lineStr);
			}
		}

		if (line) {
			free(line);
		}

		// The constructor's check_same_step_size() call ran on an empty
		// table (X/Y not populated yet), so it always took the "not enough
		// values" early-return and left step_size/start at their 0.0
		// defaults. Recompute now that the data has actually been loaded -
		// otherwise every TabulatedPotential built from this table gets
		// step_inv=0, which silently zeroes every force (force_magnitude =
		// dU * step_inv in TabulatedPotential::compute), while still
		// reporting a plausible-looking (but wrong, always-at-x=start) energy.
		check_same_step_size();
	}
};

class TablesRegistry {

  public:
	TablesRegistry() = default;

	HOST int get_or_load_angle(std::string_view file_name, std::string_view config_file_path = "") {
		// file name could be a path or a name, if it is a path, we need to get the name from the
		// path, otherwise we use the "file_name" as is
		std::string name = std::filesystem::path(file_name).stem().string();
		// translate to name without extension

		auto it = angle_name_to_idx_.find(name);
		if (it != angle_name_to_idx_.end()) {
			return it->second;
		}
		Table table(TabulatedType::Angle);
		std::string resolved_path =
			config_file_path.empty()
				? std::string(file_name)
				: resolve_file_path(std::string(file_name), std::string(config_file_path));
		table.read_file(resolved_path, name);
		angle_functions_.push_back(std::move(table));
		angle_name_to_idx_[name] = static_cast<int>(angle_functions_.size());
		return angle_name_to_idx_.size();
	}

	HOST int get_or_load_bond(std::string_view file_name, std::string_view config_file_path = "") {
		// file name could be a path or a name, if it is a path, we need to get the name from the
		// path, otherwise we use the "file_name" as is
		std::string name = std::filesystem::path(file_name).stem().string();

		auto it = bond_name_to_idx_.find(name);
		if (it != bond_name_to_idx_.end()) {
			return it->second;
		}
		Table table(TabulatedType::Bond);
		std::string resolved_path =
			config_file_path.empty()
				? std::string(file_name)
				: resolve_file_path(std::string(file_name), std::string(config_file_path));
		table.read_file(resolved_path, name);
		bond_name_to_idx_[name] = static_cast<int>(bond_name_to_idx_.size());
		bond_functions_.push_back(std::move(table));
		return bond_name_to_idx_[name];
	}

	HOST int get_or_load_dihedral(std::string_view file_name,
								  std::string_view config_file_path = "") {
		std::string name = std::filesystem::path(file_name).stem().string();
		auto it = dihedral_name_to_idx_.find(name);
		if (it != dihedral_name_to_idx_.end()) {
			return it->second;
		}
		Table table(TabulatedType::Dihedral);
		std::string resolved_path =
			config_file_path.empty()
				? std::string(file_name)
				: resolve_file_path(std::string(file_name), std::string(config_file_path));
		table.read_file(resolved_path, name);
		table.is_periodic = true;
		dihedral_functions_.push_back(std::move(table));
		dihedral_name_to_idx_[name] = static_cast<int>(dihedral_functions_.size() - 1);
		return dihedral_name_to_idx_[name];
	}

	int load_pair_nonbonded(int type_id_1, int type_id_2, std::string_view file_name) {
		type_id_1 = std::min(type_id_1, type_id_2);
		type_id_2 = std::max(type_id_1, type_id_2);
		std::string name = std::filesystem::path(file_name).stem().string();
		// check if the pair nonbonded already exists
		auto it =
			std::find_if(pair_nonbonded_types_.begin(),
						 pair_nonbonded_types_.end(),
						 [type_id_1, type_id_2](const PairNonBonded& pair) {
							 return pair.type_id_1 == type_id_1 && pair.type_id_2 == type_id_2;
						 });
		if (it != pair_nonbonded_types_.end()) {
			LOGWARN("PairNonBonded of type {} and {} already exists: {}",
					type_id_1,
					type_id_2,
					name);
			return it->function_index;
		}
		PairNonBonded pair_nonbonded(type_id_1, type_id_2, name);

		Table table(TabulatedType::NonBondedPair);
		table.read_file(file_name);
		nonbonded_functions_.push_back(std::move(table));
		pair_nonbonded.function_index = static_cast<int>(nonbonded_functions_.size() - 1);
		pair_nonbonded_types_.push_back(pair_nonbonded);
		return pair_nonbonded_types_.size() - 1;
	}
	const std::unordered_map<std::string, int>& get_angle_name_to_idx() const {
		return angle_name_to_idx_;
	}
	const std::unordered_map<std::string, int>& get_dihedral_name_to_idx() const {
		return dihedral_name_to_idx_;
	}
	const std::unordered_map<std::string, int>& get_bond_name_to_idx() const {
		return bond_name_to_idx_;
	}
	const std::vector<Table>& get_angle() const {
		return angle_functions_;
	}
	const std::vector<Table>& get_dihedral() const {
		return dihedral_functions_;
	}
	const std::vector<Table>& get_bond() const {
		return bond_functions_;
	}

	const std::vector<PairNonBonded>& get_pair_nonbonded_types() const {
		return pair_nonbonded_types_;
	}
	const std::vector<Table>& get_nonbonded() const {
		return nonbonded_functions_;
	}
	void set_resources(const std::vector<Resource>* resources) {
		resources_ = resources;
		if (resources_) {
			LOGINFO("TablesRegistry: Configured for {} resources", resources_->size());
		}
	}
	void build_device_arrays() {
		if (!resources_ || resources_->empty()) {
			LOGWARN("TablesRegistry: No resources configured, using default resource");
			// Create temporary default resource
			std::vector<Resource> default_resources = {Resource{}};
			build_device_arrays_impl(default_resources);
			return;
		}
		build_device_arrays_impl(*resources_);
	}
	const DeviceBuffer<arbd_real>& get_device_bond_buffer(size_t resource_idx,
														  size_t table_idx) const {
		return device_tabulated_bond_y_buffers_[resource_idx][table_idx];
	}

	const DeviceBuffer<arbd_real>& get_device_angle_buffer(size_t resource_idx,
														   size_t table_idx) const {
		return device_tabulated_angle_y_buffers_[resource_idx][table_idx];
	}

	const DeviceBuffer<arbd_real>& get_device_dihedral_buffer(size_t resource_idx,
															  size_t table_idx) const {
		return device_tabulated_dihedral_y_buffers_[resource_idx][table_idx];
	}
	const DeviceBuffer<arbd_real>& get_device_nonbonded_buffer(size_t resource_idx,
															   size_t table_idx) const {
		return device_tabulated_nonbonded_y_buffers_[resource_idx][table_idx];
	}

	const std::vector<Table>& get_bond_functions() const {
		return bond_functions_;
	}
	const std::vector<Table>& get_angle_functions() const {
		return angle_functions_;
	}
	const std::vector<Table>& get_dihedral_functions() const {
		return dihedral_functions_;
	}
	const std::vector<Table>& get_nonbonded_functions() const {
		return nonbonded_functions_;
	}

  private:
	void build_device_arrays_impl(const std::vector<Resource>& resources) {
		LOGINFO("TablesRegistry: Building device arrays for {} tables across {} resources",
				angle_functions_.size() + dihedral_functions_.size() + bond_functions_.size() +
					nonbonded_functions_.size(),
				resources.size());

		// Clear existing device data
		device_tabulated_bond_y_buffers_.clear();
		device_tabulated_angle_y_buffers_.clear();
		device_tabulated_dihedral_y_buffers_.clear();
		device_tabulated_nonbonded_y_buffers_.clear();

		// Resize per-resource vectors
		device_tabulated_bond_y_buffers_.resize(resources.size());
		device_tabulated_angle_y_buffers_.resize(resources.size());
		device_tabulated_dihedral_y_buffers_.resize(resources.size());
		device_tabulated_nonbonded_y_buffers_.resize(resources.size());

		// Build device arrays for each resource
		for (size_t res_idx = 0; res_idx < resources.size(); ++res_idx) {
			const auto& resource = resources[res_idx];

			LOGINFO("TablesRegistry: Transferring tables to resource {}", res_idx);

			// Transfer bond function tables
			for (size_t i = 0; i < bond_functions_.size(); ++i) {
				const auto& table = bond_functions_[i];
				if (!table.X.empty() && !table.Y.empty()) {
					// Only transfer Y values for uniform step tables (X can be calculated
					// on-device)
					device_tabulated_bond_y_buffers_[res_idx].emplace_back(table.Y.size(),
																		   resource);
					device_tabulated_bond_y_buffers_[res_idx][i].copy_from_host(table.Y.data(),
																				table.Y.size());

					// X values not needed on device for uniform step tables
					// TabulatedPotential uses step_inv and start to calculate X on-the-fly
				}
			}

			// Transfer angle function tables
			for (size_t i = 0; i < angle_functions_.size(); ++i) {
				const auto& table = angle_functions_[i];
				if (!table.X.empty() && !table.Y.empty()) {
					// Only transfer Y values for uniform step tables
					device_tabulated_angle_y_buffers_[res_idx].emplace_back(table.Y.size(),
																			resource);
					device_tabulated_angle_y_buffers_[res_idx][i].copy_from_host(table.Y.data(),
																				 table.Y.size());
				}
			}

			// Transfer dihedral function tables
			for (size_t i = 0; i < dihedral_functions_.size(); ++i) {
				const auto& table = dihedral_functions_[i];
				if (!table.X.empty() && !table.Y.empty()) {
					// Only transfer Y values for uniform step tables
					device_tabulated_dihedral_y_buffers_[res_idx].emplace_back(table.Y.size(),
																			   resource);
					device_tabulated_dihedral_y_buffers_[res_idx][i].copy_from_host(table.Y.data(),
																					table.Y.size());
				}
			}

			// Transfer nonbonded function tables
			for (size_t i = 0; i < nonbonded_functions_.size(); ++i) {
				const auto& table = nonbonded_functions_[i];
				if (!table.X.empty() && !table.Y.empty()) {
					// Only transfer Y values for uniform step tables
					device_tabulated_nonbonded_y_buffers_[res_idx].emplace_back(table.Y.size(),
																				resource);
					device_tabulated_nonbonded_y_buffers_[res_idx][i].copy_from_host(
						table.Y.data(),
						table.Y.size());
				}
			}
		}

		LOGINFO("TablesRegistry: Device arrays built successfully for all resources");
	}

	std::unordered_map<std::string, int> angle_name_to_idx_;
	std::unordered_map<std::string, int> dihedral_name_to_idx_;
	std::unordered_map<std::string, int> bond_name_to_idx_;
	std::vector<PairNonBonded> pair_nonbonded_types_;
	std::vector<Table> angle_functions_;
	std::vector<Table> dihedral_functions_;
	std::vector<Table> bond_functions_;
	std::vector<Table> nonbonded_functions_;
	const std::vector<Resource>* resources_ = nullptr;

	// Device buffers per resource: [resource_idx][table_idx] = buffer
	// Only Y values stored on device - X values calculated from step_inv and start
	std::vector<std::vector<DeviceBuffer<arbd_real>>> device_tabulated_bond_y_buffers_;
	std::vector<std::vector<DeviceBuffer<arbd_real>>> device_tabulated_angle_y_buffers_;
	std::vector<std::vector<DeviceBuffer<arbd_real>>> device_tabulated_dihedral_y_buffers_;
	std::vector<std::vector<DeviceBuffer<arbd_real>>> device_tabulated_nonbonded_y_buffers_;
};
} // namespace ARBD
