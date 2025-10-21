#include "Exclusion.h"
#include "BondedInteraction.h"
#include <queue>
#include <vector>

namespace ARBD {

// run this first, then read in extra exclusions from a file
std::vector<Exclude>
make_exclusions_cpu(int num_particles, const std::vector<Bond>& bonds, int exclusion_depth) {

	if (num_particles <= 0 || exclusion_depth <= 0) {
		return {}; // Return empty list if there's nothing to do
	}

	std::vector<std::vector<int>> adjacency_list(num_particles);
	for (const auto& bond : bonds) {
		adjacency_list[bond.ind1].push_back(bond.ind2);
		adjacency_list[bond.ind2].push_back(bond.ind1);
	}

	std::vector<Exclude> all_exclusions;

	// 2. Perform a Breadth-First Search (BFS) starting from each particle.
	for (int start_node = 0; start_node < num_particles; ++start_node) {
		std::queue<std::pair<int, int>> q; // Stores {particle_index, current_depth}
		q.push({start_node, 0});

		std::vector<bool> visited(num_particles, false);
		visited[start_node] = true;

		while (!q.empty()) {
			auto [current_node, current_depth] = q.front();
			q.pop();

			// Stop exploring if we've reached the desired depth
			if (current_depth >= exclusion_depth) {
				continue;
			}

			// Explore all neighbors of the current node
			for (int neighbor : adjacency_list[current_node]) {
				if (!visited[neighbor]) {
					visited[neighbor] = true;

					all_exclusions.push_back(
						{std::min(start_node, neighbor), std::max(start_node, neighbor)});

					// Add the neighbor to the queue to explore in the next level
					q.push({neighbor, current_depth + 1});
				}
			}
		}
	}

	// 3. Sort and remove duplicate exclusions.
	std::sort(all_exclusions.begin(), all_exclusions.end());
	all_exclusions.erase(std::unique(all_exclusions.begin(), all_exclusions.end()),
						 all_exclusions.end());

	return all_exclusions;
}
} // namespace ARBD
