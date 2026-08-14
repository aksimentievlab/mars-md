#include "BondedInteraction.h"
#include <queue>
#include <vector>

namespace ARBD {

// run this first, then read in extra exclusions from a file
void BondedInteractions::make_exclusions(int num_particles, int exclusion_depth) {

	std::vector<std::vector<int>> adjacency_list(num_particles);
	for (const auto& bond : bonds_) {
		adjacency_list[bond.ind1].push_back(bond.ind2);
		adjacency_list[bond.ind2].push_back(bond.ind1);
	}

	// 2. Perform a Breadth-First Search (BFS) starting from each particle.
	for (int start_node = 0; start_node < num_particles; ++start_node) {
		std::queue<std::pair<int, int>> q; ///< Stores {particle_index, current_depth}
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

					exclusions_.push_back(
						{std::min(start_node, neighbor), std::max(start_node, neighbor)});

					// Add the neighbor to the queue to explore in the next level
					q.push({neighbor, current_depth + 1});
				}
			}
		}
	}
	// 3. Sort and remove duplicate exclusions.
	std::sort(exclusions_.begin(), exclusions_.end());
	exclusions_.erase(std::unique(exclusions_.begin(), exclusions_.end()), exclusions_.end());
}
} // namespace ARBD
