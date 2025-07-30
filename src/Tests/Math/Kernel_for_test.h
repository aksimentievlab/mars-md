#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Math/Types.h"
#include "Math/Vector3.h"
#include "Random/Random.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct TransformKernel {
	HOST DEVICE void operator()(size_t i, const float* input, float* output) const {
		// Transform: y = 2*x + 1
		output[i] = 2.0f * input[i] + 1.0f;
	}
};

struct CombineKernel {
	HOST DEVICE void
	operator()(size_t i, const float* uniform, const float* gaussian, float* combined) const {
		// Simple combination: 70% uniform + 30% gaussian
		combined[i] = 0.7f * uniform[i] + 0.3f * gaussian[i];
	}
};

// template<typename... Args>
// HOST DEVICE void operator()(size_t i, Args... args) const {

// 	auto tuple_args = std::make_tuple(args...);
// 	auto* input = std::get<0>(tuple_args);
// 	auto* output = std::get<1>(tuple_args);
// ============================================================================
// Kernel Functors for Profiling Tests
// ============================================================================

struct InitializeWalkersKernel {
	HOST DEVICE void operator()(size_t i, ARBD::Vector3_t<float>* positions) const {
		positions[i] = ARBD::Vector3_t<float>{0.0f, 0.0f, 0.0f};
	}
};

struct RandomWalkKernel {
	size_t NUM_STEPS;
	size_t NUM_WALKERS;

	HOST DEVICE void operator()(size_t walker_id,
								const ARBD::Vector3_t<float>* steps,
								ARBD::Vector3_t<float>* positions) const {
		ARBD::Vector3_t<float> pos = positions[walker_id];

		// Take NUM_STEPS/NUM_WALKERS steps per walker
		size_t steps_per_walker = NUM_STEPS / NUM_WALKERS;
		size_t start_step = walker_id * steps_per_walker;

		for (size_t step = 0; step < steps_per_walker && (start_step + step) < NUM_STEPS; ++step) {
			size_t step_idx = start_step + step;
			ARBD::Vector3_t<float> step_vec = steps[step_idx];

			// Normalize step to unit length
			float length = std::sqrt(step_vec.x * step_vec.x + step_vec.y * step_vec.y +
									 step_vec.z * step_vec.z);
			if (length > 0.0f) {
				step_vec.x /= length;
				step_vec.y /= length;
				step_vec.z /= length;
			}

			// Take the step
			pos.x += step_vec.x;
			pos.y += step_vec.y;
			pos.z += step_vec.z;
		}

		positions[walker_id] = pos;
	}
};

struct CalculateDistancesKernel {
	HOST DEVICE void
	operator()(size_t i, const ARBD::Vector3_t<float>* positions, float* distances) const {
		ARBD::Vector3_t<float> pos = positions[i];
		distances[i] = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
	}
};
inline double calculate_correlation(const std::vector<float>& x, const std::vector<float>& y) {
	if (x.size() != y.size() || x.empty())
		return 0.0;

	double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
	double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
	double mean_x = sum_x / x.size();
	double mean_y = sum_y / y.size();

	double numerator = 0.0;
	double sum_sq_x = 0.0;
	double sum_sq_y = 0.0;

	for (size_t i = 0; i < x.size(); ++i) {
		double dx = x[i] - mean_x;
		double dy = y[i] - mean_y;
		numerator += dx * dy;
		sum_sq_x += dx * dx;
		sum_sq_y += dy * dy;
	}

	double denominator = std::sqrt(sum_sq_x * sum_sq_y);
	return (denominator > 1e-10) ? (numerator / denominator) : 0.0;
}
struct SimpleKernel {
	HOST DEVICE void operator()(size_t i, const float* input, float* output) const {
		output[i] = static_cast<float>(i);
	}
};
struct SmoothingFilterKernel {
	size_t GRID_SIZE;

	HOST DEVICE void operator()(size_t i, const float* input, float* output) const {
		size_t x = i % GRID_SIZE;
		size_t y = i / GRID_SIZE;

		// Simple 3x3 averaging filter
		float sum = 0.0f;
		int count = 0;

		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				int nx = static_cast<int>(x) + dx;
				int ny = static_cast<int>(y) + dy;

				if (nx >= 0 && nx < static_cast<int>(GRID_SIZE) && ny >= 0 &&
					ny < static_cast<int>(GRID_SIZE)) {

					size_t idx = ny * GRID_SIZE + nx;
					sum += input[idx];
					count++;
				}
			}
		}

		output[i] = (count > 0) ? sum / count : input[i];
	}
};

struct GradientCalculationKernel {
	size_t GRID_SIZE;

	HOST DEVICE void operator()(size_t i, const float* input, float* output) const {
		size_t x = i % GRID_SIZE;
		size_t y = i / GRID_SIZE;

		float grad_x = 0.0f, grad_y = 0.0f;

		// Calculate finite difference gradients
		if (x > 0 && x < GRID_SIZE - 1) {
			size_t left_idx = y * GRID_SIZE + (x - 1);
			size_t right_idx = y * GRID_SIZE + (x + 1);
			grad_x = (input[right_idx] - input[left_idx]) * 0.5f;
		}

		if (y > 0 && y < GRID_SIZE - 1) {
			size_t top_idx = (y - 1) * GRID_SIZE + x;
			size_t bottom_idx = (y + 1) * GRID_SIZE + x;
			grad_y = (input[bottom_idx] - input[top_idx]) * 0.5f;
		}

		output[i] = std::sqrt(grad_x * grad_x + grad_y * grad_y);
	}
};
