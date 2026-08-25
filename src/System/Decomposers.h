#pragma once
/**
 * @file System/Decomposers.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Factory and includes for all patch decomposer implementations
 * @version 2.0
 * @date 2025-09-09
 *
 * This file provides a unified interface to all decomposer implementations
 * and factory functions to avoid circular dependency issues.
 *
 * @copyright Copyright (c) 2025
 */

#include "System/PatchDecomposer.h"

// Include all concrete decomposer implementations
#include "PatchOperation/ZOrderKernels/ZOrderKernels.h"
#include "System/PatchDecomposer/Gemoetric.h" // Note: keeping original filename
#include "System/PatchDecomposer/RecursiveBisection.h"
#include "System/PatchDecomposer/Spatial.h"
#include "System/PatchDecomposer/ZOrder.h"

namespace MARS {

//================================================================================
// Factory Functions
//================================================================================

/**
 * @brief Factory function to create decomposer by type
 * @param type Decomposer type to create
 * @return Unique pointer to decomposer
 */
std::unique_ptr<PatchDecomposer> create_decomposer(DecomposerType type);

/**
 * @brief Get all available decomposer types
 * @return Vector of supported decomposer types
 */
std::vector<DecomposerType> get_available_decomposer_types();

/**
 * @brief Choose optimal decomposer based on system characteristics
 * @param system Simulation system to analyze
 * @param state Current system state
 * @return Recommended decomposer type
 */
DecomposerType recommend_decomposer_type(const SimSystem& system, const SystemState& state);

/**
 * @brief Check if decomposer type is available
 * @param type Decomposer type to check
 * @return True if type is supported
 */
bool is_decomposer_available(DecomposerType type);

/**
 * @brief Get human-readable name for decomposer type
 * @param type Decomposer type
 * @return Descriptive name
 */
const char* get_decomposer_name(DecomposerType type);

} // namespace MARS
