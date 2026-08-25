#pragma once
/**
 * @file System/SystemForward.h
 * @brief Forward declarations for System classes to break circular dependencies
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @version 2.0
 * @date 2025-12-12
 */

#include <memory>
namespace MARS {

// Forward declarations
class SimSystem;
class SystemState;
class PatchManager;
class PatchDecomposer;

enum class DecomposerType;

// Factory function declaration (implemented in SimSystem.cpp)
std::unique_ptr<PatchDecomposer> create_patch_decomposer(DecomposerType type);

} // namespace MARS
