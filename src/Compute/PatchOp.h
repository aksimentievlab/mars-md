/*********************************************************************
 * @file  PatchOp.h
 *
 * @brief Declaration of BasePatchOp class.
 *
 * @details This file contains the declaration of the abstract base
 *          class BasePatchOp, which operates on Patch data. It also
 *          includes headers of derived classes for convenient access
 *          to factory methods. The virtual method
 *          `BasePatchOp::compute(Patch* patch)` is called by
 *          `patch->compute()` by any Patch to which the PatchOp has
 *          been added.
 *********************************************************************/

#pragma once

namespace ARBD {
class Patch;

// describes what should be computed in the system, invariant to changes in
// decomposition, but doesn't do the work directly. That is left to PatchOp,
// created from ComputeSpec by a Decomposer.
class SymbolicOp {
public:
  // TODO: implement
};

/**
 * @brief Abstract base class that operates on Patch data.
 *
 * @details PatchOp is an abstract base class for all classes that
 *          operate on Patch data.  It provides the `compute()`
 *          method, which is called by `Patch::compute()` for each
 *          PatchOp attached to a Patch.
 */
class PatchOp {
public:
  /**
   * @brief Performs the computation for this PatchOp on the given Patch.
   *
   * @param patch The Patch on which the computation should be performed.
   */
  virtual void compute(Patch *patch) = 0;

  /**
   * @brief Returns the number of patches that this PatchOp requires.
   *
   * @return The number of patches required by this PatchOp.
   */
  virtual int num_patches() const = 0;

private:
  void *compute_data;
};

} // namespace ARBD
