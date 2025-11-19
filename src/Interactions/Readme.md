# Interactions

This module defines both **bonded** and **nonbonded** interactions for the simulation framework. Interactions are implemented as modular classes, following the `PatchOp` interface, and are responsible for computing forces and energies between objects (e.g., particles, rigid bodies) within a patch.

## Structure

- **Bonded Interactions:**
  Handle forces between particles connected by bonds, angles, or dihedrals (e.g., Harmonic, Tabulated, Lennard-Jones).
- **Nonbonded Interactions:**
  Handle pairwise or long-range interactions not requiring explicit bonds (e.g., Lennard-Jones, Coulomb).

## Implementation Notes

- All interaction types inherit from `LocalInteraction` (see [`Interaction.h`](Interaction.h)), which provides a common interface and configuration structure.
- Interactions are designed to be backend-agnostic, with backend-specific kernels implemented in the appropriate backend folders (see `src/Backend/`).
- Factory methods are provided for instantiating interactions based on configuration.
- Header-only implementation is preferred for lightweight and extensible interaction definitions, but backend-specific kernels may require separate compilation units.

## See Also

- [`src/Interactions/Interaction.h`](Interaction.h) — Base class and configuration
- [`src/Backend/`](../Backend/) — Backend-specific kernel implementations
