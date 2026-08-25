# BondedInteraction.h
  When non-empty, SimSystem::build_name_to_id_maps() resolves these into type_id_1/type_id_2 (see resolve_type_names below); the legacy `tabulatedFile i@j@file` config path supplies ids directly and leaves these empty. Safe to defer because the ids aren't read until Patch::calculate_nonbonded_forces builds the type-pair matrix on the first  step, long after type ids are assigned.

# Interactions.h

## CalcDistance overloads

`compute(const Vector3& from, const Vector3& to, const PeriodicBox*)` is the primitive; `compute(const Vector3* positions, const int2&, const PeriodicBox*)` forwards to it with `positions[x]`, `positions[y]`, preserving the original `y - x` direction.

The two-point form exists for `HarmonicRestraintComputer`, whose anchor is a fixed point rather than a particle, so there is no index to dereference — but the minimum-image wrap, the length, and the 1e-6 unit-vector guard are the same work. It lives in `Bonded/BondComputer.h`; see `Bonded/dev_notes.md`.

# DeviceBondedInteraction.h

## Launch helpers

The `launch_*` helpers for bonds, angles, dihedrals, analytical bonds and
harmonic restraints live at the bottom of this header (moved out of
`Bonded/BondComputer.h` on 2026-08-15). Each pairs a device functor with the
`DeviceBondedInteractions` buffers it reads, which is why they sit next to those
buffers rather than next to the functors.

`restraint_spring_constants()` returns `const mars_real*`, matching
`DeviceBuffer<mars_real> restraint_spring_constants_`. It was declared
`const float*` and only compiled because `mars_real` is `float` (`Header.h`).

# TabulatedPotential.h
If table is not periodic->  Past the end of the table the potential is held at its final value, so the force there is zero. Matches legacy MARS's TabulatedPotential::compute, which returns `EnergyForce(v0[n-1], Vector3(0.0f))` for `home >= n` and yields `du = 0` in the final bin. Clamping `home` alone (the previous behaviour here) is not enough: `w` keeps growing with `dx`, so `dU*w + U0` below linearly extrapolates the last bin's slope instead of holding the endpoint value, and the returned force stays at that slope rather than falling to zero. For a nonbonded table that ends at the interaction cutoff, every out-of-range pair then acquires a spurious constant force and an unbounded energy - a non-decaying long-range interaction applied to whatever fraction of the pair list lies beyond the cutoff.
Below the table, clamp the index but keep the fractional weight,  so the first bin's slope still applies. Legacy does the same (`home = home < 0 ? 0 : home`). Returning zero force here instead would delete the repulsive core that keeps particles apart.
`force_magnitude` must be -dU/dr (not the raw energy slope): the bond/angle/dihedral/nonbonded computers all apply it via the same "-force to particle x, +force to particle y" convention that AnalyticalForceComputer uses with an already-negated force (e.g. harmonic bond's force = -k*(distance-r0)). Returning +dU/dr here inverted every tabulated potential well into a force hilltop.

## DeviceBondedInteraction.h — Z-order reorder (ENABLE_ZORDER_REORDER)

Stage 3 of `buzzing-tickling-music.md`. All gated on the flag; flag-off is a no-op.

`remap_particle_indices<Sorter>` fixes every particle-slot reference after a Morton
reorder — bonds (int2), angles (int3), dihedrals (int4), restraints (int). These are
remapped **in place on device** by reinterpreting the int2/int3/int4 buffer as a flat
int array and calling `sorter.remap_indices` (indices[i] = inv[indices[i]]). Composes
across repeated reorders: each buffer is in current slot order, inv maps current→new.

**Exclusions can't be remapped in place** — the CSR (excl_offsets_/excl_neighbors_) is
variable-length per particle. `rebuild_exclusions_after_reorder` remaps the retained
canonical edge list (`excl_pairs_host_`, kept in sync each reorder), re-canonicalizes
(swap so a<b), re-sorts, and rebuilds the CSR with the exact same host logic as
copy_from_host. num_excl_particles_ and the neighbor total can change, so offsets/
neighbors buffers are reallocated. Host roundtrip is fine at reorder cadence.

Template method so ZOrderSort.h stays out of this header (instantiated in Patch.cpp).
