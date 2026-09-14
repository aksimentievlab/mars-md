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

**Exclusions can't be remapped in place** — the Compressed Sparse Row (CSR) (excl_offsets_/excl_neighbors_) is variable-length per particle. `rebuild_exclusions_after_reorder` remaps the retained canonical edge list (`excl_pairs_host_`, kept in sync each reorder), re-canonicalizes (swap so a<b), re-sorts, and rebuilds the CSR with the exact same host logic as
copy_from_host. num_excl_particles_ and the neighbor total can change, so offsets/
neighbors buffers are reallocated. Host roundtrip is fine at reorder cadence.

Template method so ZOrderSort.h stays out of this header (instantiated in Patch.cpp).

## DeviceExclusions.h

`ExclusionView` is the borrowed device view of the exclusion CSR that
`DeviceBondedInteractions` owns (`excl_offsets_` / `excl_neighbors_`). It sits
here rather than under `PatchOperation/` because the storage and its reorder
rebuild are owned here; the pairlist only borrows the pointers, and they must
stay valid until the next build.

Consumers and the reasoning behind filtering at pairlist-emit time are in
`PatchOperation/PairListKernels/dev_notes.md`.

## 2026-09-12 — rigid-body exclusions use an O(1) same-body test, not the CSR

v1 excludes every pair of particles attached to the same rigid body, unconditionally, with no
cutoff and no bond graph (`arbd/src/Configuration.cpp`, the block before the second
`printf("Built %d exclusions.")`). On npc_6enl_beads that is C(437,2) = 95,266 pairs; v2 was
generating none of them, so it evaluated nonbonded forces on 27% of pairs v1 drops.

**Do not route this through the exclusion CSR.** `ExclusionView::row_contains` is a linear scan,
called per in-cutoff candidate. The all-pairs rule gives each of the 437 attached particles a
436-entry row, so every candidate touching an attached particle would cost up to 436 comparisons.
The CSR sizing comment in `DeviceBondedInteraction.h` assumes "~degree(i), typically 1-2 for a
linear polymer" -- off by ~200x. `rebuild_exclusions_after_reorder` would also do a host
round-trip and re-sort of all 95,266 pairs at reorder cadence.

Instead `ExclusionView` carries `rigid_body_id`, the per-particle RB instance id (-1 unattached),
and the pairlist drops a pair when both endpoints share a non-negative id. `body_of()` hoists out
of the neighbour loop next to `row_begin`/`row_end`; only the partner's id is a per-candidate
gather, alongside the `sorted_to_original[j]` load already there. Semantically identical to v1,
O(1), no CSR growth, and the reorder path needs nothing new.

A null `rigid_body_id` disables the test, and `body_of()` returns -1 in that case so `same_body()`
short-circuits before dereferencing. That null is also the off switch for
`Patch::set_exclude_rigid_body_attached` (default on).

The host side needed no new plumbing: `ConfigParser::fold_in_attached_particles` already sets
`ParticleIO::attached_rigid_body_id = rb.id`, `rb.id` is assigned per *instance* (not per type),
`SystemState` pushes it into the global SoA, and `SystemState.cpp` carries it through the Z-order
permutation. What was missing was only the upload: `DeviceParticle` now owns the buffer and
`Patch::copy_particles_from_host` copies the column. Absent host data fills -1 rather than
leaving a stale buffer.

Verification is by pair count, not by any log line: this design adds no CSR entries, so the
"Prepared N exclusions" message is unchanged. Toggle the flag and the emitted pair count must
differ by exactly the intra-body pair total.

## 2026-09-14 — the nonbonded pair list lives in NonBondedInteractions, not TablesRegistry

`NonBondedInteractions` was dead: `Patch::calculate_nonbonded_forces` took it and never read
it, `prepare_device_data`/`cleanup_device_data` were empty, `LongRangeNonBonded` had no consumer,
and the pair list Patch really used was `TablesRegistry::pair_nonbonded_types_`. So "which type
pairs interact" had ended up inside the table cache, and an analytical pair (no table at all)
could not be declared: `PairNonBonded::set_function` recognised only `AnalyticalNameList =
{"LJ"}` while the device encoder (`analytical_term_bit`) knew `coulomb`, `debye_huckel`, `onck`,
`gaussian`, `softcore`. On top of that `Patch.cpp` passed a hardcoded `PAIR_TERM_TABULATED` as
the enabled mask, so a correctly tagged analytical pair would still have been skipped.

Now, mirroring the bonded side (`BondedInteractions` owns terms, the registry owns tables):

- `TablesRegistry` holds tables only. `load_nonbonded(path)` / `add_nonbonded(Table)` return an
  index into `get_nonbonded()`, deduplicated by name (file stem) like the bonded loaders. The old
  `load_pair_nonbonded(i, j, path)` made a fresh table per pair even for the same file.
- `NonBondedInteractions` (owned by `SimSystem`) is the pair list. `PairNonBonded(i, j, name)`
  is analytical, `PairNonBonded(i, j, name, table_index)` tabulated; by type name too, resolved
  at `build_name_to_id_maps`. `add_pair_nonbonded` keeps the first declaration of a pair.
- One name list, `pair_term_from_name` in `NonBondedInteraction.h` (host-only; `Pairwise.h` is
  device code shared with Metal and stays free of `std::string`). The device encoder calls it.
- `Patch` builds the type-pair matrix from `interactions.get_pair_nonbonded()` and passes
  `interactions.enabled_terms()` as the mask. Tabulated-only runs get the same mask as before.

Physics of what a declared pair gets: the device tag carries only that pair's own bits, and the
kernel applies `tag & enabled_terms`, so a tabulated pair never picks up an analytical term and
a tabulated-only run is bit-for-bit what it was. `coulomb` is unscreened vacuum Coulomb,
`332.0636 qi qj / r` kcal/mol with charges in e (`constants::COULOMB`), and `softcore` takes
`eps`/`radius` per particle type. The screened terms need solvent constants that are neither
per type nor derivable in the kernel: Debye-Huckel λ and ε, Onck κ, Sz and z. One solvent per
system, so they are global, in `SolventParams` on the pair list
(`NonBondedInteractions::set_solvent_params`), and `launch_pairwise_nonbonded` copies them onto
the functors before launch. Before this the functor defaults (λ = 10 Å, ε = 80; κ = 0.1,
z = 6.86) were the only values a run could ever use, and `SimSystem::salt_concentration_` was
stored and read nowhere. `SolventParams` defaults equal the functor defaults, so nothing moves
for a run that never sets them. Deriving λ from salt and temperature is the caller's job
(Python: 3.04 Å / sqrt(I[M]) at 298 K); the engine does not guess.

Open: `gaussian` amp and σ are pair properties (well depth ε_ij, width σ_ij), not solvent
constants, and the kernel still reads them from the functor defaults (amp = 1, σ = 1,
no cutoff). Making them per pair needs a per-pair parameter matrix beside the tag matrix in
`DevicePairNonBondedInteractions`, filled from fields on `PairNonBonded`, and
`GaussianPotential::compute` in `Pairwise.h` (device code, shared with Metal) taking them from
it. Until then a `gaussian` pair is only good for exercising the kernel.

`load_nonbonded` dedupes by resolved path, not by file stem: two different `nb.dat` files in
different directories are different potentials. The bonded loaders still key by stem.

`NonBondedInteraction.cpp` is an empty stub to be removed together with its CMake entry.
