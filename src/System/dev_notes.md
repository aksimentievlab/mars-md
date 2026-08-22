# src/System — implementation notes

## RigidBodyManager.h

### DLM must not run substeps 0-2 in one call

`integrate_motion(dt, box)` defaults to running all three substeps, which is
correct only for callers that do not care about energy conservation. The
simulation loop must instead call `integrate_drift()` (substeps 0-1), evaluate
forces, then `integrate_kick()` (substep 2).

Legacy is laid out that way: `GrandBrownTown.cu:837-838` runs `integrateDLM(0)`
and `(1)`, `:953` clears, `:1097` re-evaluates forces, `:1130-1131` redraws the
Langevin noise, `:1132` runs `integrateDLM(2)`. Forces are primed before the
loop at `:694` / `:775` / `:789-790`.

Running 0-2 back to back leaves both half-kicks on a pre-drift force. For a
harmonic mode that map is `[[1 - w^2 dt^2/2, dt], [-w^2 dt, 1]]`, determinant
`1 + w^2 dt^2 / 2 > 1` — it pumps phase-space volume at a rate set by that
mode's own stiffness. Not a second force evaluation; the evaluation just moves
between substeps 1 and 2.

**Known sharp edge:** `first`/`last` are unvalidated, and
`RBIntegrateDLMKernel` treats every substep other than 0 and 2 as the drift
branch. So `integrate_motion(dt, box, 3, 3)` silently performs a full
drift-and-rotate rather than erroring. The defaults also preserve the
non-Verlet ordering for any caller that does not split.

### `prepare_grid_grid_dispatch` — why candidates are precomputed

Type-level force pairs (Phase 3) are expanded into concrete RB-instance pairs
once, since RB counts and types are static for a run. Only the per-step distance
cull then needs to run on-device (`RBGridCullKernel`). Call once after
`initialize()` and after `grid_manager.build_device_arrays()`; call again if
either changes.

Worklist capacity is set to exactly `num_candidates`: culling only removes
candidates, never adds them, so this is an exact upper bound with no wasted
allocation and, in the intended usage, no possible overflow.
`RBGridCullKernel`'s overflow flag is a safety net, exercised directly in
`RigidBodyGridBatch`'s own tests.

`grid_resource_idx` is independent of `compute_resource_idx_` because the two
managers may be indexed differently until Phase 5 wires them through the same
`SimSystem` resource list.

### `compute_grid_grid_forces` — why total_blocks never round-trips to the host

Three kernels run back to back on one stream (cull -> prefix sum -> batched
force) with no host sync between them. Kernel B's launch grid is sized to the
worklist *capacity*, an exact host-known upper bound (see
`prepare_grid_grid_dispatch`), and blocks beyond the real device-computed
`total_blocks` early-return as a cheap no-op. So `total_blocks` never needs a
D2H trip.

The RB-RB distance cutoff is not applied to type-PMF terms, which always
evaluate — an external field has no "distance" to the body it acts on.

## RigidBodyManager.h — Z-order reorder (ENABLE_ZORDER_REORDER)

Stage 3 of `buzzing-tickling-music.md`. Gated on the flag; flag-off is a no-op.

`remap_attached_particle_indices<Sorter>` fixes each attached particle's
`particle_index` (its slot in the patch's arrays) after a Morton reorder. Because `particle_index` is a strided field inside `RBAttachedParticle` (not a standalone int buffer), the flat-int `remap_indices` can't touch it — so this host-roundtrips:
copy `attached_` down, `a.particle_index = inv[a.particle_index]`, copy back.
Attached count is small and this runs at reorder cadence, so the roundtrip is cheap.
Template method to keep ZOrderSort.h out of the header (instantiated in Patch.cpp).
