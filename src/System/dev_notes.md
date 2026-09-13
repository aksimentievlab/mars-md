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
# RigidBodyManager.h — implementation notes

Owns all rigid-body device state and drives its per-step physics. Phase 4 of the
rigid-body suite, on top of Phase 2's SoA device storage
(`DeviceRigidBody`/`DeviceRigidBodyTypes`) and Phase 3's type-level force-pair
list (`RigidBodyForcePairList`).

## Why it lives beside PatchManager, not inside it

Rigid bodies are global and not spatially decomposed: force-pairing is by
grid-key match rather than proximity, and an RB's density grid sometimes spans
multiple patches. Neither fits the patch model.

## Resources

Takes a resource vector + `compute_resource_idx` even though only
`resources.size() == 1` is exercised today. The grid-grid math stays centralized
on `compute_resource()`; a second resource would need only position/orientation
broadcast for grid-particle forces, via `broadcast_state_to_resources()`.

## Batched grid-grid dispatch

`prepare_grid_grid_dispatch()` / `compute_grid_grid_forces()` — see
`Interactions/Nonbonded/RigidBodyGridBatch.h`.

## gather_to_host

Refreshes a host `HostRigidBodyData` from the device instance state. The
instance count is fixed for the run, so the destination is sized once and
overwritten in place — the trajectory writer used to allocate a fresh SoA and
resize it on every output frame.

It takes the SoA rather than a `SystemState&` on purpose. Nothing about
rigid-body physics depends on `SystemState`, and keeping that dependency out is
what lets `RBOperation` (which pulls in scuff-em) and the trajectory writers
share one read-back path without dragging `SimSystem` along. `PatchManager::
gather_particles_to_state(SystemState&)` takes the opposite shape; the
difference is deliberate, not an oversight.

Callers pair it with `SystemState::mark_rigid_bodies_synced()`.

## 2026-09-12 — single-patch plan hard-coded the box origin to (0,0,0)

`SimSystem`'s single-patch decomposition plan filled `patch_min_bounds` with a literal
`Vector3(0,0,0)` and `patch_max_bounds` with the box *size*, rather than the box origin and
`origin + size`. `DecompositionPlan::set_periodic_box()` then saw `have_bounds == true`, skipped
the branch that derives bounds from `system_box.get_origin()`, and did
`pbox.set_origin(patch_min_bounds[0])` — so every patch got `origin = (0,0,0)`.

Nothing upstream was wrong: `ConfigParser` -> `SimSystem::set_origin` -> `sim_box_.set_origin`
all carry the configured origin, and `plan.system_box = sim_box_` copies it. Those four literals
were the only break, and they suppressed the code that would otherwise have been correct.

### Symptom: silently non-periodic neighbour lists

On cytoplasm (`origin -400`, `systemSize 800`, fully periodic) the pairlist reproduced the
**open** neighbour list exactly and found **zero** cross-boundary pairs:

| | pairs | vs scipy cKDTree |
|---|--:|---|
| before | 55,928,261 | matches *open* count (56,231,831 - 303,567 exclusions), 3 short |
| after | **55,929,837** | matches *periodic* count (56,233,409 - 303,567), 5 short |

1,576 pairs recovered — every one a missing force term at a box face.

Mechanism: `build_pairlist` sets the Morton domain to `[origin, origin+size)` on periodic axes.
With `origin=(0,0,0)` that is `[0,800)` while the particles occupy `[-400,400)`. Negative
coordinates clamp into edge cells, which *preserves* the open list — clamped particles share a
cell, cells are always searched, and spurious candidates die on the distance test — but destroys
periodic adjacency. A particle at `z=-393.7` clamps to cell 0; its true wrapped partner at
`z=+390.3` sits in cell 7; cells 0 and 7 are never stencil neighbours.

`wrap_diff` only needs the box *length*, not the origin, which is why forces were otherwise sane
and only the boundary shell was affected. Anything calling `PeriodicBox::wrap()` (absolute
wrapping) against these patch boxes was working in the wrong image.

### Why it hid for so long

- The deficit is 0.003% of the list and confined to a shell one cutoff thick at each face.
- It only appears when the box origin is not `(0,0,0)`.
- It only appears when particles actually approach a boundary. Here the x and y gaps across the
  periodic faces (61.1 A and 46.4 A) exceed the 45 A cutoff, so the deficit was almost entirely
  in z, whose gap is 16.0 A.
- `PLDIAG` did not print the origin. It does now.

### Related, still open

`PatchManager::initialize_local_patches` (`PatchManager.cpp`) passes a `Vector3` where `Patch`
expects a `PeriodicBox`, compiling only through the non-explicit
`PeriodicBox(const Vector3&, bool=true, bool=true, bool=true)` converting constructor, and drops
the origin the same way. It currently has **no callers**. Marking that constructor `explicit`
would turn this class of mistake into a compile error.

`ConfigParser`'s `periodicity_map` (`allperiodic`/`twodimensional`/`onedimensional`/`open`) is
declared and never used, so periodicity cannot be set from a config file at all. It defaults to
all-periodic via `SimSystem::sim_box_`, which is why this system was periodic despite no key.
