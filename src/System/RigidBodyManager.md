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
