# dev_notes — PatchOperation

## Z-order canonical reorder (ENABLE_ZORDER_REORDER) — Stage 4 orchestration

Plan: `buzzing-tickling-music.md`. Off by default; double-gated by the compile flag
AND runtime `reorder_period > 0` (SimSystem, config key `reorderPeriod`). Flag off =
identical binary.

### Pieces
- `Patch::reorder_particles()` — sorts `particles_.pos()` with a Patch-owned
  `ZOrderSort` (System mode, lazily built), permutes the SoA, remaps `device_bonded_`,
  sets `force_rebuild_`. Morton box = periodic box if set, else patch bounds; skips
  with a warning if degenerate.
- `Patch::reorder_sorter()` — exposes that sorter's inverse map so a *parallel*
  subsystem can remap its own indices.
- `calculate_nonbonded_forces` — `rebuild |= force_rebuild_`, and a forced rebuild
  skips the displacement shortcut (the sorter's reference positions are stale after a
  permute). Then `pair_table_idx` re-resolves against the rebuilt exclusion CSR.
- `ParticleReorderManager` (ReorderManager.{h,cpp}) — cadence policy only. Returns a
  bool from `maybe_reorder(step, patch)`. Knows nothing about rigid bodies.

### Why the manager lives here and returns a bool
RigidBody is a *parallel* subsystem to Patch (future `RBOperation/`), not a parent, so
the reorder manager must not depend on it. SimManager is the coordinator that sees both
subsystems: it calls `maybe_reorder`, and on a true return bridges the RB system with
`rigid_body_manager_->remap_attached_particle_indices(patch.reorder_sorter())`. So the
only Patch↔RB coupling lives in SimManager, where two parallel systems are meant to meet.

### Ordering within the step (SimManager::execute_force_calculation)
reorder (top, before RB sync) → permute SoA + remap bonded + remap RB → RB sync writes
attached positions at the new slots → force loop rebuilds the pairlist (forced) →
resolve pair_table_idx → forces → integrate. `maybe_reorder` skips `step == 0` (the
priming call) and any step where bonded topology is not yet on-device
(`patch.is_bonded_prepared()`), since remap needs it.

### Load-balancing seam (future)
The reusable machinery is NOT this manager — it is the remap methods on the data
objects: `DeviceParticle::permute`/un-permute, `DeviceBondedInteractions::remap_particle_indices`
(+ exclusion CSR rebuild), `RigidBodyManager::remap_attached_particle_indices`. A future
multi-patch load-balancer (the natural home is `System/PatchDecomposer/ZOrder` —
`ZOrderDecomposer`, which today skips index remapping because it is single-patch) reuses
exactly these when particles migrate between patches. `ParticleReorderManager` is the
*intra-patch* policy; `ZOrderDecomposer` is the *global* policy. Keep them separate;
they can share a namespace once multi-patch is real.
