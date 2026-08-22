# dev_notes — ZOrderKernels

## Z-order canonical reordering — RemapIndicesKernel / remap_indices

Part of the off-by-default `ENABLE_ZORDER_REORDER` work (see plan
`buzzing-tickling-music.md`, Patch::reorder_particles).

**Why remap_indices takes a flat `int*`.** Bonded/exclusion/RB index structures
are `int2` (bond, excl edge), `int3` (angle), `int4` (dihedral), or `int` (RB
particle_index, restraint ids). Rather than a per-type kernel, we treat any of
those buffers as a flat `int[]` of length `N*components` and remap every int in
one launch. `DeviceBuffer<int2>::data()` is contiguous POD, so
`reinterpret_cast<int*>` + count `N*2` is valid (pointer cast only; no host deref).

**Sentinel guard.** The kernel skips `old < 0 || old >= map_size` so padding and
"no particle" markers survive a reorder untouched. `map_size = num_particles_`
(the valid range of `inverse_indices_`, which is allocated to `max_particles_`
but only `[0, num_particles_)` is meaningful after a sort).

**In-place is fine.** Runs at reorder cadence (every K-th rebuild), not per step,
and the persistent bonded buffers are canonical — no aliasing since each int maps
to exactly one new slot.

**Extra ReorderDataKernel instantiations.** Stage 2 permutes the SoA fields
`id`/`type_id` (int) and `flags` (uint32_t) via `reorder_data<T>`, so
`ReorderDataKernel<int>` and `ReorderDataKernel<uint32_t>` are instantiated in
ZOrderKernels.cu alongside the existing `<Vector3>`.

**sort_particles must sync after create_inverse_mapping.** `InverseIndexKernel`
runs on the compute stream; the reorder path consumes `inverse_indices_`
cross-stream almost immediately (host `copy_to_host` in the exclusion CSR rebuild
and RB attached-index remap). Without a `synchronize_streams()` after
create_inverse_mapping, that host read races the still-running kernel -> garbage
inverse map -> corrupt indices -> a release-only, timing-dependent hang that
vanishes under gdb/debug/compute-sanitizer (all serialize) and under
CUDA_LAUNCH_BLOCKING=1. The pairlist never tripped this because it only consumes
`sorted_indices_` (synced before create_inverse_mapping). Fix: sync at the end of
sort_particles so all its outputs are settled when it returns.
