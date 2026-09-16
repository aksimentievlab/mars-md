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

**`components` is NOT what the type name says — derive it with `sizeof`.** `int3`
and `int4` are both `Vector3_t<mars_int>`, which is `alignas(4*sizeof(T))` over
`x, y, z, t` — *four* ints either way. Only `int2` (`Vec2<mars_int>`) really holds
two. Passing `num_angles * 3` for an `int3` buffer therefore walks a 4-int-stride
array as if it were packed triples: it rewrites the padding `t` slots and leaves
the last quarter of the real indices un-remapped, silently corrupting the angle
topology on the first reorder. All call sites now use
`N * (sizeof(T) / sizeof(int))` so the count cannot drift from the layout.

Symptom when this was live: with `reorderPeriod 1000`, a nupod Langevin run was
correct through frame 0 and then, from the first reorder onward, angles spanned
unrelated atoms — PE jumped +208k kcal/mol in one interval and KE ran to ~4x
equipartition. Bonds (`int2`, x2) and dihedrals (`int4`, x4) had correct counts
and were unaffected, which is why disabling angles alone made the run flat.
The padding is safe to remap: `Vector3_t(x,y,z)` zeroes `t`, and 0 is a valid
slot, so it maps to a real index and is never read by the angle kernel.

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

## BUG: the Morton sort is wrong at scale (2026-09-11)

`ZOrderSort::validate_sorting()` on the 305k-particle cytoplasm case
(`Tests/privite_test/cytoplasm/bench300.bd`) reports **~49,000 adjacent
inversions out of 304,910 particles (16%)**, and the number changes every run:

```
run 1:  sort errors = 49,044    pairs = 102,973,946
run 2:  sort errors = 49,154    pairs =  83,743,905
run 3:  sort errors = 49,166    pairs = 100,019,301
```

True half-pair count at the configured 45 A cutoff, from a scipy cKDTree over
the same input file, is **56,231,831**. So the pairlist is 1.5-1.9x oversized
AND different on every run.

### Why this matters beyond the pairlist
`sorted_positions` and `morton_codes` disagree, so particles are indexed into
cells they are not in. `BuildCellRangesKernel` then writes bogus `[begin,end)`
spans, threads scan far beyond a cell's real ~82 particles, and the same pair is
emitted from several cells. This is almost certainly the whole explanation for
`ZOrderCellNeighborKernel` measuring **707 ms vs v1's 2.32 ms (305x)** - it is
not a kernel-shape problem, it is scanning garbage.

**Every performance number taken before this is fixed is measured on a corrupt,
oversized pair list and must be retaken.**

### What was ruled out
- Not the cutoff: diagnostic prints `cutoff=45.000 cutoff2=2025.0`, correct.
- Not cell sizing: `m=4`, 4096 cells, 50 A cells vs a 45 A cutoff, correct.
- Not periodicity: a periodic cKDTree count differs from the open one by 1,578
  pairs, because the particles span 739-784 A inside an 800 A box.
- Not the `cell_begin_/cell_end_` fill racing the kernels. DeviceBuffer binds to
  `StreamType::Memory` while kernels run on Compute, and `fill()` only
  self-syncs under SYCL (Buffer.h:543), so a `synchronize_streams()` was added
  in `find_neighbors_zorder` - correct on its own merits, but it did not change
  the error count.
- Not CUB's DoubleBuffer result-side handling: `CUDASort.cu:74` already copies
  back from `Current()` when it differs.
- Not null-stream vs explicit-stream ordering: `cudaStreamCreate` (CUDAStreams.h:81)
  makes *blocking* streams, which serialize against the null stream the CUB sort
  uses.

### Why CI never caught it
`Unit_test/mars_zorder_tests` passes with `sort errors = 0` everywhere, but the
largest system it sorts is **400 particles** (most are 10-100). The failure lives
somewhere between 400 and 304,910. **A large-N sort test is the missing coverage.**

### Next
Bisect N to find where `validate_sorting()` first goes nonzero, then read
`MortonEncodeKernel` + `create_inverse_mapping` + the CUB driver at that size.
Note `validate_sorting()` itself reads `error_count_` back on the Memory stream
without waiting on its Compute-stream kernel (ZOrderSort.cpp:130-134), so the
number it returns is a lower bound and needs its own fix before being trusted
as an exact count.

### RESOLVED: it was the CUDA toolkit, not MARS code (2026-09-11)

`CMAKE_CUDA_COMPILER` in the existing build cache still pointed at
`/software/cuda-12.8/bin/nvcc` even though `build_cuda.sh` loads
`cuda-toolkit/13`. CMake does not change the compiler on reconfigure, so the
module load had no effect on an already-configured tree.

**CUB's radix sort in CUDA 12.8 silently returns wrong results on Blackwell
(sm_120).** No error, no launch failure, just an unsorted array. Rebuilding with
CUDA 13 fixes it: `validate_sorting()` returns 0 on all three runs, and
`Unit_test/mars_zorder_tests`'s "Sort 1G elements" case passes.

**If you are on Blackwell, you must build with CUDA 13. Check the cache, not the
module.** A stale `CMAKE_CUDA_COMPILER` is silent and corrupts physics.

Effect of the fix on the 305k cytoplasm case:

| | before (12.8) | after (13) |
|---|--:|--:|
| pairs built | 78-106M, varying | 56,231,936, exact and stable |
| `ZOrderCellNeighborKernel` | 707 ms | **2.66 ms** |
| force kernel median | 1767 us | 1311 us |
| wall clock, 300 steps | 1.41 s | 0.50 s |

The pair count now matches a scipy cKDTree count at the 45 A cutoff exactly
(56,231,831 rounded up to a 256-thread block). The 305x pairlist-build gap
against v1 was entirely this bug; the kernel shape was never the problem.
