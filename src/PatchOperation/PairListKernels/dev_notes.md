# PairListKernels — dev notes

Implementation rationale moved out of the source comments. Code keeps only
punchy one-liners; the "why" lives here.

## ZOrderPairlist

### build_pairlist — periodic-axis Morton box
On a periodic axis the Morton encoding box must be the **simulation** box, not
the particle bounding box. Two things depend on it:

- The neighbour stencil wraps cell indices modulo the grid, i.e. it asserts cell
  `n-1` is physically adjacent to cell `0`. That is only true when the encoded
  extent equals the periodic extent. If particles occupy less than the full box,
  the wrap joins two cells that are not neighbours and real cross-boundary pairs
  go missing.
- The coarse cell width is `(encoded extent)/2^m` while `m` is chosen against the
  same extent. Deriving one from the box and the other from the bounding box lets
  cells come out narrower than the cutoff, silently breaking the guarantee that
  27 cells cover the cutoff sphere.

Open axes keep the bounding box: nothing to wrap, and a tight box gives finer
cells for the same `m`.

### build_pairlist — position snapshot
`sorter_.update_positions_incremental()` snapshots the positions this list was
built from, so `needs_update()` can later measure drift. Nothing else called it,
which left the reference buffer at its uninitialised construction value and made
`needs_update()` report a meaningless displacement.

### find_neighbors_zorder — coarse-cell resolution
Cells must be at least the pairlist cutoff wide in every dimension, otherwise a
27-cell stencil would not cover the cutoff sphere and pairs would be missed.
Morton codes carry `MortonCode::max_coord_bits` per dimension, so the coarse
level can be any `m <= that`; take the largest `m` whose cell is still `>= cutoff`
to minimise candidates scanned.

`max_bits` must match the resolution `MortonCode::encode` actually uses. `encode()`
quantises with the compile-time constant `max_coord_device`, **not** the
runtime-configurable `max_coord_bits`, so the coarse-cell shift is derived from
the former or the cell index would not correspond to the code prefix.

Cells are sized against the extent Morton encoding actually used (`last_box_extent_`).
It is the only extent the cell width is a fraction of, so mixing in the periodic
box here (as this once did) can make cells narrower than the cutoff whenever
particles do not fill the box — and a cell narrower than the cutoff is not covered
by a 27-cell stencil, so pairs are missed. `build_pairlist` forces the two to
agree on periodic axes; this keeps them agreeing on open ones.

### find_neighbors_zorder — overflow is fatal
The kernel refuses to write past `max_pairs_`, but the atomic counter keeps
climbing, so the raw value reports how many pairs *would* have been stored.
Overflow is fatal, not a warning: the pairs that fit are whichever ones won the
atomic race, so the list is a nondeterministic subset of the true neighbours.
Every subsequent step then computes a different physical system with no scientific
value — failing here costs a job, continuing costs a trajectory the user might
trust. (Logging and clamping is what this used to do.) Buffer sizing now comes
from `Pairlist::ensure_pair_capacity`, so overflow means that density estimate was
too tight — raise its safety factor or shorten the pairlist cutoff.

### set_periodic_box (header)
On a periodic axis the 27-cell stencil wraps at the grid edges and displacements
use the minimum image convention, so pairs spanning that boundary are enumerated.
Without this, such pairs are silently absent from the pairlist even though the
force kernel would apply minimum image to them.

Periodicity is per axis, so mixed boundary conditions are handled directly.
Treating a partly periodic box as fully open would drop exactly the cross-face
pairs the force kernel still wraps.

The origin matters as much as the length: on a periodic axis the Morton encoding
box is forced to `[origin, origin + length)` rather than the particle bounding
box, because wrapping a cell index modulo the grid is only geometrically correct
when the encoded extent *is* the periodic extent.

### kMaxCoarseBits (header)
Largest coarse-cell resolution used by the neighbor search, as bits per
dimension. Capping this bounds the cell arrays at `8^7 = 2M` entries; exceeding
the cap only makes cells wider than the cutoff, which costs extra candidates to
scan but never misses a pair.

## ZOrderNeighbor — BuildCellNeighborsKernel (27-cell cache)
Each coarse cell's up-to-27 neighbor cell indices are precomputed into
`cell_neighbors_` ([num_cells * MAX_NEIGHBORS], padded with `kInvalidCell`) and
the neighbor kernel just walks that table instead of recomputing `compact_by3` /
`split_by3` + wrapping per particle. The topology depends only on the grid
(`coarse_bits_`, per-axis periodicity), not on positions, so `find_neighbors_zorder`
rebuilds it only when the coarse grid or periodicity changes — for a fixed box that
is once, at the first build (~patch init). Memory is `num_cells * 27 * 4 B`, trivial
at realistic `m`; only near `kMaxCoarseBits = 7` (2M cells → 216 MB) is it large.

## ZOrderNeighbor — ZOrderCellNeighborKernel

### overview
Particles stay Morton-sorted (that is what gives the force kernel its memory
locality), but neighbors are enumerated by visiting the 27 coarse cells around
each particle, exactly as a conventional cell list does. The coarse cell side is
chosen on the host to be at least the pairlist cutoff, so the 27-cell stencil
provably covers the cutoff sphere and no interacting pair is missed.

Periodicity is per axis, carried entirely by `box_len`: a positive component
wraps that axis's cell indices and uses minimum image displacements; a zero
component leaves the axis open and clips the stencil there. Mixed boundary
conditions work instead of degrading the whole search to open. The Morton
encoding box must equal the simulation box on every periodic axis (enforced by
`ZOrderPairlist::build_pairlist`): wrapping a cell index modulo the grid asserts
cell `n-1` is adjacent to cell `0`, which only holds when the encoded extent is
the periodic extent.

### periodic offset range
With fewer than three cells along a periodic axis the wrapped offsets -1/0/+1
alias onto the same cell (all three when `n == 1`, and -1 with +1 when `n == 2`),
so the naive -1..1 loop visits that cell repeatedly and emits each pair in it up
to 27 times, multiplying its force by the same factor. Narrowing the range keeps
every distinct cell visited exactly once and stays complete: when `n <= 2` the
stencil covers the entire grid either way. Open axes are unaffected — out-of-range
indices are skipped rather than wrapped, so they cannot alias — and keep the full
range to reach the cell below.

### emit each pair once
Keyed on the *sorted* index. Because a cell occupies a contiguous run of the
sorted array, a whole cell lying before this particle collapses to an empty loop,
and the particle's own cell is entered at `i+1` — so roughly half the stencil is
skipped outright rather than enumerated and rejected pairwise. The stencil
relation is symmetric under wrapping, so a pair dropped here is always emitted by
the other particle's thread.

## Pairlist

### max_pairs is a device-memory budget, not a per-config estimate
`kPairlistMaxPairs` (Pairlist.h) is a single global buffer sized to ~30% of device
memory (`GPU_MEM` GiB from CMake, int2 = 8 B/pair) — e.g. 3 GiB / 402M pairs on a
10 GiB RTX 3080. It does *not* depend on particle count or configuration.

Why a fixed hardware budget rather than a density estimate: a config-dependent
guess underallocates exactly the case that matters — the minimization / relaxation
phase (as in oxDNA), which *starts* from a raw configuration with heavy overlaps
(coincident particles → near all-pairs locally). The buffer must survive that
worst transient so the dynamics can push particles apart; only when the true pair
count exceeds what the GPU can physically hold is the fatal overflow a real error
(raise `GPU_MEM`, or shorten the pairlist cutoff). A per-particle density estimate
threw on a legitimate overlapping start; a global memory budget does not.

## ZOrderCellNeighborKernel — tiled block-per-cell rewrite (2026-08-20)

### Why the per-particle version was slow (profile: 391 us, stddev <0.3%)
One thread per particle, no shared-memory staging: every candidate `pos_j` was
fetched from *global* memory once per home particle, and every accepted pair did
its own global `atomicAdd`. Dead-constant timing => global-load-bound serial scan.
v1's `createPairlists` beat it (148 us) purely on structure: block-per-cell,
shared-memory tiling, warp-aggregated atomics — NOT a better cell scheme.

Cell size is NOT the lever: Morton coarse cells are powers of two and the encode
budget caps cells/dim < 1024 (patch/pairlist_cutoff). Finer cells overflow the
code, so the fix is kernel shape, not granularity.

### New shape
- **Block per coarse cell**, grid-strided over `num_cells` (`num_blocks` = grid.x,
  capped at 65535). Sparse grids stay cheap; dense ones loop.
- **Shared-memory staging**: for each of the 27 neighbor cells, stage its particles
  into `stage[]` one `block_size`-particle tile at a time. Each neighbor position is
  read from global memory once per block, not once per home particle. Shared memory
  is fixed (one tile), independent of cell occupancy — big cells just add tile loops.
  At TPB=128, Vector3=16 B: stage 2048 + scan 512 + base 4 = 2564 B/block (trivial
  vs 48 KB default; not the occupancy limiter).
- **Block-aggregated emit** (portable replacement for v1's warp `atomicAggInc`,
  built on WorkItem barriers since no warp-intrinsic abstraction exists):
  Pass A counts each thread's hits over the staged tile; thread 0 does a sequential
  exclusive prefix sum in `scan[]` and one global `ATOMIC_ADD(pair_count, total)` to
  reserve the block-batch; Pass B rewalks the tile and writes hits at
  `base + scan[tid]`. Collapses N global atomics into one per (block, neighbor-tile).

### Invariants preserved
- Dedup by sorted index (`sj > si`): each unordered pair is seen from both cells;
  only `sj > si` passes. Emitted int2 is ordered on *original* indices (`a<b?..`),
  the x<y invariant the sorted key drops.
- Overflow stays fatal: `pair_count` counts the true total (increment always);
  writes guarded by `out < max_pairs`; host checks and throws.

### Correctness gotchas baked in
- Every barrier is block-uniform: cell/neighbor `continue`s depend only on
  block-wide ranges, so all threads skip together — no half-block barrier hang.
- Tail threads (`validI == false`, or `tid >= tile_n`) still hit every barrier;
  they just contribute 0 hits.
- Two-pass recomputes `wrap_diff` (Pass A + Pass B). Accepted cost: both passes read
  `pos_j` from shared memory, and the win was removing the *global* re-fetch.

### Still TODO / verify before trusting
- Compile (CUDA): forces `launch_cuda_kernel_with_workitem<ZOrderCellNeighborKernel>`
  via ZOrderNeighbor.cu; host .cpp must see the `extern template` (KernelHelper.cuh).
- **Validate pair count matches the old kernel** on a known system before profiling.
- Profile: expect the 391 us to drop toward / below v1's 148 us. Tune TPB (128 vs 256).

## Big-system profile fixes (300K mpipi, 4096 cells) — 2026-08-21

Grid was fine (4096 blocks, saturated). The slowness was overhead, two parts:

### 1. BoundingBoxKernel was 6 ms/rebuild of pure waste -> removed
For a periodic box the computed bbox was immediately overwritten by the sim box,
so the O(N) reduction ran for nothing. Deeper: the Morton domain IS the box.
ZOrderPairlist now stores a `PeriodicBox box_` (set from Patch's system box, which
carries box_size on every axis incl. open ones) and uses origin+box_size as the
domain every build. Deleted: BoundingBoxKernel usage, get_bounding_box, auto_bbox_,
manual bounds, set_bounding_box_mode, persistent bbox buffers. box_ is per-patch and
re-read each build (multi-GPU: a patch's box may change). Every system defines a box,
so there is no particle-extent fallback.

### 2. Two-pass emit -> single-pass warp-aggregated atomic
The block-aggregation (count -> prefix-sum -> reserve -> rewrite) computed wrap_diff
twice and serialized a per-tile prefix sum in thread 0. That was ~27% over v1.
Replaced with v1's atomicAggInc idea: single pass, `__ballot_sync`+`__popc` give one
atomicAdd per warp, each lane writes at base+rank. Guarded by `#if defined(__CUDA_ARCH__)`
(warp intrinsics are device-only even under USE_CUDA); SYCL/CPU keep the per-hit
ATOMIC_ADD via the portable macro. Shared memory dropped to just the staged tile
(scan/base buffers gone).

Correctness of warp path: the s-loop over a tile is warp-convergent (tile_n is
block-uniform) so every lane reaches the ballot; validI/hit=false lanes still vote.
Dedup (sorted_j>sorted_i) and original-index ordering unchanged.

## v1 vs v2 head-to-head, same machine (2026-09-11)

`Tests/privite_test/cytoplasm`, 305k particles, cutoff 35 / pairlistDistance 10,
300 steps, energy off, RTX PRO 6000 Blackwell. v1 = `arbd_stampede/build/arbd`
on `mpipi_rna_arbd1.bd`, v2 on the matching `_arbd2.bd`.

| kernel | v1 | v2 | ratio |
|---|--:|--:|--:|
| force, median | 1524 us | 1767 us | 1.16x slower |
| force, total | 508.6 ms | 643.5 ms | |
| **pairlist build, 1 launch** | **2.32 ms** | **707 ms** | **305x slower** |
| wall clock | 0.55 s | 1.41 s | 2.6x |

**One v2 pairlist build costs more than v1's entire 300-step run.** This is now
the top optimisation target, not the force kernel.

### What the build cost is NOT

- **Not the single-address atomic on `pair_count`.** Replacing the per-hit atomic
  with a count pass + one block reservation per particle (113x fewer atomics)
  made the kernel take exactly 2x as long, i.e. two stencil walks with the atomic
  contributing ~13 ms of the 707. The walk is the cost.
- **Not clustering.** 3734 of 4096 coarse cells occupied, mean 82 particles,
  max 407. Effectively uniform, ~2200 candidates per particle, ~3.4e8 candidate
  tests total.
- **Not `wrap_diff`.** Orthogonal fast path, a few flops.
- **Not cell sizing.** `BuildCellNeighborsKernel` launches 4096 threads,
  confirming m=4, 16 cells/dim at 50 A against a 45 A cutoff. Correct.

The anomaly: the build does ~0.5 G candidate tests/s while the force kernel does
~23 G pairs/s on strictly more expensive work per item. ~50x off what the work
justifies. `ncu` could not confirm why (ERR_NVGPUCTRPERM on this host; needs a
sysadmin to enable GPU performance counters).

### What v1 does differently (`ComputeForce.cuh:301`, `createPairlists<64,64,8>`)

1. Tiles particles through `__shared__ float4 particle[N]`, N=64, so the inner
   loop reads shared rather than global.
2. Reads positions and the neighbour table through texture objects.
3. Applies exclusions during the build via a merge-scan against `excludeMap`, so
   the emitted list is already exclusion-free.
4. Warp-aggregated allocation, `atomicAggInc` (worth little here, see above).

Item 1 is the likely bulk of the 305x and is the thing to try first.

## Pair emission order is load-bearing (2026-09-11)

Reserving a contiguous block per particle makes consecutive pair slots share the
same `indices.x`. A warp in the *force* kernel then fires 32 atomics at one
address and serialises: force-kernel median went 1767 -> 2521 us (+43%) with the
build kernel otherwise untouched. The per-hit atomic interleaves slots across
particles, so a warp lands on 32 distinct addresses.

Consequence for any per-particle pairlist redesign: row-contiguous pairs plus a
per-pair force kernel is the worst combination. It only pays off if the force
kernel also becomes per-particle and accumulates in registers, committing once.

## Correction: the earlier v1-vs-v2 numbers in this file were measured on a broken build

Superseded by the CUDA 12.8/Blackwell sort bug (see `../ZOrderKernels/dev_notes.md`) and by
the `Patch` constructor ordering bug below. The "305x slower pairlist build" figure was
entirely the corrupt sort. After both fixes, `ZOrderCellNeighborKernel` on the 305k cytoplasm
case is **20.2 ms**, against v1's `createPairlists` at 2.32 ms.

### The box never reached the pairlist (2026-09-11)

`Patch`'s constructor ran `set_periodic_box(periodic_box)` *before*
`pairlist_ = create_pairlist(...)`. `set_periodic_box` forwards the box via
`dynamic_cast<ZOrderPairlist*>(pairlist_.get())`, which on a null `pairlist_` yields nullptr
and silently drops it. Consequence, for the whole history of the code:

- the pairlist ran with **periodicity disabled**, so no cross-boundary pairs were ever
  enumerated, on systems configured `per=(true,true,true)`;
- the Morton domain came from the particle bounding box rather than the simulation box.

Fixed by constructing the pairlist first. Also:

- `Pairlist` (base) now owns `set_periodic_box(const PeriodicBox&)`, filling the `box_` member
  that was already declared and unused. It takes a whole box rather than a `Vector3` where a
  zero component had to mean both "no extent" and "do not wrap" - those are different
  questions and conflating them is what made the bug invisible.
- Building with a zero-size box now throws naming `set_periodic_box()`, instead of silently
  producing a degenerate Morton domain. `PeriodicBox()` default-constructs to size zero.

### get_bounding_box stays, and is NOT waste

An earlier note here claimed `BoundingBoxKernel` was pure waste because a periodic box
overwrites the result. That is only true on periodic axes. The domain selection is per axis:

- **periodic axis**: `[origin, origin + box_size)`, because wrapping is defined against it;
- **open axis**: the particle extent plus a 1% margin, which is tighter and packs the coarse
  cells better.

This is the original logic. It looked dead only because the box never arrived, so every axis
silently took the extent path.

### Measurement caveat that cost an afternoon

`export CMAKE_CUDA_ARCHITECTURES=...` in `build_cuda.sh` only seeds the cache on a **fresh**
configure. An existing `CMakeCache.txt` keeps its old value, and the build silently targets
the wrong architecture (we ran sm_75 code JIT-translated on a sm_120 device). Pass it as
`-DCMAKE_CUDA_ARCHITECTURES="86;120"` on the cmake line and verify with:

```
grep CMAKE_CUDA_ARCHITECTURES: build/tbgl-cuda-release/CMakeCache.txt
```

The same trap applies to `CMAKE_CUDA_COMPILER`: loading the cuda-13 module does nothing to a
tree already configured against 12.8.
