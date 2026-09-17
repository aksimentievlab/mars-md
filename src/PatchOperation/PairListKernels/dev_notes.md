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

## Exclusion filtering moved into the build (step 1 of 5)

Excluded pairs used to reach `neighbor_pairs_` and were dropped later by
`ResolvePairTableKernel`'s CSR scan (`Interactions/Nonbonded/Pairwise.h`).
They are now dropped at emit time instead.

### Why at emit time

The CSR scan runs on exactly the pairs that already passed the cutoff test, so
the scan itself costs the same either way. What it buys:

- The list shrinks by the excluded-and-in-cutoff count. Every downstream pass
  over `num_pairs` gets shorter, not just the nonbonded one.
- `ResolvePairTableKernel` loses the scan and four members (steps 2-3 will cut
  it to three loads and a store).
- Exclusions apply to *every* nonbonded term, so this is the right layer.
  The type->table lookup stays in Resolve because Coulomb and the analytical
  terms want pairs that have no tabulated entry.

### Shared view

`ExclusionView` (`Interactions/DeviceExclusions.h`) holds the borrowed CSR
pointers and `is_excluded(a, b)`. It lives in `Interactions/` because the CSR
is owned by `DeviceBondedInteractions`; the pairlist only borrows it.

`Pairlist::set_exclusions` stores one, and each builder passes it to its emit
kernel. A default-constructed view has `num_particles == 0`, so `is_excluded`
returns false before touching the null pointers — a builder that is never
given exclusions behaves exactly as before.

### Emit sites covered

- `ZOrderCellNeighborKernel` (ZOrderNeighbor.h) — the default builder.
- `CellNeighborKernel` (DecomposeKernels.h) — used by `CellListPairlist`.
- `AdaptiveZOrderNeighborKernel` (AdaptiveZOrderNeighbor.h) — **not covered**;
  it is not launched from anywhere. If it is ever revived it needs the same
  three lines, otherwise exclusions silently stop being applied once
  `ResolvePairTableKernel` drops its scan in step 3.

### Index space

The CSR is indexed by patch-local particle index. Z-order emits
`sorted_to_original[...]`, which is that same space, so the lookup is direct.
`ensure_bonded_topology_ready()` runs earlier in `calculate_nonbonded_forces`
than the rebuild block, so a reorder's rebuilt CSR is already in place.

One incidental change in `ZOrderCellNeighborKernel`: the two
`sorted_to_original` loads moved out of the `pair_idx < max_pairs` guard, since
the exclusion test needs them before the atomic. That guard only fails on
overflow, which is fatal anyway.

### Step 1 is independently verifiable

`ResolvePairTableKernel` keeps its exclusion scan for now, so the filtering is
redundant rather than load-bearing. Pair counts should drop; forces and
energies should not move. The scan comes out in step 3.

## The build's 8.4x gap to v1 is oversized coarse cells on anisotropic boxes (2026-09-16)

`find_neighbors_zorder` picks `m = floor(log2(min_extent / cutoff))` and
`BuildCellNeighborsKernel` walks a fixed +/-1 stencil (`MAX_NEIGHBORS = 27`). Morton codes
divide every axis the same number of times, so **the narrowest axis dictates `m` and the wide
axes get cells far larger than the cutoff**.

| | box | cutoff+skin | m | cell sizes | cell/cutoff | stencil volume |
| --- | --- | --: | --: | --- | --: | --: |
| cytoplasm | 800^3 cubic | 45 | 4 | 50.0 / 50.0 / 50.0 | 1.11x | 1.4x ideal |
| nupod | 2748.7 x **2040** x 2748.7 | 75 | 4 | 171.8 / 127.5 / 171.8 | **2.29x** | **12x ideal** |

nupod cannot use m=5 today because y would fall to 63.75 A, under the 75 A cutoff, and a
+/-1 stencil would then miss pairs. So m=4 is *correct* given the fixed stencil — the fixed
stencil is the bug.

This predicts everything observed. Scanned volume is 12x ideal on nupod and the measured
build gap to v1 is 8.4x (the shortfall is because the particle blob fills only part of the
box). On cytoplasm the ratio is 1.11x and v2's build is *faster* than v1's (2.08 ms vs
2.22 ms). The earlier note in this file dismissing cell sizing — "16 cells/dim at 50 A
against a 45 A cutoff, correct" — was measured on cytoplasm, where it happens to be true, and
does not generalise to an anisotropic box.

### Fix: per-axis stencil radius, not a traditional cell list

Let the stencil radius per axis be `ceil(cutoff / cell_size_axis)` instead of 1, and choose
`m` to minimise total scanned volume under a memory cap:

| m | cells/dim | cell sizes | stencil | scanned volume | vs now |
| --: | --: | --- | --- | --: | --: |
| 4 (today) | 16 | 171.8 / 127.5 / 171.8 | 3x3x3 | 1.02e8 | — |
| **5** | 32 | 85.9 / 63.8 / 85.9 | **3x5x3** | 2.12e7 | **4.8x less** |
| 6 | 64 | 43.0 / 31.9 / 43.0 | 5x7x5 | 1.03e7 | 9.9x less |

m=5 needs a 32,768-cell table at 45 neighbours = 5.9 MB; m=6 would need 183 MB. So m=5 is the
practical choice and should recover most of the build gap.

Everything the current design depends on survives: cells remain Morton-code prefixes, so a
cell is still a contiguous run of the sorted array, and `BuildCellRangesKernel` is unchanged.
What changes is `MAX_NEIGHBORS` becoming a runtime per-axis product rather than a hardcoded
27, and the `m` search minimising scanned volume rather than just clamping at cell >= cutoff.

**Do not replace this with a traditional cell list to fix it.** Morton ordering is doing real
work elsewhere — the force kernel decays 24.5% between reorders (see
`Interactions/Nonbonded/dev_notes.md`) — and the search grid and the memory layout are
separable concerns. Only the search grid is wrong.

### Implementation, branch `stencil` (2026-09-16)

Three files, all portable — no CUDA-specific path, nothing that a SYCL or Metal backend
cannot follow.

**`ZOrderNeighbor.h`.** `BuildCellNeighborsKernel` gains `int3 z_order_raddi` and
`int neighbors_per_cell`; `ZOrderCellNeighborKernel` gains `neighbors_per_cell`. Every
`MAX_NEIGHBORS` in both becomes the runtime stride. The hardcoded offset bounds are replaced
by `axis_range()`:

```
axis_range(periodic, r, n) = periodic && 2r+1 >= n ? [0, n-1] : [-r, r]
```

This generalises the old `p_lo/p_hi` narrowing rather than replacing it. At r=1 it reproduces
master exactly: n>=3 gives [-1,1], n==2 gives [0,1], n==1 gives [0,0]. The periodic branch
matters because wrapping offsets onto a grid narrower than the stencil would list the same
cell twice and emit every pair in it twice.

`MAX_NEIGHBORS` in `Header.h` is left at 27 — `BaseGridDevice.h` uses it for an unrelated
`IndexList`, and the pairlist no longer refers to it except as the initial allocation size.

**`ZOrderPairlist.h`.** New members `cell_radii_`, `neighbors_per_cell_`, and
`cell_neighbors_radii_` (the table's cache key now includes the radii, so a radius change
rebuilds it — master keyed on `m` and periodicity alone, which was already slightly unsound
since an open axis's extent drifts with the particle blob).

**`ZOrderPairlist.cpp`.** `select_coarse_grid()` replaces the one-line `m` formula. It sweeps
m = 0..min(10, kMaxCoarseBits), and for each computes per-axis `r = ceil(cutoff / side)` and
`span = min(2r+1, n)`.

#### Why the objective is not just scanned volume

Minimising scanned volume alone over-refines on sparse systems: at m=6 a particle walks 175
stencil slots, and if the cells are nearly empty that is 175 `cell_begin`/`cell_end` loads to
find almost nothing. The cost model therefore charges both terms, per particle:

```
cost(m) = slots + scanned_volume * density
```

— one header load per slot, one distance test per particle inside the scanned volume,
weighted equally. Equal weighting is the conservative choice: it over-charges refinement
(a header load is two adjacent uint32 shared across the warp, cheaper than a distance test),
so a fine grid has to earn its place. This removed the need for a separate minimum-occupancy
guard; the `slots` term self-regulates.

Where the crossover lands (N = particle count):

| system | m=4 | m=5 | m=5 wins above |
| --- | --- | --- | --: |
| nupod | 27 + 6.59e-3 N | 45 + 1.37e-3 N | N > 3,450 |
| cytoplasm | 27 + 6.59e-3 N | 125 + 3.81e-3 N | N > 35,000 |

Both production systems are far above their crossover, so both should land on m=5. Cytoplasm
moving too is expected and harmless — 1.7x fewer distance tests on a build that already beat
v1.

#### Memory cap, and why it is 32 MB

Bounded as `num_cells * (slots * 4 + 8)` bytes, counting the neighbor table plus
`cell_begin`/`cell_end`. Both terms rise monotonically with m, so the sweep can `break`.

| m | nupod table | cytoplasm table |
| --: | --: | --: |
| 5 | 6.4 MB | 16.6 MB |
| 6 | 185 MB | 766 MB |

32 MB admits m=5 for both and excludes m=6 for both. m=6 would be another 2x on nupod's
scanned volume and is worth a try on a large-memory card — hence the override below rather
than a recompile.

#### Overrides

- `MARS_ZORDER_TABLE_MB` — raise the cap (e.g. `256` to let nupod reach m=6).
- `MARS_ZORDER_BITS` — force `m` outright, bypassing both cost model and cap. For A/B against
  master, `MARS_ZORDER_BITS=4` reproduces master's grid on these two systems.

`PLDIAG` now logs `stencil=(rx,ry,rz)xSLOTS` and the per-axis cell size, so the chosen grid is
visible at LOGDEBUG without re-deriving it.

#### Correctness argument

- **Coverage.** `side = extent / 2^m` *under*-estimates the true cell width, which is
  `extent * 2^(10-m) / 1023`. So `r = ceil(cutoff / side)` is at or above the true
  requirement — conservative, never short.
- **No double emission.** `j > i` on the sorted index still does the deduplication; it does
  not depend on stencil shape, only on the stencil being symmetric, which `[-r, r]` and the
  full-axis sweep both are.
- **No table overrun.** Per axis the builder writes at most `min(2r+1, n)` entries (periodic
  hits it exactly; open clamps below it), which is exactly the `span` the host multiplied into
  the stride.
- **Unchanged.** `BuildCellRangesKernel`, the Morton encode, the per-hit emission atomic, and
  the reorder cadence. This touches only which cells get scanned.

Not yet measured — build and profile before trusting the table above.

### Measured on branch `stencil`, nupod 20k steps (2026-09-16)

All three on the same idle GPU, `m` pinned with `MARS_ZORDER_BITS` so the grid is the only
difference. `s_m4` reproduces master's grid exactly (r=1,1,1 -> 27 slots).

| m | cells | slots | build | force | GPU busy |
| --: | --: | --: | --: | --: | --: |
| 4 (master) | 4,096 | 27 | 30,113 us | 2,108.3 us | 49,794 ms |
| **5 (auto)** | 32,768 | 45 | **23,617 us  -21.6%** | 2,105.0 us | **48,600 ms  -2.4%** |
| 6 (forced) | 262,144 | 175 | 23,200 us | 2,198.9 us **+4.5%** | 50,717 ms **+1.9%** |

The chooser picked m=5 on its own. First-build pair count identical across all three
(61,745,536) — coverage is exact and nothing is double-emitted. Later medians drift by ~0.04%
only because trajectories diverge once summation order changes.

#### The volume model in the note above was wrong by 3x

Subtracting the atomic (sec 6.1 of `cuda_warpnode.md`, unchanged since pair count is
unchanged) to isolate the walk:

| m | candidates/particle | reduction | walk speedup |
| --: | --: | --: | --: |
| 4 | 108,822 | 1.0x | 1.00x |
| 5 | 22,671 | 4.8x | **1.54x** |
| 6 | 11,004 | 9.9x | 1.60x |

Predicted 4.8x on the walk, got 1.54x, and it **saturates** — m=6 halves candidates again for
4%. The walk is not candidate-bound past m=5. Per-slot header loads (27 -> 45 -> 175) are the
obvious suspect but were not measured. Do not trust scanned volume as a cost proxy here.

#### m=6 loses on the force kernel, not the build

m=6's build is marginally better than m=5's but its force kernel is 4.5% slower, so total GPU
busy is worse than *master*. This is the sec 4.1 emission-order effect: the grid changes the
order the walk visits cells, which changes what the emission race produces, which changes
force-kernel atomic conflicts. **The build cannot be tuned in isolation from the force
kernel** — always check the force median when changing the grid.

#### The byte cap is load-bearing, and two errors are cancelling

`density = num_particles / box_volume` is **253x too low** on nupod: 1893 neighbours within
75 A implies local density 1.07e-3/A^3 against a box average of 4.23e-6/A^3, i.e. the pod
fills ~0.4% of its bounding box. With the true local density the cost function prefers m=6,
which the table shows is wrong. The 32 MB cap excludes m=6 (185 MB) regardless, so m=5 is
selected either way.

So the current selection is right but not for a principled reason — under-counting candidates
happens to offset the walk's sub-linearity. Anyone raising `MARS_ZORDER_TABLE_MB` must
re-measure, including the force kernel. A principled chooser would need the force-kernel
coupling, which the pairlist builder cannot see.

#### Is the build worth more work? No.

Build is 9.7% of GPU time at `decompPeriod 125`. Even a v1-class build (4,338 us, another
5.4x) would be worth **-8.3% of total**. Force is 85.6%. The build is done as a target.

#### Acceptance: the 20% bar does not apply here

Clarified 2026-09-16: the ~20% threshold was about whether a win justifies **forking the
CUDA / SYCL / Metal paths**, not about raw speed. It is the price of permanent backend
divergence plus the coupling hazard (layout and warp aggregation must never be changed
independently). The warp-aggregate work on `pairlist_stride_build` is CUDA-only and measured
~15%, so it has to argue past that bar.

The stencil fix has no `#ifdef` and no warp intrinsics — every backend gets it. There is no
bar to clear; it is taken on being faster. Do not compare its 2.4% against the 15% as if they
were the same kind of number.

#### Correction: where the 253x density error actually comes from (2026-09-16)

The earlier entry said "the pod fills ~0.4% of its bounding box". Wrong. Measured from the
written PDB, the 253x splits into two independent factors:

| factor | cause | numbers |
| --: | --- | --- |
| 28.1x | structure occupies 3.55% of the periodic box | bbox 1155.6 x 412.6 x 1149.1 vs box 2748.7 x 2040 x 2748.7 |
| 9.0x | **the pod is hollow** — voids inside it | particles fill only **11%** of their own bbox |
| 253x | product | local 1.071e-3 vs box-average 4.233e-6 /A^3 |

All three axes are periodic in this config (`min_extent = 2040`, `floor(log2(2040/75)) = 4`,
matching the observed 4,096 cells), so `last_box_extent_` is the **simulation box** and
`BoundingBoxKernel`'s per-build reduction is discarded on every axis. The chooser therefore
never sees even the 28x, let alone the voids.

Using the pair count to recover true local density is cheap and available after the first
build (`nbr = 2*num_pairs_/N`, `local = nbr / (4/3 pi cutoff^3)`). **Do not do it without
recalibrating the slots weight.** All three density estimates -- box 4.2e-6, bbox 1.2e-4,
local 1.1e-3 -- still select m=5, because the 32 MB cap excludes m=6 (185 MB) in every case.
Improving the density alone would only matter if the cap were raised, and then it would push
toward m=6, which is measurably worse.

The measured walk scaling is roughly `candidates^0.28` (4.8x fewer candidates -> 1.54x
faster), so the candidate term wants a strongly sublinear weight, not a linear one. Three
points on one system is not enough to fit that, and the cap makes it moot. Left as is,
deliberately.
