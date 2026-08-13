# BaseGridDevice.h — notes

## Why three types

| Type | Holds | For |
|---|---|---|
| `GridGeometry<T>` | origin/basis/basis_inv/dimensions, no data pointer | shared base; the one thing both views agree on |
| `BaseGridView<T>` | GridGeometry + `CONSTANT_PTR(T)` | sampling |
| `BaseGridMutableView<T>` | GridGeometry + `DEVICE_PTR(T)` | mutation kernels |

## Why the views inherit from GridGeometry

Inheritance, not composition, so `view.origin` / `view.dimensions` keep resolving
as inherited members — no call site changed when this was introduced (~87 field
accesses across 17 files would otherwise have needed rewriting). Data-structure
organization only: no virtuals, all dispatch compile-time.

## Why const and mutable are separate types

`CONSTANT_PTR(T)` is `const T*` on CUDA/SYCL, `constant T*` on Metal — a read-only
address space. A mutating method on `BaseGridView` is a compile error on CUDA/SYCL
today and an address-space violation on Metal later. Splitting costs nothing now
and is what keeps Metal viable without gating on it.

## Why the view methods only forward

The sampling math lives in the free functions lower in the file; the view methods
are thin forwards. This is deliberate.

`BaseGridView` used to carry its own trilinear implementation that **clamped** out
of bounds, while the free functions **return zero**. Because both existed,
`PmfKernels::sample_force_grid_value` dispatched its `scheme != 0` ("cubic") branch
to `grid.interpolate()` — which was trilinear — so cubic silently produced linear
results for force grids, while `sample_pmf_grid` right above it dispatched
correctly. Two implementations of one thing is how that hid.

Deleted at the same time, both dead and both broken:
- `ScaleGrid` — body was `grid_values = grid_values * scale` on a `T*`; invalid C++,
  survived only by never being instantiated.
- `InterpolateGridPoint` — missing `HOST DEVICE`, and its bounds check tested
  `>= nx` before reading `i0+1`, so positions in `[nx-1, nx)` read out of bounds.

## Boundary conditions

Previously `boundary_condition` was accepted by every sampler and ignored: out-of-bounds
reads returned zero (Dirichlet-like) while neighbor taps wrapped (Periodic-like) —
inconsistent inside a single sampler. Now `map_grid_index` / `fetch_grid_value` apply it
per tap, and all four samplers route through them.

| BC | Out-of-range tap |
|---|---|
| Dirichlet (0) | reads 0 |
| Neumann (1) | clamps to the edge value (zero derivative) |
| Periodic (2) | wraps modulo the dimension |

`boundary_condition` is compared as `int`, never cast to `GridBoundaryCondition`: it is
unvalidated, and `-1` ("use the grid's own", see `GridTerm`) is a legal value that is not
an enumerator of a `uint8_t`-backed enum.

Two behavior details worth knowing:

- `interpolate_grid_point` no longer early-returns for out-of-bounds positions, because
  Periodic/Neumann need the taps mapped. Dirichlet is unaffected — every tap falls
  outside, so the result is still 0. Indices now use `floor`, not truncation, so negative
  positions land in the cell below rather than at 0.
- `compute_gradient` keeps its interior-only guard **for Dirichlet only**. A central
  difference against zero-padded taps at an edge manufactures a large spurious gradient
  pointing out of the grid; Periodic/Neumann have real neighbors, so they don't need it.

### Why the default flipped to Dirichlet

`BaseGrid::Config::boundary` used to default to `Periodic`, and `DXReader` never sets it,
so every `.dx` grid was nominally periodic while the samplers behaved as Dirichlet. Honoring
the flag without changing the default would have silently wrapped every sub-volume PMF or
force grid — particles outside such a grid would go from zero force to a force sampled from
the opposite face. The default is now `Dirichlet`, which preserves the effective behavior
that existing runs were getting; `Periodic` is opt-in via `gridFileBoundaryConditions`.
