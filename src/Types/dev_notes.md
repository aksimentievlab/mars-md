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

## Matrix3.h — rotation_matrix_x/y/z were transposed

Legacy builds its elementary rotations with a **row-major** 9-argument
constructor (`RigidBody.cu:528-557`):

```cpp
return Matrix3( 1.0f, 0.0f, 0.0f,
                0.0f,  cos, -sin,
                0.0f,  sin,  cos);   // rows
```

The port copied those numbers into the **3-Vector3 constructor, which takes
columns** — the same convention `transpose()` relies on, where it builds
`Matrix3(r0, r1, r2)` out of M's rows. So every generator came out as its own
transpose, i.e. `R(t)` was really `R(-t)`: `rotation_matrix_z(0.3)` rotated
x_hat toward **-y** where legacy rotates it toward **+y**.

Consumers: `RBBD.h` (Brownian) and `RBDLM.h` (Langevin/DLM). Both were rotating
rigid bodies the wrong way. Fixing the generators fixes both, and **changes
every existing DLM trajectory** — expected, and the reason to re-baseline any
saved reference.

### Why nothing caught it

Thermal torque is isotropic, so flipping the rotation sense leaves free
rotational diffusion statistically identical. `Tests/privite_test/npc_6enl_beads`
free-body control: v1 1.09 +- 0.05 deg/step vs v2 1.07 +- 0.05, net rotation
12.4 vs 9.7 deg. Indistinguishable. Only a *directed* torque exposes the sign,
and until `inputRestraints` was wired up (same fixture, see
`Interactions/Bonded/dev_notes.md`) no configuration in the tree applied one to
a rigid body.

The diagnostic that found it: start the body 20 deg off its restraint anchor
orientation at `temperature 0.001` so noise vanishes, and watch the angle. v1
decays 20 -> 15.1 -> 11.3 -> ... -> 0. v2 grows 20 -> 24.9 -> 31.0 -> ... -> 180
and locks there, with step-1 |dtheta| equal to v1's to three significant figures
and opposite in sign. Reconstructing the applied body-frame rotation from
consecutive frames and dotting it against the analytic restraint torque gives
cos = +0.9995 for v1 and -0.9996 for v2.
