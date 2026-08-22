# src/Objects — implementation notes

## DeviceRigidBody.h

### `RigidBodyView::apply_body_frame_rotation` does not re-orthonormalize

Legacy `RigidBody::applyRotation` (`RigidBody.cu:513-523`) reads:

```cpp
angularMomentum = R * angularMomentum;
orientation = orientation * R;
orientation.normalized();   // return value discarded - a no-op
```

`normalized()` is `const` and returns a `Matrix3`, so that third line does
nothing. Legacy's only real re-orthonormalization is the
`orientation = orientation.normalized()` at the end of `integrateDLM`'s drift
branch (`RigidBody.cu:453`) — once per step, not once per sub-rotation.

`RBIntegrateDLMKernel` owns that single call. Doing it here instead would run
five Gram-Schmidts per body per step for a correction of ~1e-7 in float, since
the Cayley form is exactly orthonormal in exact arithmetic (`c^2 + s^2 = 1`
identically).

Note this cannot affect `<L_i^2>` statistics either way: every torque is built
in the body frame and consumed in the body frame, so the orientation cancels out
of the angular-momentum dynamics entirely.

### `RigidBodyTypeView` grid lists are offset+count, not fixed fields

A type's density/potential/pmf grid lists are variable length, unlike
`ParticleTypeView`'s fixed `int3 force_grid_id`. Each list is therefore an
offset+count pair indexing a flat grid-id buffer owned by
`DeviceRigidBodyTypes`, rather than a fixed-size field on the view.

`grid_terms` is the backing storage those pairs index into — one shared pointer,
not per-type, laid out per type as `[potential terms][density terms][pmf terms]`
contiguously. One `GridTerm` (grid_id + scale + scale_slope +
boundary_condition) per grid, matching `ParticleTypeView::pmf_grid_terms`, so a
grid costs a single load instead of combining parallel id/scale arrays.
# Grid.h — implementation notes

## Path resolution: resolve for reading, register verbatim

`add_dense_grid(filename, config_file_path)` resolves `filename` through
`resolve_file_path` (`Header.h`) for the `DXReader::read_from_file` call **only**.
The original, unresolved string is what goes into `fname_to_gridkey_` and
`grid_keys_`.

That split is deliberate. `SimSystem::assign_particle_type_ids()` looks grids up by
the name as written in the config (`get_grid_key(pmf_grid_names[g])`,
`SimSystem.h:584-597`). Registering the resolved absolute path would break every one
of those lookups, since the caller only ever has the config string.

Before this, all eight `ConfigParser.cpp` call sites passed the raw string straight
to `DXReader`, so `gridFile potentials/null.dx` resolved against the **process CWD**
rather than the config file's directory — unlike every other file key in the parser,
which already went through `resolve_file_path`. A fixture therefore only ran from
its own directory.

The one-argument overload is kept and delegates with `""` (process CWD), for callers
that have no config file — the Python path and tests.

**Caveat:** the dedup check keys on the unresolved string, so the same grid reached
by two different spellings (`grids/a.dx` from one config, `../x/grids/a.dx` from
another) loads twice. That was already true and is not worth fixing until grid
memory becomes a problem.
# Tables.h implementation notes

## Angle and dihedral abscissa units

Tabulated angle and dihedral files are written in **degrees**:

```
# angle-20.801-180.000.dat        # dihedral-2.668-0.000.dat
0.000000 102.650772               -180.000000 13.167265
0.100000 102.536747               -179.900000 13.152639
...                               ...
181.000000 0.003168                180.000000 13.167265
```

The kernels index them with `acos(...)` and `atan2(...)` results, which are in
**radians**. `Table::read_file` therefore rescales `X`, `start` and `step_size`
by `PI/180` for `TabulatedType::Angle` and `TabulatedType::Dihedral` only.

Legacy applies the same factor, but folds it into the step instead of the data:
`TabulatedAngle.cu` computes

```
angle_step_inv = 57.29578f * (size-1) / (angle[size-1] - angle[0]);
```

where `57.29578 == 180/PI`. Rescaling the abscissa here is equivalent and keeps
`X`/`start`/`step_size` mutually consistent, which matters because
`check_same_step_size()` derives `start` and `step_size` from `X`.

### What this fixed

Without the conversion the lookup variable was in radians while the table was
laid out in degrees, so an angle table spanning 1811 rows was only ever sampled
over its first ~31 rows: `theta` in `[0, PI]` divided by a step of `0.1`
(degrees, read literally) gives indices `[0, 31.4]`. Every angle in the system
therefore evaluated at roughly `theta = 0` regardless of its real geometry —
a nearly constant energy of ~102.65 kcal/mol per angle and an essentially
meaningless force.

On the rotor fixture (`~/server3/rotor/rotor_center_debye_30ms2`, 5682 angles)
that showed up as ~571,000 kcal/mol of angle energy, against a v1 total system
potential of about -1,400. Dihedrals had the same defect: the table starts at
-180 (degrees) while the kernel looks up `phi + PI` in radians, so the sampled
index sat near the row for 0 degrees.

The `Table(TabulatedType::Dihedral)` constructor seeds `start = -PI`, but
`check_same_step_size()` overwrites it with `X[0]` once the file is read, so
that seed never protected against this.

## DeviceParticleManager.h — Z-order reorder (ENABLE_ZORDER_REORDER)

Stage 2 of the off-by-default reorder plan (`buzzing-tickling-music.md`). Every
addition is gated on `ENABLE_ZORDER_REORDER`; flag-off, the class is byte-identical
to before (no scratch buffers, no un-permute branch).

**`permute<Sorter>(sorter)` is a template method on purpose.** DeviceParticle lives
in the widely-included `DeviceParticleManager.h`. Making permute a template keeps
`ZOrderSort.h` (radix sort, Morton, adaptive kernels) out of this header — the
sorter type is only needed where permute is instantiated (Patch.cpp, Stage 4),
which already includes ZOrderSort via the pairlist. It only calls
`sorter.reorder_data<T>(in, out, n)`.

**Double-buffer, not in-place.** `reorder_data` is out-of-place, so permute reorders
each field into a persistent scratch buffer of the same element type and `std::swap`s
the handles (DeviceBuffer has move ctor + move assign). One scratch per element type
(`reorder_scratch_vec3_/int_/uint32_`) is reused across same-type fields sequentially:
pos→mom→orient share the vec3 scratch, id→type_id share the int scratch. Persistent
(sized to capacity) so no realloc per reorder — speed over memory.

**ForceEnergy is not permuted.** It is cleared every step before force accumulation,
so its stale contents are irrelevant; only the 6 persistent fields move.

**Field read order is safe.** Each `reorder_data(field_, scratch, n)` reads `field_`
before its swap, and `sorted_indices_` indexes the pre-reorder slots. Since every
field is still in old order when its own reorder runs, there is no cross-field
contamination.

**Output un-permutation (`copy_to_host_unpermuted`).** After a reorder the in-memory
slot order is Morton, not the user's atom order, so DCD/restart would scramble atoms
for VMD/MDAnalysis. `reordered_` gates a scatter-by-`global_id` on the host: slot i's
data goes to output position `global_id[i]`. Scattering by the id anchor (not by
`inverse_indices_`) is correct across *repeated* reorders — inverse_indices_ only maps
the latest pre-sort order, whereas global_id is the invariant original index.
Single-patch invariant: global_id is a dense permutation of [0,count); the loop throws
if that is violated.


