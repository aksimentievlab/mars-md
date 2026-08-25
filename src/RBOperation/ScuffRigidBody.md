# ScuffRigidBody.h — implementation notes

Host-side coupling between MARS rigid bodies and a scuff-em BEM solve. One SCUFF
surface per plasmonic rigid body; MARS supplies the pose, and the force/torque
comes back and lands in `DeviceRigidBody::external_force/_torque` via
`ApplyExternalForcesKernel` (ApplyHostForce.h).

The solve is *not* on the MD step loop — it runs concurrently and publishes
whenever it finishes, so the bodies carry the most recent published load rather
than a current one. The synchronisation that implies is not written yet; see
Open.

## Data flow

```
HostRigidBodyData
  -> ScuffRigidBody::update_from_bd()        pose of the plasmonic subset
  -> ScuffForceCalculator::compute_forces()  UnTransform / Transform / BEM / PFT
  -> ScuffExternalForceUpload::push()        H2D + scatter kernel
  -> RBLangevinForceKernel / RBDLM           reads external_force[idx]
```

`plasmonic_particle_id[i]` is the slot in `HostRigidBodyData`, which is also the
device SoA slot (`copy_from_host` preserves order). `surface_tag[i]` is the
surface index in the loaded `.scuffgeo`. The two are independent indices and
both are needed.

## Corrections against the previous draft

- **PFT indices.** Force/torque were read from `pft_buffer[1..3]` / `[4..6]`.
  `PFTOptions.h` defines `PFT_PABS=0, PFT_PSCAT=1, PFT_XFORCE=2..4,
  PFT_XTORQUE=5..7`, so the old force triple was `(PSCAT, Fx, Fy)` and the
  torque triple was `(Fz, Tx, Ty)`. Now uses the named macros.
- **Surface addressing.** `RWGGeometry::Transform(GTComplex*)`
  (RWGGeometry.cc:888) resolves each transformation through
  `GTC->SurfaceLabels[ns]`, not through GT list order. The draft left
  `SurfaceLabels` empty while sizing `GTs` to `NumSurfaces`, so every step
  indexed past the end of an empty `sVec`. Replaced with a direct
  `Surfaces[ns]->Transform(&gt)` plus `SurfaceMoved[ns]=1` — exactly what
  `Transform(GTComplex*)` does once the label lookup succeeds, minus a
  per-step heap allocation of `NumSurfaces` `GTransformation`s and a `GTComplex`.
- **Pointer casts.** `static_cast<const float*>(Vector3*)` is ill-formed, and
  `Matrix3 orientations = scuff_rb.orientation.data();` assigned a pointer to a
  value. Both replaced by reading `Vector3`/`Matrix3` elementwise.
- **Solve.** `HMatrix` has no `Solve`; scuff-scatter's sequence is
  `AssembleBEMMatrix -> LUFactorize -> AssembleRHSVector -> LUSolve`. The draft
  also reallocated the BEM matrix, RHS and KN vectors on every call while the
  preallocated workspaces went unused; assembly now writes into them.
- **`clear_force`.** `clear()` on the force vectors dropped the entry count and desynchronised them from `surface_tag`; now assigns zeros in place.

## Ordering constraint

Transform first, then assemble. Both the BEM matrix and the RHS depend on the
transformed mesh, and scuff-scatter only pre-assembles diagonal `T` blocks
before a transform because those are the mate-invariant ones; the off-diagonal
`U` blocks are assembled after. With a single full `AssembleBEMMatrix` per step
there is nothing to hoist, so the simple order is correct.

`UnTransform()` composes-and-undoes through `RWGSurface::GT`, so each solve
starts from the pose the geometry had when read.

## Why `check_geo_format()` rejects in-block DISPLACED/ROTATED

A `DISPLACED`/`ROTATED` line inside `OBJECT...ENDOBJECT` is parsed into
`RWGSurface::OTGT` (RWGSurface.cc:166-186), applied to the vertices at birth
(RWGSurface.cc:427) and never un-applied — `UnTransform()` rewinds only
`RWGSurface::GT`. The MARS pose would then compose on top of that placement
rather than replace it, offsetting every body by the `.scuffgeo` displacement
with nothing to signal it.

The check reads `OTGT != nullptr` rather than re-parsing the file: SCUFF has
already done the parsing, and the flag survives (it is freed only in
`~RWGSurface`), so comments, casing and multiple `DISPLACED` lines per block all
resolve correctly. It is strict about non-null, so an explicit no-op
`DISPLACED 0 0 0` also trips it — `GTransformation::IsIdentity()` would let that
through if it ever turns out to matter.

Meshes still come from `MESHFILE` as usual; the constraint is only that the
`.scuffgeo` must not *place* them.

## Units

| quantity | SCUFF | MARS | factor |
| --- | --- | --- | --- |
| length | micron | Angstrom | `ANGSTROM_TO_MICRON = 1e-4` |
| force | nanonewton | kcal/mol/Angstrom | `1000 / PNPERKCALMOL` = 14.393 |
| torque | nanonewton·micron | kcal/mol | above × `1e4` = 1.4393e5 |

`constants::PNPERKCALMOL = 69.479` (pN per kcal/mol/Å) is the existing MARS
constant; the SCUFF units are documented in
`applications/scuff-scatter/OutputModules.cc` (`WritePFTFilePreamble`) and
`PFTOptions.h` (`TENTHIRDS`).
`omega` is in SCUFF's own frequency unit, 3e14 rad/sec = c / 1 micron. Add to constants

## Open

- **No reciprocal coupling into `HostRigidBodyData`.** The plasmonic force is pushed to the device only. That is the intended path (todo_rb.md step 1: never copy the whole external force/torque SoA from/to host), but it means the host SoA never shows the external load.
- **Concurrency is not implemented.** The solve is meant to run on its own
  thread and publish whenever it finishes, not once per MD step, but nothing
  here is synchronised yet. Three things this needs:
  - `update_from_bd()` must take a *snapshot* into a buffer the worker owns.
    Its source is now live — `RigidBodyManager::gather_to_host()` refreshes
    `SystemState`'s SoA in place, and `SimManager::gather_rigid_body_data()`
    drives it — but that buffer belongs to the MD thread, so the next gather
    would overwrite poses mid-solve. The copy stays; it just copies from
    current data instead of the init-time snapshot it used to read.
  - `ScuffExternalForceUpload::push()` must stay on the MD thread. It launches
    onto `resource_`'s stream, so calling it from the worker races the MD
    thread's own launches. The worker should publish a finished force set and
    the MD thread should pick it up and push.
  - `compute_forces()` fuses transform + solve + readback; split it so the
    transform consumes the snapshot rather than the live struct.

  The staleness semantics are already handled by design: `DeviceRigidBody` keeps
  `external_force_` out of `clear_forces()` precisely so a published load holds
  until its owner overwrites it, which is what a lagged EM force needs.
- **Cost.** One full `AssembleBEMMatrix` + `LUFactorize` is O(N_edges^3), which
  is what sets the publish interval. Caching the diagonal `T` blocks per surface
  (invariant under rigid motion of that surface) and reassembling only the `U`
  blocks — the `NumTransformations>1` path in scuff-scatter.cc:404-450 — shortens
  it, and needs `AssembleBEMMatrixBlock`, `InsertBlock`/`InsertBlockAdjoint` and
  `BFIndexOffset`.
- **Not built.** `src/RBOperation/CMakeLists.txt` exists but is deliberately not added by `src/CMakeLists.txt` yet; nothing here has been compiled.
