# AnalyticalPairKernels.h implementation notes

## (2026/9/12) DEAD and absorbed into Pairwise.h

## One kernel, not one per potential

Every analytical nonbonded term runs inside a single `AnalyticalPairKernel`
rather than a kernel per potential. Two reasons:

1. **One explicit CUDA instantiation.** A concrete (non-template) kernel needs
   exactly one entry in `NonbondedInstantiations.cu`. Templating on the
   potential, or shipping one kernel per term, multiplies that list and each
   omission is a runtime `NotImplementedError` from the stub in
   `KernelHelper.cuh` rather than a compile error.
2. **The geometry is the expensive part and is shared.** `CalcDistance::compute`
   does the wrap and the square root once, and every enabled term reuses it.
   Separate launches would repeat that work and re-walk the pair list.

The terms are selected by the `enabled_terms` bitmask
(`AnalyticalPairTerm`).

### Does the mask branch diverge?

No. `enabled_terms` is a member of the functor, copied by value at launch, so
it is identical for every thread — a warp-uniform predicate, which costs a
test and no divergence. The branches that genuinely diverge are the per-pair
ones (`i >= num_pairs`, the cutoff test, the near-zero distance guard), and
those are inherent to any pairwise kernel.

Templating on the mask would let dead terms compile out entirely, but that
reintroduces exactly the instantiation explosion point 1 exists to avoid.

## Sign convention

`force_magnitude` is @f$-dU/dr@f$ — positive when the pair repels — matching
`AnalyticalBondComputer` and `TabulatedNonBondedComputer`. It is applied as
`-force` to the first particle and `+force` to the second, along the
first-to-second unit vector, so a repulsive pair pushes them apart.

`DebyeHuckelPotential`, `OnckElecPotential`, `GaussianPotential` and
`ColumbPotential` all already return @f$-dU/dr@f$.

`SoftcoreForceKernel::softcoreForce` does **not**: it returns @f$dU/dr@f$
divided by @f$r@f$, because it was written to scale `r_ij` directly rather than
a unit vector. The kernel converts with `-fe.force_magnitude * distance`.
That mismatch is a trap for anyone reusing `softcoreForce` elsewhere.

## Energy

Each term's energy is summed and then split half to each endpoint, so a sum
over particles gives @f$\sum_{ij} U_{ij}@f$ with no double counting — the same
accounting as `TabulatedNonBondedComputer` (see
`Unit_test/Interactions/NonbondedEnergy.md`).

## Per-type parameters

Charges come from `ParticleTypeView::charge`. Softcore additionally reads
`radius` and `eps`, combining them per pair with an arithmetic mean radius and
a geometric mean epsilon. The other potentials carry their parameters as
members of the potential struct, since they are global to the run rather than
per type.

## Superseded

`ColumbForceKernel` (Columb.h) and `SoftcoreForceKernel::operator()`
(Pairwise.h) predate this and are not launched anywhere. `SoftcoreForceKernel`'s
`operator()` computed a force and discarded it without writing any output.
`ColumbForceKernel` applied its force with the sign inverted, so like charges
attracted; that was fixed before it was superseded. Only
`SoftcoreForceKernel::softcoreForce` is still used, as the softcore math.
# GridGridKernels.h — implementation notes

Rigid-body grid–grid force/torque: one thread per `rho` voxel, block-wide
shared-memory reduction, `atomicAdd` into one (force+energy, torque) accumulator
per grid pair. Ported from legacy `ComputeGridGrid.cu`.

## Per-voxel energy: legacy is wrong, mars2 diverges deliberately

Resolved 2026-08-14. **mars2 is correct; legacy's grid–grid energy is a bug.**

Legacy computes:

```cpp
force[tid] = fe;              // fe.e = u(x), the sampled potential
//force[tid].e = fe.e;        // the correct line, left commented out
const float r_val = rho->val[r_id];
force[tid].f = basis_u_inv.transpose().transform( r_val*(force[tid].f) );
force[tid].e = r_val;         // overwrites the potential with the density
```

so its reduced energy is `Σ r_val` — the total density of `rho`. That is
**constant regardless of the two grids' relative pose**, so it cannot be an
interaction energy.

mars2 uses `r_val * sample.value`, the correct discretization of

```
E = ∫ ρ(x) u(x) dx  ≈  Σ_voxels ρ_i · u(x_i)
```

The clincher is internal consistency: both codes compute the force as
`r_val * (-∇u)`, and force must be `-∇E` of the *same* `E`. Only `r_val * u`
satisfies that. Legacy's force is right while its energy is not derived from the
same functional.

Two signs legacy's energy path was never finished rather than deliberately chosen:
the correct line sits right there commented out, and the energy accumulation
inside the reduction loop is commented out too
(`//if(get_energy) force[tid].e = force[tid].e + force[oid].e;`). This fits
`rb_energy.dat` still reporting hardcoded zeros — nothing downstream ever consumed
the value, so the bug was never visible.

**Consequence for validation:** grid–grid energies will not match a v1 reference,
by design. Forces and torques should. Do not "fix" mars2 to reproduce v1's number
here.

## Force transform

`sample.gradient` is raw `∂V/∂x`; force is `-gradient`, then transformed from `u`'s
index space to the lab frame via `basis_u_inv^T`, matching legacy's
`basisInv.transpose().transform(r_val * fe.f)`.

Torque is about `rho`'s own origin (`r_pos` is the offset from `origin_rho`), in
the lab frame.

## Shared memory contract

The caller must set `KernelConfig::shared_memory = 2 * block_size.x *
sizeof(Vector3)` — one force+energy array and one torque array, one `Vector3` each
per thread — and `block_size.x` to a power of two. Legacy used 128 (`NUMTHREADS`).

# Pairwise.h — per-pair table_idx precompute

## ResolvePairTableKernel
The exclusion test and the type-pair → table/form lookup are **static between
rebuilds**, so they are resolved once per pairlist rebuild into a per-pair
`table_idx` (`-1` = excluded, no table, or non-tabulated) instead of every step
in the force kernel. This removes, from the 60k×/step hot path, two scattered
`type_ids` gathers, two matrix loads, and a variable-length CSR exclusion scan
per pair — the bulk of v2's per-pair overhead vs v1 (which precomputed the same
thing at pairlist-build time). Runs at rebuild frequency (~292× vs 60,060× for
the profiled npc_6enl_beads case). **P1 type-switching must force a rebuild** so
the table is re-resolved.

## Cutoff stays in the force kernel
The cutoff check cannot move to resolve time: the pair list is built at a larger
radius (interaction cutoff + skin) so it survives several steps, so it holds
pairs past the interaction cutoff whose tabulated lookup would be out of range
(a spurious interaction). Positions change every step, so the cutoff must be
re-tested every step — removing the skin is exactly what would force a rebuild
every step.

## Scatter atomics: never add `.t` unless energy is wanted (2026-09-11)

`atomic_add(Vector3_t*)` issues **four** `atomicAdd`s, one per component. In the
force path `.t` is always zero, because `operator*` and unary minus both build
their result through the three-argument ctor which zeroes `t`. SASS confirmed it:
`ATOMG.E.ADD.F32 ... [R2.64+0xc], RZ` — an atomic adding the zero register.

v1's `atomicAdd(Vector3*)` (`CudaUtil.cuh:88`) adds three components. So v2 was
sending 8 atomics per pair where v1 sends 6, and the 8/6 ratio predicted v2's
measured 1.39× per-launch regression against v1 on this kernel almost exactly.

Fix: `atomic_add_xyz` (Types.h) for the force-only path, and pack energy into
`.t` when it is wanted instead of taking two extra atomics. `get_energy` is a
kernel member, hence warp-uniform, so branching on it costs nothing.

| path | before | after |
|---|--:|--:|
| energy off | 8 | 6 |
| energy on | 10 | 8 |

Measured on the 305k-particle cytoplasm case (`Tests/privite_test/cytoplasm`,
300 steps, RTX PRO 6000 Blackwell), force-kernel median **2322 µs → 1767 µs,
1.31×**. That kernel is ~97% of production GPU time.

### What did NOT work, and why

Replacing the atomics with plain `+=` (racy, measurement only) made the kernel
**34% slower**, 2322 → 3109 µs. `atomicAdd` with an unused return compiles to a
single transaction executed by the ALU in the L2 slice; the line never enters
the SM. A plain read-modify-write is `LDG` + `FADD` + `STG`: two transactions
plus full L2 latency exposed in a register dependency.

**The cost is the number of atomic messages, not their atomicity.** Do not try
to remove atomics; remove *redundant* ones. This is also why the per-particle
full neighbour list is not obviously a win: it would double the distance and
table work to eliminate atomics that are individually cheap.

# Rigid-body kernels — energy on the RB side (2026-10-10)

`RigidBodyView::force` is a 4-component `Vector3` (`// force_energy`), so
`rb.force[i].t` **is** an energy slot; `atomic_add` adds all four components.
Earlier comments claiming "no per-RB energy accumulator" were wrong. Phase-1
`ComputeGridGridForceKernel` (GridGridKernels.h) also reduces `.t`.

`rb.force[].t` = the body's **grid potential**: grid-grid (pair split
half/half) + particle-in-RB-grid. Zero automatically for a body with no grids.
`rb.torque[].t` is unused — don't put energy there.

Lifetime per step: `clear_forces()` zeroes all four components → grid-grid
and particle-grid kernels write it (every step) → RB Langevin uses `+=`
(keeps `.t`) → integrators only read → `RBEnergyReduceKernel`
(RBOperation/RBEnergyKernels.h) reduces it on device at energy-output steps.

Reminder: `Vector3::operator+=` and binary `operator+` skip `.t` (by design);
any reduction that wants energy must add `.t` explicitly.

## RBReduceAttachedForcesKernel (RigidBodyAttachedParticles.h)

- Energy not reduced into `rb.force[].t`: it stays in the particle's
  `ForceEnergy.t`. Legacy `apply_attached_particle_forces` also sums forces
  only. `rb_energy.dat` reports it as a separate "Attached" term, read
  straight from `ForceEnergy.t` by `RBEnergyReduceKernel`.
- `f_acc += fe` is safe: `+=` drops `fe.t`; `cross` uses x/y/z only.
- Torque arm = lab-frame offset from the body origin, so `position` must be the
  point its grids are built about (`referencePoint` config key).
- Particle force is left in place, not zeroed: integrators skip attached
  particles and the next nonbonded pass clears the array, and keeping it makes
  the force/energy output honest about what acted on the particle.

## RBParticleGridForceKernel (RigidBodyParticleGridBatch.h)

- Particle side: atomic scatter into `ForceEnergy` (Pmf.h packing:
  force = -scale*grad, energy = scale*value). Atomic, not plain RMW, because
  several RB/grid candidates can hit the same particle.
- RB side: Newton's-third-law reaction, torque block-reduced about the grid's
  lab-frame origin `O = R_rb*grid.origin + rb.position`, then shifted to the
  body position before the atomic: `τ(r) = τ(O) + grid_shift × F`,
  `grid_shift = R_rb*grid.origin` (2026-10-10). Port of legacy
  `RigidBody.cu:369`. Before this the torque was added about `O` — correct
  only for grids with origin 0, which is why the unit tests (origin 0) never
  caught it.
- Energy (2026-10-10): **half/half** — `fe.t = 0.5 * scale * u` on the
  particle, same half block-reduced into `rb.force[].t` (explicit `.t` add in
  the force reduction). Same convention as every other two-partner term
  (pairwise 0.5, bonds 0.5, RB-RB grid pairs 0.5), so `energy.dat` +
  `rb_energy.dat` counts the pair energy exactly once.
  **Diverges from legacy**: `computePartGridForce` gave the full `fe.e` to the
  particle *and* to the RB (double count). v1 particle PE and v1 RB PE will
  each be higher by half this term.
  Not mass-weighted: potential energy belongs to the pair; mass only governs
  how kinetic energy partitions.
- Particle-type filter (2026-10-10): a particle type samples an RB potential
  grid only if its `rigidBodyPotential` keys name that grid's key
  (`RigidBodyType::potential_grid_keys`). Port of legacy
  `RigidBodyType.cu:192-241` (`partRigidBodyGrid`). Before this, the key was
  parsed into `ParticleType::rigid_body_potential_keys` but never used, so
  every particle felt every RB potential grid.
  - Host (`prepare_particle_grid_dispatch`): a candidate is emitted only if
    ≥1 type uses its grid (legacy: `numParticles[i] == 0` → skip); each
    candidate carries a `[particle type]` row of `uint8_t` flags.
  - Device: `if (!mask[type_id[p]]) continue;` — one byte read per particle,
    no atomics for skipped ones.
  - Mask by type, not legacy's per-grid particle-index lists: index lists
    would need remapping on every Z-order reorder (as
    `remap_attached_particle_indices` does); a type mask is reorder-proof.
- `GridTerm::scale_slope` (SMD) not applied on this path yet — planned.
- Launch = num_candidates * blocks_per_candidate exactly: every candidate has
  the same num_particles, so block→item is div/mod, no prefix sum / search.

## RBGridBatchedForceKernel (RigidBodyGridBatch.h)

- Launch sized to worklist *capacity*; `total_blocks` is device-resident (no
  host round-trip), blocks past it early-return. See
  `RigidBodyManager::compute_grid_grid_forces`.
- Each item's rho voxels grid-stride over its `num_blocks`; reuses
  `gridgrid_detail::grid_grid_voxel_force_torque` + Phase-1 block reduction.
- rb_j (when not an external PMF) = closed-form reaction from rb_i's totals,
  no second grid pass: `F_j = -F_i`, `T_j(O_u) = -(T_i(O_rho) + origin_offset x F_i)` —
  torque transfer to a new reference point, valid since the reaction acts at
  the same world point.
- Reference-point shift (2026-10-10): the voxel loop reduces torque about the
  grid origins; the integrators want it about body positions. So
  `T_i(r_i) = T_i(O_rho) + rho_shift × F_i` and
  `T_j(r_j) = T_j(O_u) + u_shift × F_j`, with `rho_shift = R_i*rho.origin`,
  `u_shift = R_j*u.origin` (0 for PMF) precomputed in `RBGridCullKernel`.
  Expands to legacy's `-T_i(r_i) + (r_j - r_i) × F` (`RigidBodyController.cu`
  `processGPUForces`) without reading positions, and stays consistent with the
  unwrapped `origin_offset` the force uses. Before this, torque was about the
  grid corner: wrong by `(R·o)×F` for any grid with nonzero origin.
- Energy (2026-10-10): reduced **unconditionally**. `f.t` is computed by the
  voxel helper anyway, and `atomic_add` already issues the `.t` atomic (adding
  0 when unused) — one atomic per block, not per pair, so gating saves ~nothing
  (contrast Pairwise.h, where `.t` atomics are per pair and gating paid). Only
  the D2H readback + host sum is tied to energy output (`write_energy_output`).
  `f_acc.t += f.t` in the voxel loop, explicit `.t` add in the tree reduction
  (`+=` skips it).
- Pair split: RB-RB pair energy goes half to rb_i, half to rb_j (`reaction.t`
  set explicitly, since unary `operator-` drops `.t`). External PMF
  (`rb_j < 0`) puts all of it on rb_i. Sum over bodies = total pair energy.
  Unlike the two kernels above, this energy is recorded nowhere else.
- `t.t` from `grid_grid_voxel_force_torque` is always 0 (`cross` returns
  x/y/z only); there is no torque energy.
- Values won't match v1 `rb_energy.dat` — v1's grid-grid energy is wrong (see
  GridGridKernels.h section above).

# Pmf.h — particle PMF energy (2026-10-10)

Particle PMF energy never reached `energy.dat`: `launch_PMF` built
`ComputePMFKernel` without `get_energy`, and the BD-fused path's `+=` dropped
`.t`. Now unconditional (no atomics on this path, so gating saves nothing):
- `compute_position_dependent_force` always fills `.t = Σ scale·u`;
  `get_energy` param kept for old call sites, `[[maybe_unused]]`.
- `ComputePMFKernel` uses `accumulate()` (adds `.t`); its `get_energy` member
  is kept, unused.
- Uniform E-field force has no energy term (−qE·x is not defined under PBC).

Full energy to the particle: an external field has no partner to split with.
