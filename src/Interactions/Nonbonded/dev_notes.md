# AnalyticalPairKernels.h implementation notes

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

## Per-voxel energy: legacy is wrong, arbd2 diverges deliberately

Resolved 2026-08-14. **arbd2 is correct; legacy's grid–grid energy is a bug.**

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

arbd2 uses `r_val * sample.value`, the correct discretization of

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
by design. Forces and torques should. Do not "fix" arbd2 to reproduce v1's number
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
