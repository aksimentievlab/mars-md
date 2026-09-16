# BondComputer.h implementation notes

Vector conventions used below are described in `BondGeometry.md`.

`TabulatedPotential::compute` returns `force_magnitude = -dU/dx`, where `x` is
whatever coordinate the table is indexed by (distance for bonds, radians for
angles and dihedrals). It is *not* a Cartesian force; each computer applies its
own chain rule.

## TabulatedAngleComputer

The table is indexed by `theta = acos(cos_theta)`, so the chain rule to
Cartesian coordinates goes through `cos_theta` and picks up a `1/sin(theta)`
factor:

```
F_i = -dU/dtheta * d(theta)/dr_i
    = (dU/dtheta / sin(theta)) * d(cos theta)/dr_i
```

With `u = r_i - r_j`, `v = r_k - r_j` (so `ab = -u`, `bc = v`):

```
d(cos)/dr_i = v/(|u||v|) - cos * u/|u|^2
```

which expands to the ported form

```
force1 = (dUdtheta / |ab|) * ( ab * cos/|ab| + bc/|bc| )     # force on i
force3 = -(dUdtheta / |bc|) * ( bc * cos/|bc| + ab/|ab| )    # force on k
force2 = -(force1 + force3)                                  # force on j
```

`sin(theta)` is floored at 1e-3 (legacy does the same) because the `1/sin`
factor diverges at the collinear and fully-folded limits, where `theta` is a
stationary point of `cos` and the Cartesian gradient is genuinely ill-defined.

`force1` and `force3` are the forces *on* particles i and k, so they are added
with a positive sign; only the central particle takes a negation.

### What this replaced

Until 2026-08-13 this computer used

```
force1 = ab.cross(bc) * force_magnitude
force3 = bc.cross(ab) * force_magnitude
```

`ab x bc` is normal to the plane of the three particles, so it cannot change
the angle at all; and since `bc x ab == -(ab x bc)` the central particle
received exactly zero force. The `1/sin` factor was absent as well. The kernel
was wired into `Patch::calculate_bonded_forces` but no configuration in the
tree had a nonzero angle count, so nothing exercised it.

## TabulatedDihedralComputer

`geom.f1/f2/f3` are already the Cartesian gradients of `phi`, so the only work
here is the table lookup and the `f1, f2-f1, f3-f2, -f3` distribution.

The lookup passes `phi` **unshifted**. Legacy adds `BD_PI` first because its
`TabulatedDihedralPotential` has no notion of where the table starts — it
indexes from row 0, so the caller must supply the offset from the table's
lower bound. `TabulatedPotential::compute` here does `w = (dx - start) *
step_inv`, and a dihedral table's `start` is already `-pi` (the files run
-180..180 degrees; see `Objects/Tables.md`). Carrying legacy's `+ BD_PI` over
on top of that shifted every lookup by a full turn, landing on the `+pi` end of
the table instead of the intended angle. On the rotor fixture that inflated the
dihedral energy from ~2,000 to ~47,000 kcal/mol.

The four applied forces must sum to zero. The previous code applied `-f1` to
particle i while applying `f2-f1`, `f3-f2`, `-f3` to the rest, leaving a net
force of `-2*f1` on the group — a sign convention half-flipped. Particle i
takes `+f1`, matching legacy.

Legacy's near-collinear force zeroing and `+/-1000` clamp are deliberately not
ported; see `BondGeometry.md`.
# BondGeometry.h implementation notes

## Bond vector convention

`AngleGeometry` and `DihedralGeometry` build their bond vectors *forward along
the chain*:

```
ab = r_j - r_i
bc = r_k - r_j
cd = r_l - r_k
```

Legacy MARS (`mars.dev/src/TabulatedMethods.cuh`) builds them *backwards*:

```
ab_legacy = r_i - r_j = -ab
bc_legacy = r_j - r_k = -bc
cd_legacy = r_k - r_l = -cd
```

Every formula ported from legacy has to be transposed into this convention.
Quantities that are even in the bond vectors carry over unchanged; quantities
that are odd do not.

## Dihedral angle sign

Under the sign flip:

- `crossABC = ab × bc` is unchanged (two sign flips cancel).
- `crossBCD = bc × cd` is unchanged.
- `crossX = bc × crossABC` **flips sign** (only one factor flips).

So `cos_phi` is unchanged but `sin_phi` is negated relative to legacy. Legacy
computes `angle = -atan2(sin_phi_legacy, cos_phi)`; with forward vectors the
equivalent is `atan2(sin_phi, cos_phi)`, which is also the IUPAC convention.

Carrying legacy's leading minus over unchanged (as the code did until
2026-08-13) yields `-phi`. That is invisible for a symmetric tabulated
potential but samples the mirror image of any potential that is not symmetric
about 0, and it is inconsistent with `f1`/`f2`/`f3`, which are unchanged by the
flip and therefore still represent `d(phi)/dr` in the IUPAC sense.

## Collinear i-j-k (no guard here)

`f1` and `f3` carry `crossABC.rLength2()` and `crossBCD.rLength2()`, so the
gradients diverge as i-j-k or j-k-l goes collinear. That divergence is a real
property of the dihedral coordinate, which is undefined there, not an artifact
of this formula — the correct force genuinely is large.

Legacy zeroes the force below `sin^2 = 1/100` and clamps it at `+/-1000`.
Neither is reproduced. Both are stability hacks rather than physics, and the
zeroing is the more damaging of the two: it makes the force discontinuous over
a ~5.7-degree-wide band of configuration space, which is non-conservative.

One thing to be aware of at *exact* collinearity (measure zero in floating
point, so not reachable from ordinary dynamics): `cos_phi` and `sin_phi` are
both `0/0`, `dihedral_angle` is NaN, and `TabulatedPotential::compute` then
does `static_cast<int>(floorf(NaN))`. The periodic modulo keeps the index in
range, so the result is a finite but arbitrary force rather than a NaN or an
out-of-bounds read. Note also that `Vector3_t::rLength2()` returns 0, not
infinity, for a zero-length vector, so any guard written in legacy's divided
form (`|ab|^2 |bc|^2 * rLength2 > 100`) evaluates to `0 > 100` and misses this
case entirely; the multiplicative form
(`|ab|^2 |bc|^2 > 100 |ab x bc|^2`) is the one that catches it, should a guard
ever be wanted.

# HarmonicRestraintComputer implementation notes

## Harmonic restraints

v1 applied these in `ComputeForce.cuh::computeHarmonicRestraints`:

```cuda
const Vector3 dr = sys->wrapDiff(pos[id] - r0[i]);
Vector3 f = -k[i]*dr;
atomicAdd( &force[ id ], f );
```

`HarmonicRestraintComputer` reproduces that exactly, assembled from the two
pieces that already existed rather than open-coding the arithmetic:

- `CalcDistance::compute(anchor, position, pbox)` for the geometry. This is the
  `(Vector3, Vector3)` overload added to `Interactions.h` next to the original
  `(positions, int2)` one — a restraint's anchor is not a particle, so there is
  no index to look up, but the wrapping, the length and the 1e-6 unit-vector
  guard are identical and worth sharing. The index overload now forwards to it.
- `AnalyticalForceComputer<0>` for the force, called with `params = {k, 0}`. A
  restraint *is* a harmonic bond of zero equilibrium length, so
  `force_magnitude = -k*(d - 0) = -k*d`; multiplying by the anchor->particle
  unit vector gives `-k*dr`, v1's expression with v1's sign.

The 1e-6 guard also disposes of the divide-by-zero at `d == 0`: `unit_vector` is
zero there, so the force is zero, which is the right answer for a particle
sitting on its anchor.

Energy is `0.5*k*d^2` from the same computer. Unlike the bond computers it is
**not** halved before accumulating — a bond splits its energy between two
particles, a restraint has only one endpoint that is a particle. v1 accumulated
no restraint energy at all.

## Why nothing applied them until 2026-08-15

Three independent breaks in the same chain, each of which alone was enough:

1. `BondConfigReader::read_file()` had no `RESTRAINT` branch, so every line of
   an `inputRestraints` file was read and discarded (`IO/dev_notes.md`).
2. `SimManager::init()`'s guard on `pending_bonded_interactions_` did not test
   restraints, so a config carrying only restraints never reached `sys_state_`
   (`src/dev_notes.md`).
3. No kernel read the device buffers back. `copy_from_host` had always uploaded
   the ids, anchors and spring constants, but `restraint_particle_ids()`,
   `restraint_positions()` and `restraint_spring_constants()` were referenced
   nowhere else in `src/`.

`Tests/privite_test/npc_6enl_beads` is what surfaced it. v1 holds its rigid body
to 0.87 A net displacement over 98 steps; v2 diffused 8.2 A away, because the
437 restraints on that body's attached particles were doing nothing.

# Cross-cutting notes

## Analytical.h sign convention

Every `AnalyticalForceComputer<N>::compute` returns `force_magnitude` along the
pair's `x -> y` unit vector, negative meaning attractive. Harmonic is
`-k*(distance - r0)`: negative when stretched, pulling the pair back together.
Morse follows the same convention
(`-2*D0*a*exp(-a(r-r0))*[1-exp(-a(r-r0))]`, negative for `r > r0`), as does the
non-bonded `Nonbonded/AnalyticalPairKernels.h`.

`AnalyticalBondComputer` consumes it as `force = unit_vector * force_magnitude`,
applying `-force` to `x` and `+force` to `y`.

## What `tables`, `table_indices` and `forms` hold

`tables` has one entry per distinct bond *type*, as loaded by `TablesRegistry` — not one per bond instance. `table_indices[i]` selects which entry bond `i` uses (see `DeviceBondedInteractions::bond_table_indices()`).

`forms[i]` is an `InteractionForm` (Grid=0, Tabulated=1, Analytical=2). The
`Tabulated*Computer`s process Tabulated entries and skip the rest; Analytical
bonds go through `AnalyticalBondComputer` in a separate launch.

## Where the launch helpers live

`launch_tabulated_bonds/_angles/_dihedrals`, `launch_analytical_bonds` and
`launch_harmonic_restraints` moved to `Interactions/DeviceBondedInteraction.h`
on 2026-08-15. `BondComputer.h` defines the device functors; each launcher pairs
a functor with the `DeviceBondedInteractions` buffers it reads, so it belongs
beside those buffers. `Patch::calculate_bonded_forces` is the only caller and
already included that header for its `device_bonded_` member.

## Explicit instantiation

`BondedInstantiations.cu` forces `launch_cuda_kernel` for every computer in this directory into a real CUDA translation unit, with matching `extern template` declarations at the bottom of each header.

Without them, any host-only `.cpp` that calls a launcher — or calls `Patch::calculate_bonded_forces`, which calls all of them — implicitly instantiates the non-CUDA stub in `KernelHelper.cuh` and throws `NotImplementedError` at runtime. This applies to the concrete computers (`TabulatedBondComputer`, `HarmonicRestraintComputer`, ...) exactly as much as to the templated `AnalyticalBondComputer<N>`: being a non-template type is no exemption, because it is `launch_cuda_kernel` that is the template.

## Bonded force clamp (`kMaxBondedForce`, `clamp_bonded_force`)

ARBD v1 guards the dihedral force in `TabulatedMethods.cuh`; MARS had nothing. Without
a guard the nupod hinged-nup98 Langevin run shows intermittent KE spikes
(295k -> 399k -> 507k kT) decaying back, one dihedral transiting near-collinear per
spike. With all bonded terms disabled the same run sits at exactly 1.00x equipartition.

v1 does it by zeroing inside a collinearity band, plus a magnitude cap:

    force = (ab2*bc2*crossABC.rLength2() > 100 || bc2*cd2*crossBCD.rLength2() > 100) ? 0 : fe.x;
    if (force > 1000) force = 1000;

`ab2*bc2/|ab x bc|^2` is `1/sin^2` of the ab-bc angle, so that fires within ~5.7 degrees
of collinear — where `f1 = -|bc| * crossABC.rLength2() * crossABC` (magnitude
`|bc|/|ab x bc|`) diverges.

**MARS clamps instead of zeroing, and clamps the assembled force, not the table term.**
Two reasons. Zeroing a force that is otherwise correct is itself non-conservative, and
it discards a real restoring term. And clamping `fe.force_magnitude` would not help:
the divergence lives in the geometric prefactor `|bc|/|ab x bc|`, not in `dU/dphi`, so
capping the table term leaves `f1` unbounded.

`clamp_bonded_force` scales a vector back to `kMaxBondedForce` keeping its direction,
and is applied to `f1`/`f2`/`f3` (dihedral) and `force1`/`force3` (angle). Momentum is
still conserved exactly: the applied forces telescope
(`f1 + (f2-f1) + (f3-f2) + (-f3) = 0`, and `force1 + -(force1+force3) + force3 = 0`)
whatever the clamped values are. The end particles therefore carry the clamp bound
while the middle ones receive differences and are bounded by twice it.

The clamp is inert for healthy geometry: the near-collinear unit-test triple sits at
`|f1| = |bc|/|ab x bc| * SLOPE = 50`, far below the bound, which is why
`Tabulated dihedral stays exact on a near-collinear triple` still holds.

Exactly-zero cross products need no guard — `Vector3::rLength2()` returns 0 rather than
inf/NaN for zero length, so `f1`/`f2`/`f3` collapse to zero on their own. Residual
hazard left alone (v1 has it too): a *perfectly* collinear quadruplet makes
`cos_phi = 0/0 = NaN`, so `dihedral_angle` is NaN and the table lookup indexes on
`floor(NaN)`. The force is zero either way, but a NaN energy can still be accumulated.

**Angle: do NOT clamp `dUdtheta`.** This was tried and reverted. `dUdtheta =
(dU/dtheta)/sin(theta)` does diverge as theta -> 0 (the nupod tables rest at 180 deg,
so dU/dtheta stays at -33.6 while sin -> 0, giving -1913 at 1 deg and -33577 at 0),
but that divergence is *cancelled*: the bracket it multiplies collapses like
`n_hat*sin(theta)/|u|`, so the force tends to `(dU/dtheta)*n_hat/|u|` and stays small
(0.2-1.8 at the smallest angles actually observed). Clamping `dUdtheta` breaks the
cancellation and makes the force non-conservative exactly where it fires.

Measured against a numerical gradient of the same tabulated energy, over the 400
smallest-theta angles of a real frame: uncapped gives median relative error 5e-8 and
max 0.000; with a 1000 clamp, two angles break by up to 43%. That is energy injection,
not protection. `Tabulated angle force tracks finite differences across a theta sweep`
in `Unit_test/Interactions/TabulatedBonded.cpp` pins this down to 0.5 deg.

The residual hazard is the `sin_floor = 1e-3` already in the kernel, which is the same
kind of non-conservative clamp but only bites within 0.057 deg of collinear. No angle
in the nupod trajectory reaches it (min theta observed 0.34 deg).

**What actually triggered it.** One bond — atoms 8924-8925, two mass-30 S005 beads on
`wlcbond-2.560-12.000.dat`, k ~ 3875 kcal/mol/A^2 — has `dt*omega = 6.58` at
`dt = 2e-5 ns`, well past the velocity-Verlet stability limit of 2. It is the only such
bond out of 460,316 (every other is <= 0.48). v1 never sees it: v1 launches bonds over
`numBonds/2` on the assumption that the file is fully bidirectional, but this file has
66,668 single-listed bonds, so the first `numBonds/2 = 426,982` *lines* cover only
238,840 *unique* pairs — v1 silently omits 221,476 unique bonds, 48% of the topology,
including this one (line 459,382, past the cutoff). Split by that cutoff, the bonds v1
sees have max `dt*omega` 0.475 and zero unstable; the bonds v1 never sees contain the
single unstable one. v1's apparent stability on this system is partly an artifact of
simulating half the bonded topology.

That bond is a modeling/timestep issue, not a code one: k=3875 on mass-30 beads needs
dt <~ 6 fs. But it is *not* what drove the 4x heating — with bonds+dihedrals enabled
and angles off the run is flat, so the unstable bond alone is not sufficient.
