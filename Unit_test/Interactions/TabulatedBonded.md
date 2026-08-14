# TabulatedBonded.cpp notes

Covers `TabulatedAngleComputer` and `TabulatedDihedralComputer`
(`src/Interactions/Bonded/BondComputer.h`). Both kernels were wired into
`Patch::calculate_bonded_forces` long before any configuration in the tree had
a nonzero angle or dihedral count, so nothing had ever executed them. The bugs
these tests pin down are described in `src/Interactions/Bonded/BondComputer.md`
and `src/Interactions/Bonded/BondGeometry.md`.

## Reference method

The tabulated potential is a pure ramp, `U(x) = SLOPE * x`. Linear
interpolation reproduces a ramp exactly, so `dU/dx == SLOPE` everywhere and the
expected force collapses to

```
F = -SLOPE * d(coordinate)/dr
```

The derivative is taken by central differences of an angle/dihedral formula
written independently in the test file (dot-product form for the angle, the
standard `atan2((n1 x n2).b2hat, n1.n2)` for the dihedral). Neither the table
lookup nor the geometry code under test appears in the reference, and the
finite-difference error is ~1e-8, well under the assertion margins.

Configurations are deliberately generic — no right angles, no axis-aligned
bonds, unequal bond lengths, non-planar — so a formula that only happens to
work in a symmetric arrangement still fails.

## Why the torque check matters

The cross-product form that `TabulatedAngleComputer` used previously produced
an out-of-plane force that still summed to zero, so a net-force check alone
passes on it. The net-torque check and the in-plane check are what catch it.
The dihedral's previous half-flipped sign convention, by contrast, left a net
force of `-2*f1` on the group, which the net-force check catches directly.

## Table range

Angle tables span `[0, pi)` and are non-periodic. Dihedral tables span
`[0, 2*pi)` and are periodic because the computer looks up `phi + PI`; a ramp
is discontinuous across the periodic wrap, so the dihedral configurations are
chosen to sit near the middle of the table, and the test asserts that.

## Host vs device

The computers' `operator()` is `DEVICE`, which expands to nothing outside a
device compilation unit, and `atomic_add` falls back to plain `+=` on the host
(`Types/Types.h`). Most cases therefore run host-side, which keeps them exact
and backend-independent. The final case re-runs the same inputs through
`launch_tabulated_angles` / `launch_tabulated_dihedrals` and compares against
the host result, so the device launch path is covered too.
