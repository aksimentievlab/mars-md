# Tabulated pairwise nonbonded energy tests

Covers `launch_pairwise_nonbonded` / `TabulatedNonBondedComputer` from
`Interactions/Nonbonded/Pairwise.h`.

These live in their own file rather than in `Nonbonded.cpp` (which covers the
*analytical* softcore/Coulomb kernels) because `Nonbonded.cpp` is not listed in
`Unit_test/CMakeLists.txt` and has never been compiled. It has drifted out of
sync with the kernels it names — `Resource::synchronize()` no longer exists,
`SoftcoreForceKernel`/`ColumbForceKernel` take different arguments than it
passes, and several assertions use `REQUIRE(x == Catch::Matchers::WithinAbs(...))`
where `REQUIRE_THAT` is required. Adding to it would have produced tests that
silently never run.

## Why

The rotor fixture (`~/server3/rotor/rotor_center_debye_30ms2`) put arbd2's
total potential energy ~13,400 kcal/mol above v1's, and the decomposition put
essentially all of it in the nonbonded term (8,000 of it, versus 486 for bonds
and 6,360 for angles + dihedrals). v1's own number turned out not to be a
usable reference — see the entry in `todo.md` — so the nonbonded energy needed
validating against something other than v1. Before this, nothing exercised
`launch_pairwise_nonbonded` at all: `Compute/BDKernels.cpp` loads 190 pairwise
tables into a `TablesRegistry` but only ever launches the *bond* kernel.

## Reference

`PairTable` samples an arbitrary `U(r)` onto the same grid the kernel reads,
and `PairTable::interpolated` reproduces `TabulatedPotential::compute`'s linear
interpolation on the host. The reference total is then a direct double loop
over pairs, which is the quantity the rotor comparison actually needed.

`U(r) = (r-3)^2 - 1` is used throughout: it has a real negative well (minimum
-1 at r=3) and a positive repulsive wall. That matters because the open
question on the rotor was a *sign* — whether arbd2 can report negative
nonbonded energy at all. The second test pins exactly that.

## Accounting

`TabulatedNonBondedComputer` adds `U/2` to each endpoint of a pair, so the sum
over particles is `sum U_ij` with no double counting. The first test asserts
both the total and the per-particle halves, so a future change to either
convention fails loudly rather than silently doubling the reported energy.

## Coverage

- one pair: total is `U(r)`, each endpoint carries `U(r)/2`
- attractive well: energy is negative at `r = 3`, positive on the wall
- 16 jittered lattice sites, all 120 pairs, against a direct CPU pair sum
- exclusions: an excluded pair drops exactly its own `U` from the total
- cutoff: a pair beyond `cutoff` drops exactly its own `U`

Exclusions are supplied as CSR (`excl_offsets` / `excl_neighbors`) with each
exclusion stored on both endpoints, matching what
`DeviceBondedInteractions::exclusion_offsets()` builds; the kernel only scans
`indices.x`'s list.
