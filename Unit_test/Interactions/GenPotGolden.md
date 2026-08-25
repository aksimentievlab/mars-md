# GenPotGolden — notes

Validates `convolve_grids` against the original `gen_pot` tool on a real production grid.

## Running it

Tagged `[.golden]`, so it is hidden from the default run and needs the reference data:

```
MARS_GENPOT_REF=/data/server5/hchou10/2019NIH_proposal/cytoplasm/steric/1A/output \
  ./Unit_test/mars_unit_tests "[.golden]"
```

Optional: `MARS_GENPOT_STEM` (default `1ema.protein`), `MARS_GENPOT_C6`, `MARS_GENPOT_C12`.

The reference pair came from `make_rigid_potential.sh`:

```
gen_pot output/<p>.Density.dx lenard_jones_repulsion input_CC output/<p>.pot.dx
```

with `input_CC` = C6 1228.8, C12 2516582.4 → σ = 3.5636 Å, ε = 0.15, WCA cutoff exactly 4.0 Å.
`lenard_jones_repulsion` was chosen because it has compact support; `gen_pot` convolves over
the whole box by FFT, so only a compact kernel is reproducible by a truncated real-space
stencil.

## The half-voxel convention — read before changing the kernel construction

The grid is 58×66×76, origin (−28.5, −32.5, −37.5), δ = 1. So **world-0 falls at index 28.5,
half a voxel off a lattice point**, and `gen_pot` samples its kernel at half-integer offsets
(±0.5, ±1.5, …). It never evaluates r = 0.

Two consequences:

1. **Sampling on integers changes sum(K) by 8.5%.** WCA is steep near the core and saturates
   at `gen_pot`'s cap of 100, so shifting the sample points by half a cell is not a small
   perturbation. An odd-width kernel here produced `sum(ours)/sum(gen_pot) = 1.085460`,
   against a predicted kernel-sum ratio of `1.085458` — six-figure agreement, i.e. the
   discrepancy was entirely the sampling offset and not the convolution. The test therefore
   builds an **even-width** kernel, whose centers `(m − (n−1)/2)·δ` are half-integers.

2. **`gen_pot`'s output carries a −0.5 voxel displacement.** Its k-space checkerboard
   (`inner_product_grid`) shifts the result by exactly N/2 = 29, while the kernel is centered
   at 28.5. The two cancel only when origin is an integer multiple of δ, which it is not here.
   This is a property of that dataset meeting `gen_pot`'s convention, not something to
   reproduce — so the test asserts on **total mass**, which is invariant under displacement
   and isolates normalization and kernel amplitude from centering.

## Normalization

`gen_pot`'s `inner_product_grid` divides by `V = nx·ny·nz`, which exactly cancels FFTW's
unnormalized round-trip scaling. The result is the plain circular convolution `Σ ρ(y)K(x−y)`
with no cell-volume factor — this is `ConvolutionNormalization::VoxelSum`. `Volume` multiplies
by the density's cell volume and is the physically-dimensioned alternative.

Note `arbdmodel`'s `convolve_kernel_truncate` uses a *third* convention: it divides by a
`count` array and rescales by `prod(kernelShape)` as an edge correction, which a periodic
implementation does not need.
