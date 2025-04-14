# Atomic Resolution Brownian Dynamics (ARBD 2.0-alpha)

Brownian dynamics (BD) simulation is method for studying biomolecules,
ions, and nanomaterials that balances detail with computational
efficiency.

This development branch of ARBD has the aim of scaling ARBD up to
larger systems and accelerating to the hardware limits, while making
it easier to maintain diverse features. In particular we are seeking
speed and good scaling on multi-GPU clusters.

It is currently non-functional as many developments remain.

## Building

### Dependencies

Linux workstation with CUDA-compatible GPU (minimum 3.5 compute capability)
  - CMake >= 3.9
  - gcc >= 4.9
  - cuda >= 9.0  (> 11.5 recommended)
  - spdlog >= 1.10.0 (note: this is normally installed to extern/spdlog by running `git submodule update --init` from this directory)

### Build process

From the root arbd directory (where this README is found), ensure you have spdlog installed to the extern directory, usually by running `git submodule update --init`.

From the root arbd directory (where this README is found), run:
```
## Determine the compute capability of your CUDA-enabled graphics card
export CMAKE_CUDA_ARCHITECTURES="35;50;75;80"   ;# especially important for CMake < 3.24.0
## export CUDA_INCLUDE_DIRS="$CUDA_HOME/include" ;# optionally be explicit about cuda include paths; usually not needed
cmake -S src -B build &&
(
  cd build
  make -j
)
```

If your CUDA toolkit is installed in a nonstandard location that CMake
is unable to find, you may provide use the environement variable
`CMAKE_CUDA_COMPILER` to specify the path to nvcc. You may also find
it neccesary to set the environment variable `CUDA_INCLUDE_DIRS` if
compilation fails due to the compiler being unable to find <cuda.h>.

Note that ARBD has been developed using CUDA-9.0 and targets NVIDIA
GPUs featuring 6.0 compute capability. The code should work with
devices with compute capability >=2.0, but there are no guarantees.

## Authors

ARBD2 is being developed by the Aksimentiev group
(http://bionano.physics.illinois.edu).

  - Christopher Maffeo <mailto:cmaffeo2@illinois.edu>
  - Han-yi Chou
  - Pin-Yi Li <mailto:pinyili2@illinois.edu>

Please direct questions, problems or suggestions to Chris.
