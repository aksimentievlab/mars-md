# Atomic Resolution Brownian Dynamics (ARBD 2.0-alpha)

Brownian dynamics (BD) simulation is method for studying biomolecules,
ions, and nanomaterials that balances detail with computational
efficiency.

This development branch of ARBD has the aim of scaling ARBD up to
larger systems and accelerating to the hardware limits, while making
it easier to maintain diverse features. In particular we are seeking
speed and good scaling on multi-GPU clusters.


This development branch of ARBD focuses on scaling simulations to larger systems and accelerating performance to hardware limits, while maintaining code maintainability and diverse feature support. Our primary objectives include:

- **Performance**: Achieve optimal speed and scaling on multi-GPU clusters
- **Scalability**: Handle larger molecular systems efficiently
- **Maintainability**: Clean, modular codebase for easier development

> ⚠️ **Development Status**: This is an alpha version under active development. Many features are not yet functional.

## Requirements

### Linux (CUDA)
- **Operating System**: Linux workstation with CUDA-compatible GPU
- **Build Tools**: 
  - CMake ≥ 3.25
  - GCC ≥ 4.9 or Clang
- **CUDA Toolkit**: CUDA ≥ 12.0
- **Compute Capability**: NVIDIA GPU with compute capability ≥ 2.0 (developed and tested on 6.0+)

### Other Systems
- **Operating System**: macOS with Apple Silicon (M1/M2/M3)
- **Build Tools**: CMake, Homebrew
- **Parallel Computing**: OpenMP and OpenCL support
- **SYCL**: AdaptiveCpp (ACPP) (recommanded for Mac) or Intel DPC++ 

## Building

### Prerequisites

Ensure you have the spdlog submodule initialized:
```bash
git submodule update --init
```

### Linux with CUDA

1. **Set CUDA Architecture** (especially important for CMake < 3.24.0):
   ```bash
   export CMAKE_CUDA_ARCHITECTURES="35;50;75;80"
   ```

2. **Configure and Build**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

#### Troubleshooting CUDA Build

If CMake cannot find your CUDA installation:
- Set the CUDA compiler path: `export CMAKE_CUDA_COMPILER=/path/to/nvcc`
- Specify CUDA include directory: `export CUDA_INCLUDE_DIRS="$CUDA_HOME/include"`

### macOS-arm64 with SYCL

```bash
mkdir build && cd build
/opt/homebrew/bin/cmake \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DUSE_SYCL_ACPP=ON \
  -DCMAKE_CXX_COMPILER=acpp \
  ..
make -j$(sysctl -n hw.ncpu)
```

> **Note**: Use `-DCMAKE_CXX_COMPILER=icpx` for Intel DPC++ instead of AdaptiveCpp.

### API Reference
```bash
doxygen Doxyfile
```

## Usage

Documentation and usage examples will be provided as development progresses.

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## Authors

ARBD2 is developed by the [Aksimentiev Group](http://bionano.physics.illinois.edu) at the University of Illinois at Urbana-Champaign.

**Core Development Team:**
- **Christopher Maffeo** - Lead Developer ([cmaffeo2@illinois.edu](mailto:cmaffeo2@illinois.edu))
- **Pin-Yi Li** - Developer ([pinyili2@illinois.edu](mailto:pinyili2@illinois.edu))
- **Han-yi Chou** - Developer

## Support

For questions, problems, or suggestions, please contact Chris Maffeo at [cmaffeo2@illinois.edu](mailto:cmaffeo2@illinois.edu).

## License

This project is licensed under the UIUC License - see the [LICENSE](LICENSE) file for details.
