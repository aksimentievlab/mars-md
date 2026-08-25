# Mesoscale to Atomistic Resolution Software for Molecular Dynamics (MARS-MD / arbd2 alpha, a successor of arbd)

This development branch of MARS focuses on scaling simulations to larger systems and accelerating performance to hardware limits, while keeping the codebase maintainable and its features diverse. In particular we are targeting speed and good scaling on multi-GPU clusters. Our primary objectives are:

- **Portability**: Cross-vendor support.
- **Scalability**: Handle larger molecular systems efficiently
- **Performance**: Achieve optimal speed and scaling on multi-GPU clusters
- **Maintainability**: Clean, modular codebase for easier development

> **Development Status**: This is an alpha version under active development. Expect breaking changes, incomplete features, and rough edges — use at your own risk.

## Tested Systems

### Linux (CUDA)
- **Operating System**: Linux workstation with a CUDA-compatible GPU (also builds against the SYCL backend on the same hardware)
- **GPU**: NVIDIA A100 (`sm_80`), A40 (`sm_86`), RTX PRO 6000 Blackwell (`sm_120`)
- **Build Tools**:
  - CMake 4.0
  - GCC 14 or Intel oneAPI 25.1
  - CUDA 12.8 for A100 / A40; CUDA 13 required for Blackwell

### Linux (SYCL-intel)
- **Operating System**: TACC Stampede3 Supercomputer
- **GPU**:  Intel Data Center GPU Max 1550s
- **Build Tools**:
  - CMake 4.0
  - GCC 14
  - Intel oneAPI 26.1

## Building

### Prerequisites

Ensure you have the spdlog submodule initialized:
```bash
git submodule update --init
```
### Build Unit Tests

Unit Test devices can be set as
```bash
cmake --preset tbgl-cuda-debug -DUNIT_TEST_DEVICE_ARRAY="0;1;2" -DUNIT_TEST_DEVICE_ID=0
```
Omit `-DUNIT_TEST_DEVICE_ID` to use the first id in `UNIT_TEST_DEVICE_ARRAY`.

### Linux with CUDA

Configure and build with a CUDA preset:
```bash
cmake --preset tbgl-cuda-release   # or tbgl-cuda-debug
cmake --build build/tbgl-cuda-release -j$(nproc)
```

Override the target GPU architectures if needed:
```bash
export CMAKE_CUDA_ARCHITECTURES="80;90"
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

MARS is developed by the [Aksimentiev Group](http://bionano.physics.illinois.edu) at the University of Illinois at Urbana-Champaign.

**Core Development Team:**
- **Pin-Yi Li** - Lead Developer ([pinyili2@illinois.edu](mailto:pinyili2@illinois.edu))
- **Christopher Maffeo** - Developer ([cmaffeo2@illinois.edu](mailto:cmaffeo2@illinois.edu))

**Past Contributors:**
- **Jeffrey Comer**
- **Max Belkin**
- **Terrance Howard**
- **Han-yi Chou**
- **Emmanual Guzman**
- **Justin Dufresne**

## Support

For questions, problems, or suggestions, please contact Pin-Yi Li [pinyili2@illinois.edu](mailto:pinyili2@illinois.edu) or Chris Maffeo at [cmaffeo2@illinois.edu](mailto:cmaffeo2@illinois.edu).

## License

This project is licensed under the UIUC License - see the [LICENSE](LICENSE) file for details.
