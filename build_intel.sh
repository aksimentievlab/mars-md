#!/usr/bin/env bash
set -euo pipefail

source /opt/apps/lmod/lmod/init/bash
module load intel/26

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICPX_ROOT="/scratch/projects/compilers/intel26.0/compiler/2026.0"
OCLOC_REAL_DIR="/scratch/projects/compilers/intel26.0/vtune/2026.0/bin64/gma/GTPin/Profilers/ocloc/Bin/intel64"
TOOLS_DIR="${SCRIPT_DIR}/cmake/stampede-sycl-tools"
BUILD_DIR="${SCRIPT_DIR}/build/stampede-icpx-release"

mkdir -p "${TOOLS_DIR}/lib"
ln -sf "${OCLOC_REAL_DIR}/libigc.so"          "${TOOLS_DIR}/lib/libigc.so.1"
ln -sf "${OCLOC_REAL_DIR}/libigdfcl.so"       "${TOOLS_DIR}/lib/libigdfcl.so.1"
ln -sf "${OCLOC_REAL_DIR}/libopencl-clang.so" "${TOOLS_DIR}/lib/libopencl-clang.so.14"

cat > "${TOOLS_DIR}/ocloc" <<EOF
#!/bin/bash
exec env LD_LIBRARY_PATH="${TOOLS_DIR}/lib:${OCLOC_REAL_DIR}:\${LD_LIBRARY_PATH:-}" "${OCLOC_REAL_DIR}/ocloc" "\$@"
EOF
chmod +x "${TOOLS_DIR}/ocloc"

export PATH="${TOOLS_DIR}:${ICPX_ROOT}/bin/compiler:${PATH}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${ICPX_ROOT}/bin/icpx" \
    -DICPX_ROOT="${ICPX_ROOT}" \
    -DUSE_SYCL_ICPX=ON \
    -DSYCL_DEVICE_TYPE=2 \
    -DUSE_PYTHON=OFF\
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_CXX_FLAGS="-O3 -DSG_SIZE=32"

cmake --build "${BUILD_DIR}" -j 20

Unit_test/arbd_zorder_tests