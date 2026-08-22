if [[ -z "${GPUS:-}" ]]; then
	command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required when GPUS is not set" >&2; exit 1; }
	GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | awk -F', *' '$2 < 5 && $3 < 40 { printf "%s ", $1 }')
	[[ -n "$GPUS" ]] || { echo "No GPU has utilization < 5% and temperature < 40 C" >&2; exit 1; }
fi
module load gcc/14
module load cuda-toolkit/12.8
module load cmake/4
export CMAKE_CUDA_ARCHITECTURES="86;120"
cmake --preset tbgl-cuda-release -DUNIT_TEST_DEVICE_ARRAY="3 4 5 6"
cd build/tbgl-cuda-release
ninja -j 20
Unit_test/arbd_unit_tests
Unit_test/arbd_zorder_tests
