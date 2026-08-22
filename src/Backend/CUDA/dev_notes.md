# Backend/CUDA dev notes

## CUDA 13 removed the device-ordinal managed-memory hints (2026-08-15)

`tbgl-cuda-debug` built fine under CUDA 12.8 and failed with ~95 errors under
CUDA 13.0. All of them came from four lines in `CUDABuffer.h` (`UnifiedPolicy`),
amplified across ~15 translation units because it is a header.

CUDA 12.x carried two generations of the managed-memory hint API:

| | 12.2 – 12.x | 13.0 |
|---|---|---|
| ordinal form | `cudaMemPrefetchAsync(p, n, int dev, stream)` | **removed** |
| | `cudaMemAdvise(p, n, advice, int dev)` | **removed** |
| location form | `cudaMemPrefetchAsync_v2(p, n, cudaMemLocation, unsigned flags, stream)` | same, renamed to `cudaMemPrefetchAsync` |
| | `cudaMemAdvise_v2(p, n, advice, cudaMemLocation)` | same, renamed to `cudaMemAdvise` |

CUDA 13 deleted the ordinal overloads and promoted the `_v2` signatures to the
plain names. So the old `int` argument landed in the `cudaMemLocation` slot
(`could not convert 'device_id' from 'int' to 'cudaMemLocation'`), and for
prefetch the trailing `cudaStream_t` then slid into the new `unsigned int flags`
slot (`argument of type "cudaStream_t" is incompatible with parameter of type
"unsigned int"`). Two error messages, one root cause.

### Why the `#if` cannot be removed entirely

12.8 *does* have `struct cudaMemLocation` and `cudaMemLocationType{Device,Host}`
(`driver_types.h:2130`), so the argument types are portable across both
toolkits — only the **function name** differs (`_v2` vs unsuffixed). The
version branch is therefore confined to two inline wrappers,
`CUDA::prefetch_async` and `CUDA::advise`; every call site uses the modern
location form unconditionally.

### Consequences

- Floor is now **CUDA 12.2** (first release with the `_v2` entry points).
  Building against 12.0/12.1 fails on an undeclared `cudaMemPrefetchAsync_v2`.
  Nothing in the presets targets those.
- The new `flags` parameter is reserved; pass `0`.
- `cudaCpuDeviceId` still exists (`(int)-1`) but is only meaningful as a
  `cudaMemLocation::id`, never as a function argument. `mem_location()` maps it
  to `cudaMemLocationTypeHost`.
- For `cudaMemAdviseSetReadMostly` the driver ignores the location, so passing a
  device location there is harmless.
- `prefetch_async` takes `const void*`, which dropped the `const_cast` that
  `copy_to_host` previously needed.
