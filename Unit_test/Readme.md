### Unit_test

- Folder for unit tests
- Centralized Test Backend init are in catch_boiler.cpp.
- test_unit was the old unit tests that uses different infrasctucture
- DO NOT CALL lauch_kernel directly in Unit_Test or nvcc would be unhappy. A .cu is needed for pre-instantiation
- Type Safety: for data types passing into kernels, use MARS::types eg. MARS::int2.

## CUDA and SYCL passed all tests in the folder.
