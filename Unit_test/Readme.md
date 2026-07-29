### Unit_test

- Folder for unit tests
- Centralized Test Backend init are in catch_boiler.cpp.
- test_unit was the old unit tests that uses different infrasctucture
- DO NOT CALL lauch_kernel directly in Unit_Test or nvcc would be unhappy. A .cu is needed for pre-instantiation

## CUDA and SYCL passed all tests in the folder.

# Tests only passed in CUDA:

# Tests only passed in SYCL: Types/basegrid_device_test.cpp
