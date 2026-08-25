### Unit_test

- Folder for unit tests
- Centralized Test Backend init are in catch_boiler.cpp.
- test_unit was the old unit tests that uses different infrasctucture
- DO NOT CALL lauch_kernel directly in Unit_Test or nvcc would be unhappy. A .cu is needed for pre-instantiation
- Type Safety: for data types passing into kernels, use MARS::types eg. MARS::int2.

## CUDA and SYCL passed all tests in the folder.

# Tests only passed in CUDA:

# Tests only passed in SYCL: Types/basegrid_device_test.cpp

###
Please change single_resource_id to whichever gpu is not busy.
```
namespace Global {
static short single_resource_id = 6; // For testing, we use a single resource id
static std::vector<short> device_ids = {0, 1};
}
```
