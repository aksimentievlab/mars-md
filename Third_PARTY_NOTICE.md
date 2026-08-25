# Third-Party Dependencies and Licenses

This project incorporates code from several open-source projects. We are grateful to the developers for their contributions.

---
Document style: doxygen-awesome.css
| Submodule | License | Source Repository |
| :--- | :--- | :--- |
| `extern/nanobind` | BSD-3-Clause | [https://github.com/wjakob/nanobind](https://github.com/wjakob/nanobind) |
| `extern/colvars`| LGPL-3.0 license | [https://github.com/Colvars/colvars](https://github.com/Colvars/colvars) |
| `extern/Catch2` | Boost Software License 1.0 | [https://github.com/catchorg/Catch2](https://github.com/catchorg/Catch2) |
| `extern/fmt` | MIT License | [https://github.com/fmtlib/fmt](https://github.com/fmtlib/fmt) |
| `extern/OpenRAND` | MIT | [https://github.com/glotzerlab/OpenRAND](https://github.com/glotzerlab/OpenRAND) |
| `extern/metal-cpp-cmake`| Apache License 2.0 | [https://github.com/LeeTeng2001/metal-cpp-cmake](https://github.com/LeeTeng2001/metal-cpp-cmake) |
| `extern/openvdb/nanovdb` | Apache License 2.0 | [https://github.com/AcademySoftwareFoundation/openvdb](https://github.com/AcademySoftwareFoundation/openvdb) |
| `extern/scuff-em` | GPL-2.0 / GPL-3.0 | [https://github.com/HomerReid/scuff-em](https://github.com/HomerReid/scuff-em) |


> **Note on `extern/scuff-em` (GPL):** scuff-em is copyleft-licensed and is
> **not** part of the default MARS build. It is gated behind the CMake option
> `USE_SCUFF` (default `OFF`) for internal/private use only. Do **not** distribute
> any binary built with `USE_SCUFF=ON`, as linking scuff-em subjects the combined
> work to the GPL. Default MARS builds remain under the project's own license.

Please find the full license text within each submodule's respective directory after running `git submodule update --init`

---
