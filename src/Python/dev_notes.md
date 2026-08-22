# src/Python dev notes

## PyTypeCasters.h (pybind11 -> nanobind migration, 2026-08-21)

- Vector3_t/Matrix3_t get no dedicated Python class. They convert
  transparently to numpy arrays ((3,) and (3,3)) on the way out, and accept
  any equivalently-shaped list/tuple/numpy array on the way in. This
  replaced the earlier `py::class_` bindings, which forced every call site
  to wrap plain coordinates in `arbd.Vector3(x, y, z)` /
  `arbd.Matrix3(...)` first - arithmetic, dot/cross, transpose, inverse are
  just numpy's now, not a bespoke reimplementation.
- type_caster specializations are resolved per translation unit, so the
  header must be included (before any `.def(...)`/`nb::init<...>()` call
  that takes or returns one of these types) by every .cpp file that binds
  one.
- nanobind's ndarray return needs `.cast()` to force immediate array
  creation while the stack-local `data` array is still alive (see
  extern/nanobind/docs/ndarray.rst, "Returning temporaries"). The one-arg
  `Array(data)` ctor works because `nb::shape<3>` / `nb::shape<3,3>` are
  fully static, so nanobind infers ndim/shape at compile time - no
  explicit shape array or owning capsule needed since `.cast()` copies
  immediately.
- Matrix3_t stores three column vectors; numpy arrays are read row-major,
  so both directions transpose (matches the old pybind11 caster's
  behavior). A (3,3) array's rows are the matrix's rows, as a reader would
  expect from `np.eye(3)` or a rotation matrix.

## CMakeLists.txt (pybind11 -> nanobind, 2026-08-21)

- Dropped the CMAKE_INTERPROCEDURAL_OPTIMIZATION save/restore-around-CUDA
  workaround: pybind11_add_module enabled LTO by default (breaking CUDA
  device linking, pybind/pybind11#4825); nanobind_add_module only turns on
  IPO/LTO when the `LTO` keyword arg is passed (see `nanobind_lto()` in
  extern/nanobind/cmake/nanobind-config.cmake), which we don't pass. No
  workaround needed.
- Dropped the explicit `find_package(Python ...)` call: extern/nanobind's
  own CMakeLists.txt already does `find_package(Python 3.10 REQUIRED
  COMPONENTS Interpreter Development.Module)` when `Python::Module`/
  `Python::Interpreter` targets don't yet exist. Since the top-level
  CMakeLists.txt does `add_subdirectory(extern/nanobind)` before
  `add_subdirectory(src/Python)`, that target already exists by the time
  this file is processed - a second find_package call here would be
  redundant (and risks a COMPONENTS mismatch complaint).
- extern/nanobind is a git submodule but shows `-` (uninitialized) in
  `git submodule status` even though its files (including the nested
  ext/robin_map -> tsl submodule) are present on disk. Doesn't block the
  CMake build since add_subdirectory only cares about files on disk, but
  worth fixing with `git submodule update --init` if that ever bites.

## pyarbd.cpp / pysystem.cpp / pyobjects.cpp / pybonded.cpp / pyloadfile.cpp
## / pysim.cpp (pybind11 -> nanobind, 2026-08-21)

- Mechanical renames: `py::`->`nb::`, `py::module_`->`nb::module_`,
  `py::enum_`->`nb::enum_` (unchanged shape), `py::init<Args...>()` stays
  `nb::init<Args...>()` for direct constructors, `py::def_readwrite`/
  `def_readonly`/`def_property`/`def_property_readonly` ->
  `def_rw`/`def_ro`/`def_prop_rw`/`def_prop_ro`,
  `py::return_value_policy::X` -> `nb::rv_policy::X`. `<pybind11/stl.h>`'s
  catch-all is gone in nanobind - each container type needs its own header
  (`<nanobind/stl/vector.h>`, `string.h`, `optional.h`, ...).
- `py::init([](Args...) { return T{...}; })` lambda factories became
  `.def("__init__", [](T* self, Args...) { new (self) T{...}; }, ...)`
  placement-new in-place construction - nanobind's `nb::init<>()` only
  wraps real constructors, it has no lambda-factory overload. See
  extern/nanobind/docs/porting.rst "Type casters" section and
  extern/nanobind/tests/test_thread.cpp's `ClassWithClassProperty` binding
  for the canonical pattern.
- pyloadfile.cpp's `py::array_t<float, c_style|forcecast>` +
  `.request()`/`py::buffer_info` became
  `nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu>` with
  direct `.shape(i)`/`.data()`/`.size()` accessors - no buffer_info
  intermediary needed. nanobind's implicit-conversion dispatch pass
  (extern/nanobind/docs/ndarray.rst "Overload resolution") already covers
  what `forcecast` did in pybind11 (copy-convert to the requested
  dtype/contiguity on a second dispatch attempt), so no explicit forcecast
  flag exists or is needed.
- `grid_to_numpy`'s returned array is heap-allocated with an
  `nb::capsule` deleter (same pattern as
  extern/nanobind/tests/test_ndarray.cpp's `return_no_framework`), not the
  `.cast()`-on-stack-array trick used in PyTypeCasters.h - that trick only
  applies to fixed small stack buffers where `.cast()` forces an immediate
  copy before the stack storage goes out of scope; grid data is
  heap-sized and needs real ownership transfer instead.
- `py::index_error`/`py::value_error` -> `nb::index_error`/
  `nb::value_error` (same call shape, `throw nb::index_error("msg")`).
- `py::gil_scoped_release`/`py::call_guard<...>` -> same names under `nb::`.
- ConfigParser.h/.cpp's `pybind11::object`/`pybind11::cast<T>`/
  `pybind11::cast_error` -> `nanobind::` equivalents. This code
  (`ConfigParser(SimSystem&, const std::map<std::string, object>&)`,
  `parse_dictionary`) is gated by `#ifdef USE_PYTHON`.
- 2026-08-21 follow-up: renamed the CMake option `USE_PYBIND` ->
  `USE_PYTHON` everywhere (top-level CMakeLists.txt, Tests/CMakeLists.txt,
  Tests/interactive/CMakeLists.txt, Tests/interactive/particle_tests.cpp's
  `#ifdef` guard) so the build option name matches the source-level macro
  name ConfigParser.h/.cpp already checked. This does **not** revive the
  dict-constructor path above: `lib${PROJECT_NAME}` (which compiles
  ConfigParser.cpp) never receives `USE_PYTHON` as a
  `target_compile_definitions` anywhere - only the two Python *module*
  targets (`py${PROJECT_NAME}` in src/Python/CMakeLists.txt via
  `nanobind_add_module`, and `arbd_interactive` in
  Tests/interactive/CMakeLists.txt) get it, and neither compiles
  ConfigParser.cpp itself (they link the pre-built lib). Per
  pythontodo.md's explicit decision ("Leave USE_PYTHON dead code in place,
  never define it... Reviving it would create a third config path" -
  arbdpy.bd.apply.BdApplier is meant to supersede it), that dead-code
  status is preserved; only the naming confusion is resolved.
- Note: `nb::cast_error` is `using cast_error = std::bad_cast;` (a bare
  alias, not a nanobind-specific exception class) - `e.what()` in
  ConfigParser.cpp's `catch (const nanobind::cast_error& e)` now returns
  `std::bad_cast`'s generic message instead of pybind11's
  descriptive "Unable to cast Python instance of type X to C++ type Y"
  text. Same catch-and-rethrow-as-Exception structure otherwise.
- 2026-08-21: Tests/interactive/particle_tests.cpp converted too (was
  flagged as out-of-scope, then requested). It's a real `MODULE` target
  (`nanobind_add_module`/formerly `pybind11_add_module`), not an embedded
  interpreter, so this was the same mechanical `py::`->`nb::`,
  `def_readwrite`->`def_rw`, `PYBIND11_MODULE`->`NB_MODULE` swap as the six
  pyarbd files - no `nb::init<>()` lambda-factory rewrites needed here
  since this file only binds plain member fields. Its `Vector3_t<float>`
  binding is its own standalone `nb::class_` (x/y/z fields), separate from
  and NOT using PyTypeCasters.h's numpy-array caster - correct to leave
  as-is, since a type can't have both a custom type_caster and a
  `nb::class_` registration.
  Pre-existing bugs in this file's `#else` (Catch2) branch - calling
  `.get_particle_count()` on the `std::tuple` that
  `create_interactive_test_system` returns, `SimSystem::get_num_particles`
  not existing, undeclared `system_state` - are untouched; they predate
  this migration and Tests/ isn't wired into the build via
  `add_subdirectory` from the top-level CMakeLists.txt at all (confirmed:
  no `add_subdirectory(Tests)` anywhere, not in CMakeLists.txt or
  CMakePresets.json), so none of this - including the now-renamed
  `USE_PYTHON` plumbing here - is on any currently-configured build path.
