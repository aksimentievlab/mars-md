# Grid.h — implementation notes

## Path resolution: resolve for reading, register verbatim

`add_dense_grid(filename, config_file_path)` resolves `filename` through
`resolve_file_path` (`Header.h`) for the `DXReader::read_from_file` call **only**.
The original, unresolved string is what goes into `fname_to_gridkey_` and
`grid_keys_`.

That split is deliberate. `SimSystem::assign_particle_type_ids()` looks grids up by
the name as written in the config (`get_grid_key(pmf_grid_names[g])`,
`SimSystem.h:584-597`). Registering the resolved absolute path would break every one
of those lookups, since the caller only ever has the config string.

Before this, all eight `ConfigParser.cpp` call sites passed the raw string straight
to `DXReader`, so `gridFile potentials/null.dx` resolved against the **process CWD**
rather than the config file's directory — unlike every other file key in the parser,
which already went through `resolve_file_path`. A fixture therefore only ran from
its own directory.

The one-argument overload is kept and delegates with `""` (process CWD), for callers
that have no config file — the Python path and tests.

**Caveat:** the dedup check keys on the unresolved string, so the same grid reached
by two different spellings (`grids/a.dx` from one config, `../x/grids/a.dx` from
another) loads twice. That was already true and is not worth fixing until grid
memory becomes a problem.
