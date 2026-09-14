"""The ``.bd`` parser.

Reads a config file into a :class:`~marsmd.bd.model.BdConfig`. Opens nothing
else: auxiliary filenames are recorded as written and loaded later by
:mod:`marsmd.bd.aux`, driven by the applier.

See dev_notes.md for the block-cursor semantics and why they are a superset of
``ConfigParser::get_elements``.
"""

from __future__ import annotations

from .errors import BdParseError
from .keywords import (
    BLOCK_HEADERS,
    DECOMPOSER_MAP,
    GLOBAL_KEYS,
    INTEGRATOR_MAP,
    KNOWN_UNSUPPORTED_KEYS,
    LONG_RANGE_MAP,
    OUTPUT_FORMAT_MAP,
    PARTICLE_FIELD_KEYS,
    RIGID_BODY_FIELD_KEYS,
    RIGID_BODY_PMF_KEYS,
    TOPOLOGY_KEYS,
)
from .model import (
    BdConfig,
    GridEntry,
    GridScale,
    ParticleBlock,
    RigidBodyBlock,
    TabulatedPair,
    TopologyFile,
    UnsupportedKey,
)
from .tokens import (
    SourceLine,
    iter_parameter_lines,
    parse_int,
    parse_matrix3_rows,
    parse_vector3,
    parse_vector3_strict,
    tokenize,
)

__all__ = ["BdParser", "parse_file", "parse_string"]

_BOUNDARY_CONDITIONS = {"dirichlet": 0, "neumann": 1, "periodic": 2}


def _parse_boundary_condition(token: str) -> int:
    """One ``gridFileBoundaryConditions`` token; <0 keeps the grid's own."""
    return _BOUNDARY_CONDITIONS.get(token.lower(), -1)


def _parse_grid_entry(value: str) -> tuple[str, str] | None:
    """Parse a rigidBody grid line: ``<gridKey> <file>`` or a bare ``<file>``.

    A bare file doubles as its own key.
    """
    toks = tokenize(value)
    if len(toks) == 1:
        return toks[0], toks[0]
    if len(toks) == 2:
        return toks[0], toks[1]
    return None


def _parse_grid_scale(value: str) -> GridScale | None:
    """Parse ``<gridKey> <value>`` or a bare ``<value>``."""
    toks = tokenize(value)
    try:
        if len(toks) == 1:
            return GridScale("", float(toks[0]))
        if len(toks) == 2:
            return GridScale(toks[0], float(toks[1]))
    except ValueError:
        return None
    return None


def _apply_pmf_list(entries: list[GridEntry], toks: list[str], setter) -> list[GridEntry]:
    """Apply a deferred particle-block list to that block's grid entries.

    A single value applies to every entry; a list must line up one-to-one.
    Any other count is ignored, matching ``apply_pmf_grid_list``
    (``ConfigParser.cpp:360-381``).
    """
    if not toks or not entries:
        return entries
    if len(toks) != 1 and len(toks) != len(entries):
        return entries
    return [
        setter(entry, toks[0] if len(toks) == 1 else toks[j])
        for j, entry in enumerate(entries)
    ]


def _apply_rb_scales(
    entries: list[GridEntry], scales: list[GridScale]
) -> list[GridEntry]:
    """Apply deferred rigidBody scale lines, keyed or positional.

    A keyed line scales the one grid registered under that key; a bare value
    scales every grid of that kind. Unmatched keys are ignored.
    """
    if not entries:
        return entries
    out = list(entries)
    for scale in scales:
        if not scale.grid_key:
            out = [
                GridEntry(e.key, e.filename, scale.value, e.scale_slope, e.boundary_condition)
                for e in out
            ]
            continue
        for j, entry in enumerate(out):
            if entry.key == scale.grid_key:
                out[j] = GridEntry(
                    entry.key,
                    entry.filename,
                    scale.value,
                    entry.scale_slope,
                    entry.boundary_condition,
                )
                break
    return out


class BdParser:
    """Turn ``.bd`` text into a :class:`BdConfig`.

    A key the engine does not recognize is collected in ``config.unsupported``
    with a reason, never raised: the engine ignores such keys, and so does this.

    :example:
        >>> config = BdParser().parse_file("run.bd")
        >>> config.globals.temperature
        295.0
        >>> [p.name for p in config.particles]
        ['B', 'P', 'rb']
    """

    # -- entry points ------------------------------------------------------

    def parse_file(self, path: str) -> BdConfig:
        """Parse the ``.bd`` file at ``path``. No other file is opened."""
        with open(path, encoding="utf-8", errors="replace") as fh:
            return self.parse_string(fh.read(), source_path=path)

    def parse_string(self, text: str, *, source_path: str = "") -> BdConfig:
        """Parse ``.bd`` text. ``source_path`` only records where it came from."""
        config = BdConfig(source_path=source_path)
        lines = list(iter_parameter_lines(text))

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.key == "particle":
                i = self._parse_particle_block(lines, i, config)
            elif line.key == "rigidBody":
                i = self._parse_rigid_body_block(lines, i, config)
            else:
                self._parse_top_level(line, config)
                i += 1

        return config

    # -- diagnostics -------------------------------------------------------

    def _record_unsupported(self, line: SourceLine, config: BdConfig, reason: str) -> None:
        config.unsupported.append(
            UnsupportedKey(
                key=line.key, value=line.value, line_no=line.line_no, reason=reason
            )
        )

    def _misplaced_block_key(
        self, line: SourceLine, config: BdConfig, *, in_block: str
    ) -> bool:
        """Record a key that belongs to the *other* block kind.

        ``get_elements`` breaks a block at the first key its own table does not
        claim, then drops it at top level. Recorded here rather than silently
        ignored, and never a strict-mode failure.
        """
        other = (
            RIGID_BODY_FIELD_KEYS if in_block == "particle" else PARTICLE_FIELD_KEYS
        )
        mine = PARTICLE_FIELD_KEYS if in_block == "particle" else RIGID_BODY_FIELD_KEYS
        if line.key in other and line.key not in mine:
            self._record_unsupported(
                line,
                config,
                f"{line.key} is a {'rigidBody' if in_block == 'particle' else 'particle'}"
                f" key; the engine drops it inside a {in_block} block",
            )
            return True
        return False

    def _handle_unknown(self, line: SourceLine, config: BdConfig) -> None:
        """Record a key no table claims. The engine ignores it; so do we."""
        reason = KNOWN_UNSUPPORTED_KEYS.get(line.key, "unrecognized; the engine ignores it")
        self._record_unsupported(line, config, reason)

    # -- top-level ---------------------------------------------------------

    def _parse_top_level(self, line: SourceLine, config: BdConfig) -> None:
        key, value = line.key, line.value

        if key in GLOBAL_KEYS:
            self._apply_global(line, config)
            return

        if key in TOPOLOGY_KEYS:
            config.topology_files.append(TopologyFile(kind=key, filename=value))
            return

        if key in ("tabulatedFile", "tabulated_file"):
            pair = self._parse_tabulated_file(line, config)
            if pair is not None:
                config.tabulated_pairs.append(pair)
            return

        if key in ("inputParticles", "input_particles"):
            config.input_particles = value
            return

        if key in ("restartCoordinates", "restart_coordinates"):
            config.restart_coordinates = value
            return

        if key in ("inputRBCoordinates", "input_rb_coordinates"):
            config.input_rb_coordinates = value
            return

        if key in ("rigidBodyGridGridPeriod", "rigid_body_grid_grid_period"):
            config.globals.rb_grid_grid_period = parse_int(
                value, key=key, line=line, path=config.source_path
            )
            return

        self._handle_unknown(line, config)

    def _parse_tabulated_file(
        self, line: SourceLine, config: BdConfig
    ) -> TabulatedPair | None:
        """Parse ``i@j@path``. A malformed entry is recorded and skipped, as the engine does."""
        parts = line.value.split("@", 2)
        try:
            if len(parts) != 3:
                raise ValueError(line.value)
            return TabulatedPair(int(parts[0]), int(parts[1]), parts[2])
        except ValueError:
            self._record_unsupported(line, config, "malformed tabulatedFile; expected i@j@path")
            return None

    def _apply_global(self, line: SourceLine, config: BdConfig) -> None:
        key, value = line.key, line.value
        g = config.globals
        path = config.source_path

        def as_float() -> float:
            return float(tokenize(value)[0])

        def as_int() -> int:
            return parse_int(value, key=key, line=line, path=path)

        def as_vec3():
            return parse_vector3(value, key=key, line=line, path=path)

        def as_enum(table: dict[str, str]) -> str | None:
            return table.get(tokenize(value)[0].lower())

        try:
            if key in ("temperature_grid", "temperatureGrid"):
                g.temperature_grid = value
            elif key == "temperature":
                g.temperature = as_float()
            elif key == "seed":
                g.seed = as_int()
            elif key == "cutoff":
                g.cutoff = as_float()
            elif key in ("pairlistDistance", "pairlist_distance"):
                g.pairlist_distance = as_float()
            elif key == "timestep":
                g.timestep = as_float()
            elif key == "steps":
                g.steps = as_int()
            elif key in ("decompPeriod", "pairlist_rebuild_period"):
                g.pairlist_rebuild_period = as_float()
            elif key in ("reorderPeriod", "reorder_period"):
                g.reorder_period = as_int()
            elif key in ("outputPeriod", "output_period"):
                g.output_period = as_float()
            elif key in ("outputEnergyPeriod", "output_energy_period"):
                g.output_energy_period = as_float()
            elif key in ("outputName", "output_name"):
                g.output_name = value
            elif key in ("systemSize", "system_size"):
                g.system_size = as_vec3()
            elif key == "origin":
                g.origin = as_vec3()
            elif key in ("outputFormat", "output_format"):
                g.output_format = as_enum(OUTPUT_FORMAT_MAP)
            elif key == "decomposer":
                g.decomposer = as_enum(DECOMPOSER_MAP)
            elif key in ("longRangeMethod", "long_range_method"):
                g.long_range_method = as_enum(LONG_RANGE_MAP)
            elif key in (
                "ParticleDynamicType",
                "particleDynamicType",
                "particle_dynamic_type",
                "algorithm",
            ):
                g.particle_dynamic_type = as_enum(INTEGRATOR_MAP)
            elif key in (
                "RigidBodyDynamicType",
                "rigidBodyDynamicType",
                "rigid_body_dynamic_type",
                "rigidBodyAlgorithm",
                "rigid_body_algorithm",
            ):
                g.rigid_body_dynamic_type = as_enum(INTEGRATOR_MAP)
        except (ValueError, IndexError) as exc:
            raise BdParseError(
                f"{key}: could not parse {value!r}", path, line.line_no, line.raw
            ) from exc

    # -- particle block ----------------------------------------------------

    def _parse_particle_block(
        self, lines: list[SourceLine], start: int, config: BdConfig
    ) -> int:
        header = lines[start]
        block = ParticleBlock(name=header.value, line_no=header.line_no)

        pending_scale: list[str] = []
        pending_slope: list[str] = []
        pending_bc: list[str] = []

        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.key in BLOCK_HEADERS:
                break
            if line.key not in PARTICLE_FIELD_KEYS:
                # Not a particle field: hand it back to the top level. The
                # block cursor stays open (legacy semantics) -- see dev_notes.
                if not self._misplaced_block_key(line, config, in_block="particle"):
                    self._parse_top_level(line, config)
                i += 1
                continue

            key, value = line.key, line.value
            try:
                if key == "num":
                    block.num = parse_int(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "mass":
                    block.mass = float(tokenize(value)[0])
                elif key == "diffusion":
                    block.diffusion = parse_vector3(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "transDamping":
                    block.trans_damping = parse_vector3(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "gridFile":
                    for fname in tokenize(value):
                        block.grid_files.append(GridEntry(key=fname, filename=fname))
                elif key == "diffusionGridFile":
                    block.diffusion_grid_file = value
                elif key == "forceGridFiles":
                    toks = tokenize(value)
                    block.force_grid_files = toks if len(toks) == 3 else []
                elif key == "forceGridScale":
                    block.force_grid_scale = parse_vector3(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "gridFileScale":
                    pending_scale = tokenize(value)
                elif key == "gridFileScaleSlope":
                    pending_slope = tokenize(value)
                elif key == "gridFileBoundaryConditions":
                    pending_bc = tokenize(value)
                elif key == "gridFileSMD":
                    block.grid_file_smd = parse_int(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "rigidBodyPotential":
                    block.rigid_body_potential_keys.append(value)
            except (ValueError, IndexError) as exc:
                raise BdParseError(
                    f"particle {block.name}: {key}: could not parse {value!r}",
                    config.source_path,
                    line.line_no,
                    line.raw,
                ) from exc
            i += 1

        block.grid_files = _apply_pmf_list(
            block.grid_files,
            pending_scale,
            lambda e, s: GridEntry(e.key, e.filename, float(s), e.scale_slope, e.boundary_condition),
        )
        block.grid_files = _apply_pmf_list(
            block.grid_files,
            pending_slope,
            lambda e, s: GridEntry(e.key, e.filename, e.scale, float(s), e.boundary_condition),
        )
        block.grid_files = _apply_pmf_list(
            block.grid_files,
            pending_bc,
            lambda e, s: GridEntry(
                e.key, e.filename, e.scale, e.scale_slope, _parse_boundary_condition(s)
            ),
        )

        config.particles.append(block)
        return i

    # -- rigidBody block ---------------------------------------------------

    def _parse_rigid_body_block(
        self, lines: list[SourceLine], start: int, config: BdConfig
    ) -> int:
        header = lines[start]
        block = RigidBodyBlock(name=header.value, line_no=header.line_no)

        pending_density: list[GridScale] = []
        pending_potential: list[GridScale] = []
        pending_pmf: list[GridScale] = []

        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.key in BLOCK_HEADERS:
                break
            if line.key not in RIGID_BODY_FIELD_KEYS:
                if not self._misplaced_block_key(line, config, in_block="rigidBody"):
                    self._parse_top_level(line, config)
                i += 1
                continue

            key, value = line.key, line.value
            try:
                if key == "mass":
                    block.mass = float(tokenize(value)[0])
                elif key == "num":
                    n = parse_int(value, key=key, line=line, path=config.source_path)
                    if n < 0:
                        raise BdParseError(
                            f"rigidBody {block.name}: num must be >= 0, got {value!r}",
                            config.source_path,
                            line.line_no,
                            line.raw,
                        )
                    block.num = n
                elif key == "inertia":
                    block.inertia = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "transDamping":
                    block.trans_damping = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "rotDamping":
                    block.rot_damping = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "inputPdb":
                    block.input_pdb = value
                elif key == "inputPsf":
                    block.input_psf = value
                elif key == "referencePoint":
                    block.reference_point = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key in ("densityGrid", "potentialGrid") or key in RIGID_BODY_PMF_KEYS:
                    entry = _parse_grid_entry(value)
                    if entry is not None:
                        gkey, gfile = entry
                        target = {
                            "densityGrid": block.density_grids,
                            "potentialGrid": block.potential_grids,
                        }.get(key, block.pmf_grids)
                        target.append(GridEntry(key=gkey, filename=gfile))
                elif key in ("densityGridScale", "potentialGridScale", "pmfScale"):
                    scale = _parse_grid_scale(value)
                    if scale is not None:
                        {
                            "densityGridScale": pending_density,
                            "potentialGridScale": pending_potential,
                            "pmfScale": pending_pmf,
                        }[key].append(scale)
                elif key == "position":
                    block.position = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "orientation":
                    block.orientation = parse_matrix3_rows(value)
                elif key == "momentum":
                    block.momentum = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "angularMomentum":
                    block.angular_momentum = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "constantForce":
                    block.external_force = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
                elif key == "constantTorque":
                    block.external_torque = parse_vector3_strict(
                        value, key=key, line=line, path=config.source_path
                    )
            except BdParseError:
                raise
            except (ValueError, IndexError) as exc:
                raise BdParseError(
                    f"rigidBody {block.name}: {key}: could not parse {value!r}",
                    config.source_path,
                    line.line_no,
                    line.raw,
                ) from exc
            i += 1

        block.density_grids = _apply_rb_scales(block.density_grids, pending_density)
        block.potential_grids = _apply_rb_scales(block.potential_grids, pending_potential)
        block.pmf_grids = _apply_rb_scales(block.pmf_grids, pending_pmf)

        if bool(block.input_pdb) != bool(block.input_psf):
            raise BdParseError(
                f"rigidBody {block.name}: inputPdb and inputPsf must be given together",
                config.source_path,
                header.line_no,
                header.raw,
            )

        config.rigid_bodies.append(block)
        return i

def parse_file(path: str) -> BdConfig:
    """Convenience wrapper for :meth:`BdParser.parse_file`."""
    return BdParser().parse_file(path)


def parse_string(text: str, *, source_path: str = "") -> BdConfig:
    """Convenience wrapper for :meth:`BdParser.parse_string`."""
    return BdParser().parse_string(text, source_path=source_path)
