"""The typed configuration graph a ``.bd`` file parses into.

Plain data only: no ``marsmd._core`` object is constructed here, and no
auxiliary file is opened. Filenames are recorded as written, unresolved --
:mod:`marsmd.bd.apply` resolves and loads them. That boundary is what lets the
parser be tested with no build and no fixture data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "Vec3",
    "Globals",
    "GridEntry",
    "GridScale",
    "ParticleBlock",
    "RigidBodyBlock",
    "TabulatedPair",
    "TopologyFile",
    "UnsupportedKey",
    "BdConfig",
]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class UnsupportedKey:
    """A key the engine does not act on: unrecognized, deprecated, or dropped."""

    key: str
    value: str
    line_no: int
    reason: str


@dataclass(frozen=True, slots=True)
class GridEntry:
    """One grid named by a block, with its deferred per-entry modifiers.

    :param key: logical name two rigid-body types must share to be paired for a
        grid-grid force. For particle ``gridFile`` entries the key is the
        filename itself.
    :param filename: path exactly as written in the config -- unresolved, so it
        stays usable as the ``GridManager`` map key.
    """

    key: str
    filename: str
    scale: float = 1.0
    scale_slope: float = 0.0
    boundary_condition: int = -1


@dataclass(frozen=True, slots=True)
class GridScale:
    """A deferred ``*GridScale`` line: which grid, and the value.

    An empty ``grid_key`` means the line gave a bare value, which applies to
    every grid of that kind.
    """

    grid_key: str
    value: float


@dataclass(frozen=True, slots=True)
class TabulatedPair:
    """One ``tabulatedFile i@j@path`` entry."""

    type_id_1: int
    type_id_2: int
    filename: str


@dataclass(frozen=True, slots=True)
class TopologyFile:
    """One ``input{Bonds,Angles,...}`` entry.

    ``kind`` records which key named the file, for diagnostics only: the reader
    dispatches on the record type inside the file, exactly as
    ``BondConfigReader::read_file`` does.
    """

    kind: str
    filename: str


@dataclass(slots=True)
class Globals:
    """Top-level scalars, with the engine's own defaults pre-applied.

    Defaults mirror ``ConfigParser::apply_defaults`` (``ConfigParser.cpp:98``),
    so a config that omits a key produces the same value the C++ would.
    """

    temperature: float = 298.15
    temperature_grid: str | None = None
    seed: int | None = None
    cutoff: float = 10.0
    pairlist_distance: float = 0.0
    timestep: float = 1e-5
    steps: int = 1000
    pairlist_rebuild_period: float = 1000.0
    reorder_period: int | None = None
    output_period: float = 10.0
    output_energy_period: float = 100.0
    output_name: str = "out"
    output_format: str | None = None
    system_size: Vec3 | None = None
    origin: Vec3 | None = None
    decomposer: str | None = None
    long_range_method: str | None = None
    particle_dynamic_type: str | None = None
    rigid_body_dynamic_type: str | None = None
    rb_grid_grid_period: int | None = None

    @property
    def pairlist_cutoff(self) -> float:
        """The neighbor-list cutoff the engine actually sets.

        ``pairlistDistance`` is a skin added to ``cutoff``; the C++ always sets
        this, using a zero skin when unspecified, rather than leaving
        ``SimSystem``'s own default (``ConfigParser.cpp:172-180``).
        """
        return self.cutoff + self.pairlist_distance


@dataclass(slots=True)
class ParticleBlock:
    """One ``particle <name>`` block."""

    name: str
    line_no: int = 0
    num: int = 0
    mass: float = 1.0
    diffusion: Vec3 | None = None
    trans_damping: Vec3 | None = None
    #: ``gridFile`` entries, in declaration order. Each may name several files
    #: on one line; each file becomes its own entry.
    grid_files: list[GridEntry] = field(default_factory=list)
    diffusion_grid_file: str | None = None
    #: Exactly three, or empty.
    force_grid_files: list[str] = field(default_factory=list)
    force_grid_scale: Vec3 | None = None
    grid_file_smd: int = 0
    rigid_body_potential_keys: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RigidBodyBlock:
    """One ``rigidBody <name>`` block: a type plus ``num`` instances.

    Instances share the block's position/orientation/momentum. A top-level
    ``inputRBCoordinates`` file overwrites them per instance.
    """

    name: str
    line_no: int = 0
    num: int = 1
    mass: float = 1.0
    inertia: Vec3 | None = None
    trans_damping: Vec3 | None = None
    rot_damping: Vec3 | None = None

    potential_grids: list[GridEntry] = field(default_factory=list)
    density_grids: list[GridEntry] = field(default_factory=list)
    pmf_grids: list[GridEntry] = field(default_factory=list)

    input_pdb: str | None = None
    input_psf: str | None = None
    reference_point: Vec3 = (0.0, 0.0, 0.0)

    # Per-instance.
    position: Vec3 = (0.0, 0.0, 0.0)
    orientation: list[list[float]] | None = None
    momentum: Vec3 = (0.0, 0.0, 0.0)
    angular_momentum: Vec3 = (0.0, 0.0, 0.0)
    external_force: Vec3 = (0.0, 0.0, 0.0)
    external_torque: Vec3 = (0.0, 0.0, 0.0)

    @property
    def has_orientation(self) -> bool:
        return self.orientation is not None


@dataclass(slots=True)
class BdConfig:
    """Everything one ``.bd`` file declares.

    :param source_path: the config's own path, as given. Every relative
        filename in this graph resolves against its directory.
    """

    source_path: str = ""
    globals: Globals = field(default_factory=Globals)
    particles: list[ParticleBlock] = field(default_factory=list)
    rigid_bodies: list[RigidBodyBlock] = field(default_factory=list)
    tabulated_pairs: list[TabulatedPair] = field(default_factory=list)
    topology_files: list[TopologyFile] = field(default_factory=list)
    input_particles: str | None = None
    restart_coordinates: str | None = None
    input_rb_coordinates: str | None = None
    unsupported: list[UnsupportedKey] = field(default_factory=list)

    @property
    def has_explicit_particle_source(self) -> bool:
        """True when per-particle data comes from a file.

        When false, the engine synthesizes ``num`` placeholder particles per
        ``particle`` block (``ConfigParser.cpp:834-841``).
        """
        return self.input_particles is not None or self.restart_coordinates is not None

    def particle_type_index(self, name: str) -> int:
        """Insertion-order index of a particle type, or -1.

        This is the id ``tabulatedFile i@j@...`` refers to, and the id
        ``assign_particle_type_ids()`` will assign.
        """
        for i, block in enumerate(self.particles):
            if block.name == name:
                return i
        return -1

    def to_dict(self) -> dict[str, Any]:
        """A plain-dict view, for golden snapshots and ``--dump-json``."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
