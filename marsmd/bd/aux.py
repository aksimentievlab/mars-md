"""Readers for the files a ``.bd`` names but does not contain.

Kept out of :mod:`marsmd.bd.parser` on purpose: the parser records filenames,
these open them. That split is what lets grammar bugs and I/O bugs be told
apart, and lets a config be parsed without its data present.

Every reader takes an already-resolved path. Resolve with
:func:`marsmd.bd.paths.resolve_file_path` first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .keywords import (
    ANALYTICAL_ANGLE_TYPES,
    ANALYTICAL_BOND_TYPES,
    ANALYTICAL_DIHEDRAL_TYPES,
)
from .tokens import is_comment_or_blank, tokenize

__all__ = [
    "ParticleRecord",
    "BondRecord",
    "AngleRecord",
    "DihedralRecord",
    "ExcludeRecord",
    "RestraintRecord",
    "Topology",
    "RigidBodyCoord",
    "read_particles_file",
    "read_restart_file",
    "read_topology_file",
    "read_rigid_body_coords",
]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParticleRecord:
    """One particle from ``inputParticles`` or ``restartCoordinates``."""

    id: int
    type_name: str
    position: tuple[float, float, float]
    momentum: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class BondRecord:
    ind1: int
    ind2: int
    function_name: str
    #: ``"Analytical"`` or ``"Tabulated"``.
    form: str
    #: Index into the analytical table, or -1 when the name is a file path that
    #: only ``TablesRegistry`` can resolve.
    function_index: int
    flag: str
    #: True when the bond supersedes the pair's nonbonded interaction, so the
    #: pair must also be excluded.
    replaces_nonbonded: bool


@dataclass(frozen=True, slots=True)
class AngleRecord:
    ind1: int
    ind2: int
    ind3: int
    function_name: str
    form: str
    function_index: int


@dataclass(frozen=True, slots=True)
class DihedralRecord:
    ind1: int
    ind2: int
    ind3: int
    ind4: int
    function_name: str
    form: str
    function_index: int


@dataclass(frozen=True, slots=True)
class ExcludeRecord:
    ind1: int
    ind2: int


@dataclass(frozen=True, slots=True)
class RestraintRecord:
    ind: int
    k: float
    r0: tuple[float, float, float]


@dataclass(slots=True)
class Topology:
    """Everything one or more topology files declare."""

    bonds: list[BondRecord] = field(default_factory=list)
    angles: list[AngleRecord] = field(default_factory=list)
    dihedrals: list[DihedralRecord] = field(default_factory=list)
    exclusions: list[ExcludeRecord] = field(default_factory=list)
    restraints: list[RestraintRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return (
            len(self.bonds)
            + len(self.angles)
            + len(self.dihedrals)
            + len(self.exclusions)
            + len(self.restraints)
        )


@dataclass(frozen=True, slots=True)
class RigidBodyCoord:
    """One instance's coordinates from ``inputRBCoordinates``."""

    position: tuple[float, float, float]
    #: Row-major 3x3.
    orientation: list[list[float]]
    momentum: tuple[float, float, float] | None = None
    angular_momentum: tuple[float, float, float] | None = None


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def read_particles_file(path: str) -> list[ParticleRecord]:
    """Read an ``inputParticles`` file.

    Format is ``ATOM id type x y z [px py pz]``. Lines with fewer than 6 tokens
    are skipped silently, matching ``load_particles_file``
    (``ConfigParser.cpp:495-532``) -- the momentum columns are optional because
    legacy writers emit exactly the six.
    """
    out: list[ParticleRecord] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if is_comment_or_blank(raw):
                continue
            toks = tokenize(raw)
            if len(toks) < 6:
                continue
            momentum = (0.0, 0.0, 0.0)
            if len(toks) >= 9:
                momentum = (float(toks[6]), float(toks[7]), float(toks[8]))
            out.append(
                ParticleRecord(
                    id=int(toks[1]),
                    type_name=toks[2],
                    position=(float(toks[3]), float(toks[4]), float(toks[5])),
                    momentum=momentum,
                )
            )
    return out


def read_restart_file(path: str, type_names: list[str]) -> list[ParticleRecord]:
    """Read a ``restartCoordinates`` file.

    Format is ``type_id x y z``; the particle id is the line's 0-based position
    among accepted lines. ``type_id`` indexes ``type_names`` in declaration
    order. An out-of-range id skips the line without advancing the counter,
    matching ``load_restart_file`` (``ConfigParser.cpp:534-582``).
    """
    out: list[ParticleRecord] = []
    line_number = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if is_comment_or_blank(raw):
                continue
            toks = tokenize(raw)
            if len(toks) < 4:
                continue
            type_id = int(toks[0])
            if not (0 <= type_id < len(type_names)):
                continue
            out.append(
                ParticleRecord(
                    id=line_number,
                    type_name=type_names[type_id],
                    position=(float(toks[1]), float(toks[2]), float(toks[3])),
                )
            )
            line_number += 1
    return out


def _classify(name: str, analytical: tuple[str, ...]) -> tuple[str, int]:
    """Decide whether a function name is analytical or a tabulated file path."""
    if name in analytical:
        return "Analytical", analytical.index(name)
    return "Tabulated", -1


def read_topology_file(path: str, topology: Topology | None = None) -> Topology:
    """Read one topology file into ``topology`` (a new one if not given).

    One reader for every ``input*`` key, because the engine dispatches on the
    record type at the start of each line -- ``BOND``, ``ANGLE``, ``DIHEDRAL``,
    ``EXCLUDE``, ``RESTRAINT`` -- not on which key named the file. Unknown
    record types are ignored, as in ``BondConfigReader::read_file``.

    Tabulated ``function_index`` is left at -1: only ``TablesRegistry`` can
    assign it, and it must be assigned before the terms reach the device.
    """
    topo = topology if topology is not None else Topology()

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if is_comment_or_blank(raw):
                continue
            toks = tokenize(raw)
            if not toks:
                continue
            record, rest = toks[0], toks[1:]

            if record == "BOND":
                bond = _parse_bond(rest)
                if bond is not None:
                    topo.bonds.append(bond)
                    if bond.replaces_nonbonded:
                        topo.exclusions.append(ExcludeRecord(bond.ind1, bond.ind2))
            elif record == "ANGLE" and len(rest) >= 4:
                form, idx = _classify(rest[3], ANALYTICAL_ANGLE_TYPES)
                topo.angles.append(
                    AngleRecord(
                        int(rest[0]), int(rest[1]), int(rest[2]), rest[3], form, idx
                    )
                )
            elif record == "DIHEDRAL" and len(rest) >= 5:
                form, idx = _classify(rest[4], ANALYTICAL_DIHEDRAL_TYPES)
                topo.dihedrals.append(
                    DihedralRecord(
                        int(rest[0]),
                        int(rest[1]),
                        int(rest[2]),
                        int(rest[3]),
                        rest[4],
                        form,
                        idx,
                    )
                )
            elif record == "EXCLUDE" and len(rest) >= 2:
                topo.exclusions.append(ExcludeRecord(int(rest[0]), int(rest[1])))
            elif record == "RESTRAINT" and len(rest) >= 5:
                topo.restraints.append(
                    RestraintRecord(
                        ind=int(rest[0]),
                        k=float(rest[1]),
                        r0=(float(rest[2]), float(rest[3]), float(rest[4])),
                    )
                )

    return topo


def _parse_bond(tokens: list[str]) -> BondRecord | None:
    """Parse a ``BOND`` record's fields.

    Legacy form is ``FLAG ind1 ind2 file``, where ``FLAG`` is ``REPLACE`` or
    ``ADD``. Files written without a flag start at ``ind1``, so the flag is
    optional. A ``REPLACE`` bond (and a flagless one) supersedes the pair's
    nonbonded interaction, so the caller must also add the exclusion.
    """
    if not tokens:
        return None

    first = tokens[0]
    if first in ("REPLACE", "ADD"):
        flag = first
        replaces = first == "REPLACE"
        rest = tokens[1:]
    else:
        flag = "DEFAULT"
        replaces = True
        rest = tokens
        if not (first.lstrip("+-").isdigit()):
            return None

    if len(rest) < 3:
        return None
    try:
        ind1, ind2 = int(rest[0]), int(rest[1])
    except ValueError:
        return None
    name = rest[2]

    if ind1 < 0 or ind2 < 0 or ind1 == ind2:
        return None

    form, idx = _classify(name, ANALYTICAL_BOND_TYPES)
    return BondRecord(
        ind1=ind1,
        ind2=ind2,
        function_name=name,
        form=form,
        function_index=idx,
        flag=flag,
        replaces_nonbonded=replaces,
    )


def read_rigid_body_coords(path: str) -> list[RigidBodyCoord]:
    """Read an ``inputRBCoordinates`` file.

    Each non-comment line is at least 12 floats: 3 position, then 9 row-major
    orientation. 18 values add momentum and angular momentum. Orientation stays
    row-major here -- ``PyTypeCasters.h``'s ``Matrix3`` caster transposes on
    the way into C++, so the applier passes these rows through unchanged.
    """
    out: list[RigidBodyCoord] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_number, raw in enumerate(fh, start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            values = [float(t) for t in tokenize(raw)]
            if not values:
                continue
            if len(values) < 12:
                raise ValueError(
                    f"{path}:{line_number}: got {len(values)} values, expected at "
                    "least 12 (3 position + 9 row-major orientation)"
                )
            coord = RigidBodyCoord(
                position=(values[0], values[1], values[2]),
                orientation=[values[3:6], values[6:9], values[9:12]],
                momentum=(
                    (values[12], values[13], values[14]) if len(values) >= 18 else None
                ),
                angular_momentum=(
                    (values[15], values[16], values[17]) if len(values) >= 18 else None
                ),
            )
            out.append(coord)
    return out
