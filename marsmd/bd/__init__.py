"""Pure-Python reader for ARBD ``.bd`` configuration files.

Two stages, deliberately separated:

1. :mod:`~marsmd.bd.parser` turns ``.bd`` text into a
   :class:`~marsmd.bd.model.BdConfig` -- plain typed data, no engine objects,
   no auxiliary file opened.
2. :mod:`~marsmd.bd.apply` resolves the filenames that graph records, loads
   them through :mod:`~marsmd.bd.aux`, and builds the engine objects.

Nothing here imports ``marsmd._core``, so the parser can be used and tested
without a compiled engine.

:example:
    >>> from marsmd.bd import parse_file
    >>> config = parse_file("run.bd")
    >>> config.globals.steps
    100000
"""

from __future__ import annotations

from .errors import BdError, BdParseError
from .model import (
    BdConfig,
    Globals,
    GridEntry,
    ParticleBlock,
    RigidBodyBlock,
    TabulatedPair,
    TopologyFile,
    UnsupportedKey,
)
from .parser import BdParser, parse_file, parse_string
from .paths import resolve_file_path

__all__ = [
    "BdConfig",
    "BdError",
    "BdParseError",
    "BdParser",
    "Globals",
    "GridEntry",
    "ParticleBlock",
    "RigidBodyBlock",
    "TabulatedPair",
    "TopologyFile",
    "UnsupportedKey",
    "parse_file",
    "parse_string",
    "resolve_file_path",
]
