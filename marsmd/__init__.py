"""MARS molecular dynamics engine.

Two layers share this name:

- ``marsmd._core`` -- the compiled nanobind extension wrapping the C++ engine.
  Its public names are re-exported here, so ``from marsmd import SimSystem``
  works.
- ``marsmd.bd`` -- the pure-Python ``.bd`` parser and applier, which replaces
  the deprecated C++ config reader.

The extension is built into the active build directory and symlinked into this
package by ``build_cuda.sh``. Set ``PYTHONPATH`` to the repository root.

``marsmd.bd``'s parser never touches the extension, so it imports and runs
against an unbuilt tree. Touching an engine name without a built ``_core``
raises :class:`ImportError` with build instructions, at first use rather than
at import.

:example:
    >>> from marsmd.bd import parse_file      # no build needed
    >>> config = parse_file("run.bd")
    >>> from marsmd.bd.apply import BdSimulation   # needs the built extension
    >>> BdSimulation(config, gpu=0).simulate()
"""

from __future__ import annotations

import importlib
from types import ModuleType

__version__ = "2.0.0.dev0"

_IMPORT_HELP = """\
marsmd._core is not built.

Build it and refresh the symlink with:

    ./build_cuda.sh

That configures tbgl-cuda-release with -DUSE_PYTHON=ON and links
build/tbgl-cuda-release/src/Python/_core*.so into marsmd/.

Then run with the repository root on PYTHONPATH:

    PYTHONPATH=$PWD python -c 'import marsmd'

The pure-Python .bd parser (marsmd.bd) needs none of this."""

_CORE: ModuleType | None = None
_CORE_ERROR: BaseException | None = None
_CORE_TRIED = False


def _load_core(*, required: bool = True) -> ModuleType | None:
    """Import the extension on first use.

    Deferred rather than imported at module scope so that ``import marsmd.bd``
    -- which needs no engine at all -- does not drag in a CUDA-linked shared
    object, and so an unbuilt tree fails only when an engine object is actually
    wanted.
    """
    global _CORE, _CORE_ERROR, _CORE_TRIED
    if not _CORE_TRIED:
        _CORE_TRIED = True
        try:
            _CORE = importlib.import_module("._core", __name__)
        except ImportError as exc:  # pragma: no cover - depends on build state
            _CORE_ERROR = exc
    if _CORE is None and required:
        raise ImportError(f"{_IMPORT_HELP}\n\nOriginal error: {_CORE_ERROR}") from _CORE_ERROR
    return _CORE


def __getattr__(name: str):
    """Resolve engine names from ``_core`` on first use."""
    if name == "_core":
        return _load_core()
    core = _load_core()
    try:
        value = getattr(core, name)
    except AttributeError:
        raise AttributeError(f"module 'marsmd' has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    core = _load_core(required=False)
    names = [] if core is None else [n for n in dir(core) if not n.startswith("_")]
    return sorted({*globals(), *names})


__all__ = ["bd", "run", "__version__"]
