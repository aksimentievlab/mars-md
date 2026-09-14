"""Path resolution, byte-compatible with the C++ engine.

Direct port of ``MARS::resolve_file_path`` (``src/Header.h:64-90``). Not
``pathlib`` -- see dev_notes.md.
"""

from __future__ import annotations

import os

__all__ = ["resolve_file_path", "config_dir_of"]


def config_dir_of(config_file_path: str) -> str:
    """Return the directory prefix a relative path resolves against.

    Includes the trailing separator, or ``"./"`` when the config path has no
    separator at all.
    """
    last_slash = config_file_path.rfind("/")
    if last_slash != -1:
        return config_file_path[: last_slash + 1]
    return "./"


def resolve_file_path(file_path: str, config_file_path: str) -> str:
    """Resolve a path named inside a ``.bd`` file against that file's location.

    :param file_path: path as written in the config: absolute, ``~``-relative,
        or relative to the config file's directory.
    :param config_file_path: path of the config file itself.
    :returns: the resolved path, unnormalized.

    Absolute paths and ``~`` expansion match the C++ exactly, including the
    fallback of returning ``file_path`` unchanged when ``HOME`` is unset.
    """
    if file_path and file_path[0] == "/":
        return file_path

    if file_path and file_path[0] == "~":
        home = os.environ.get("HOME")
        if home is not None:
            return home + file_path[1:]
        return file_path

    return config_dir_of(config_file_path) + file_path
