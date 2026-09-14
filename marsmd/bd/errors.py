"""Exceptions raised while reading a ``.bd`` configuration."""

from __future__ import annotations


class BdError(Exception):
    """Base class for every ``.bd`` reading failure."""


class BdParseError(BdError):
    """A line could not be parsed, or a value had the wrong shape.

    Carries the source location so the message points at the offending line
    rather than at a stack frame.
    """

    def __init__(self, message: str, path: str = "", line_no: int = 0, line: str = ""):
        self.path = path
        self.line_no = line_no
        self.line = line
        where = f"{path}:{line_no}" if path else f"line {line_no}"
        detail = f"\n    {line.strip()}" if line else ""
        super().__init__(f"{where}: {message}{detail}")


class BdKeywordError(BdParseError):
    """An unrecognized keyword was seen under ``strict`` policy."""
