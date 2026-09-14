"""Line iteration and tokenizing, matching the C++ ``Reader``.

Ports ``Reader::isValidParameterLine`` / ``Reader::parseLine``
(``src/IO/Reader.h:410-451``) and ``ConfigParser``'s file-local
``is_comment_or_blank`` / ``tokenize`` (``src/IO/ConfigParser.cpp:312-324``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .errors import BdParseError

__all__ = [
    "SourceLine",
    "is_comment_or_blank",
    "tokenize",
    "split_key_value",
    "strip_trailing_comment",
    "iter_parameter_lines",
    "parse_float",
    "parse_int",
    "parse_vector3",
    "parse_matrix3_rows",
]


@dataclass(frozen=True, slots=True)
class SourceLine:
    """One ``key value`` pair with the location it came from."""

    key: str
    value: str
    line_no: int
    raw: str


def is_comment_or_blank(line: str) -> bool:
    """True for lines the engine skips: blank, whitespace-only, or ``#``-led."""
    stripped = line.lstrip()
    return not stripped or stripped[0] == "#"


def tokenize(value: str) -> list[str]:
    """Split on arbitrary whitespace, dropping empties -- C++ ``iss >> token``."""
    return value.split()


def strip_trailing_comment(value: str) -> str:
    """Drop a trailing ``#`` comment from a value.

    Off by default. C++ keeps the comment text in the value and survives only
    because ``std::stoi`` stops at the first non-digit -- so ``0 # deprecated``
    reads as ``0``. Any code path that does not go through ``stoi`` sees the
    whole string.
    """
    hash_pos = value.find("#")
    if hash_pos == -1:
        return value
    return value[:hash_pos].rstrip()


def split_key_value(line: str) -> tuple[str, str]:
    """Split a line into its first token and the whitespace-joined remainder.

    Matches ``Reader::parseLine``: the value is re-joined with single spaces,
    so runs of whitespace in the original collapse.
    """
    toks = line.split()
    if not toks:
        return "", ""
    return toks[0], " ".join(toks[1:])


def iter_parameter_lines(
    text: str, *, strip_comments: bool = False
) -> Iterator[SourceLine]:
    """Yield every parameter line of a config file, in order.

    :param text: full file contents.
    :param strip_comments: drop trailing ``#`` comments from values. Off by
        default to match the engine.
    """
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if is_comment_or_blank(raw):
            continue
        key, value = split_key_value(raw)
        if not key:
            continue
        if strip_comments:
            value = strip_trailing_comment(value)
        yield SourceLine(key=key, value=value, line_no=line_no, raw=raw)


# ---------------------------------------------------------------------------
# value parsers
# ---------------------------------------------------------------------------


def parse_float(value: str, *, key: str, line: SourceLine, path: str = "") -> float:
    try:
        return float(tokenize(value)[0])
    except (ValueError, IndexError) as exc:
        raise BdParseError(
            f"{key}: expected a number, got {value!r}", path, line.line_no, line.raw
        ) from exc


def parse_int(value: str, *, key: str, line: SourceLine, path: str = "") -> int:
    """Parse an integer the way C++ ``std::stoi`` does.

    ``stoi`` stops at the first character it cannot consume, so
    ``"0 # deprecated"`` is ``0`` and ``"12abc"`` is ``12``. Reproduced here so
    values carrying trailing comments parse identically.
    """
    text = value.strip()
    end = 0
    if end < len(text) and text[end] in "+-":
        end += 1
    digits_start = end
    while end < len(text) and text[end].isdigit():
        end += 1
    if end == digits_start:
        raise BdParseError(
            f"{key}: expected an integer, got {value!r}", path, line.line_no, line.raw
        )
    return int(text[:end])


def parse_vector3(
    value: str, *, key: str, line: SourceLine, path: str = "", fallback=(0.0, 0.0, 0.0)
) -> tuple[float, float, float]:
    """Parse ``"x y z"``.

    A single value broadcasts to all three components -- the C++ does this for
    ``diffusion`` and ``transDamping`` inside particle blocks
    (``ConfigParser.cpp:677-712``), but *not* in its standalone
    ``parse_vector3`` helper, which warns and falls back. Callers choose via
    :func:`parse_vector3_strict`.
    """
    toks = tokenize(value)
    if len(toks) == 3:
        return (float(toks[0]), float(toks[1]), float(toks[2]))
    if len(toks) == 1:
        v = float(toks[0])
        return (v, v, v)
    raise BdParseError(
        f"{key}: expected 3 values (or 1 to broadcast), got {value!r}",
        path,
        line.line_no,
        line.raw,
    )


def parse_vector3_strict(
    value: str, *, key: str, line: SourceLine, path: str = "", fallback=(0.0, 0.0, 0.0)
) -> tuple[float, float, float]:
    """Parse exactly ``"x y z"``; warn-and-fall-back on any other arity.

    Port of ``ConfigParser.cpp``'s ``parse_vector3``, used by the rigidBody
    block, which does not broadcast a single value.
    """
    toks = tokenize(value)
    if len(toks) != 3:
        return fallback
    return (float(toks[0]), float(toks[1]), float(toks[2]))


def parse_matrix3_rows(value: str) -> list[list[float]]:
    """Parse 9 values as three row-major rows.

    Returned row-major. The C++ transposes here because ``Matrix3``'s
    constructor takes column vectors; ``PyTypeCasters.h`` already transposes on
    the way in, so the applier hands nanobind these rows unchanged.
    Falls back to the identity on any other arity, matching the C++.
    """
    toks = tokenize(value)
    if len(toks) != 9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    m = [float(t) for t in toks]
    return [m[0:3], m[3:6], m[6:9]]
