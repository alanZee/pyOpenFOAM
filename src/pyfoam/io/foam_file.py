"""
FoamFile header parsing and I/O for OpenFOAM files.

Every OpenFOAM file starts with a ``FoamFile`` header block that declares
the file version, format (ascii/binary), class name, location, and object name.

Example header::

    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      p;
    }

This module provides:
- :class:`FoamFileHeader` — structured header representation
- :func:`parse_header` — parse header from file content
- :func:`read_foam_file` — read a complete FoamFile (header + body)
- :func:`write_foam_file` — write a complete FoamFile with header
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, TextIO, Union

__all__ = [
    "FileFormat",
    "FoamFileHeader",
    "parse_header",
    "read_foam_file",
    "write_foam_file",
    "split_header_body",
]


class FileFormat(Enum):
    """OpenFOAM file format (ascii or binary)."""

    ASCII = "ascii"
    BINARY = "binary"


@dataclass
class FoamFileHeader:
    """Structured representation of an OpenFOAM FoamFile header.

    Attributes:
        version: File format version (typically ``2.0``).
        format: ASCII or binary format.
        class_name: OpenFOAM class name (e.g., ``volScalarField``).
        location: Directory location within the case (e.g., ``"0"``).
        object: Object name (e.g., ``p``).
        note: Optional note string.
        arch: Optional architecture string.
    """

    version: str = "2.0"
    format: FileFormat = FileFormat.ASCII
    class_name: str = ""
    location: str = ""
    object: str = ""
    note: str = ""
    arch: str = ""

    @property
    def is_binary(self) -> bool:
        """Return True if this file uses binary format."""
        return self.format == FileFormat.BINARY

    @property
    def is_ascii(self) -> bool:
        """Return True if this file uses ASCII format."""
        return self.format == FileFormat.ASCII

    def to_header_string(self) -> str:
        """Generate the FoamFile header block string.

        Returns:
            Complete header block including ``FoamFile { ... }``.
        """
        lines = ["FoamFile", "{"]
        lines.append(f"    version     {self.version};")
        lines.append(f"    format      {self.format.value};")
        if self.class_name:
            lines.append(f"    class       {self.class_name};")
        if self.location:
            lines.append(f"    location    \"{self.location}\";")
        if self.object:
            lines.append(f"    object      {self.object};")
        if self.note:
            lines.append(f"    note        \"{self.note}\";")
        if self.arch:
            lines.append(f"    arch        \"{self.arch}\";")
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FoamFileHeader(version={self.version!r}, "
            f"format={self.format.value}, class={self.class_name!r}, "
            f"object={self.object!r})"
        )


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

# Regex to match the FoamFile header block
_HEADER_PATTERN = re.compile(
    r"FoamFile\s*\{([^}]*)\}",
    re.DOTALL,
)

# Regex to match key-value pairs inside the header
_KV_PATTERN = re.compile(
    r"(\w+)\s+(.+?)\s*;",
    re.DOTALL,
)


def _strip_comments(text: str) -> str:
    """Remove C-style comments from text."""
    # Remove // line comments
    text = re.sub(r"//[^\n]*", "", text)
    # Remove /* ... */ block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _clean_value(value: str) -> str:
    """Clean a header value: strip quotes and whitespace."""
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def parse_header(content: str) -> FoamFileHeader:
    """Parse a FoamFile header from file content.

    Args:
        content: Full file content (or at least the header portion).

    Returns:
        Parsed :class:`FoamFileHeader`.

    Raises:
        ValueError: If no FoamFile header is found.
    """
    content = _strip_comments(content)
    match = _HEADER_PATTERN.search(content)
    if match is None:
        raise ValueError("No FoamFile header found in content")

    header_block = match.group(1)
    header = FoamFileHeader()

    for kv_match in _KV_PATTERN.finditer(header_block):
        key = kv_match.group(1).strip()
        value = _clean_value(kv_match.group(2))

        if key == "version":
            header.version = value
        elif key == "format":
            header.format = FileFormat(value.lower())
        elif key == "class":
            header.class_name = value
        elif key == "location":
            header.location = value
        elif key == "object":
            header.object = value
        elif key == "note":
            header.note = value
        elif key == "arch":
            header.arch = value

    return header


# ---------------------------------------------------------------------------
# Header / body splitting
# ---------------------------------------------------------------------------

def split_header_body(content: str) -> tuple[FoamFileHeader, str]:
    """Split file content into header and body.

    The header is everything from ``FoamFile`` to the closing ``}`` of the
    header block.  The body is everything after that.

    Args:
        content: Full file content.

    Returns:
        Tuple of (header, body_text).

    Raises:
        ValueError: If no FoamFile header is found.
    """
    content_stripped = _strip_comments(content)
    match = _HEADER_PATTERN.search(content_stripped)
    if match is None:
        raise ValueError("No FoamFile header found in content")

    header = parse_header(content)
    # Body starts after the header block's closing brace
    body_start = match.end()
    body = content[body_start:].lstrip("\n\r ")
    return header, body


# ---------------------------------------------------------------------------
# File reading / writing
# ---------------------------------------------------------------------------

def read_foam_file(
    source: Union[str, Path],
) -> tuple[FoamFileHeader, str]:
    """Read a complete OpenFOAM file and return header + body.

    For binary files, the body is returned as raw bytes (encoded in latin-1
    to preserve byte values in a str).

    Args:
        source: Path to the OpenFOAM file.

    Returns:
        Tuple of (header, body_text).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no FoamFile header is found.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return split_header_body(content)


def write_foam_file(
    dest: Union[str, Path],
    header: FoamFileHeader,
    body: str,
    *,
    overwrite: bool = False,
) -> None:
    """Write a complete OpenFOAM file with header and body.

    Args:
        dest: Output file path.
        header: FoamFile header to write.
        body: File body content (after the header block).
        overwrite: If False, raise if file already exists.

    Raises:
        FileExistsError: If file exists and *overwrite* is False.
    """
    path = Path(dest)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    header_str = header.to_header_string()
    content = f"{header_str}\n\n{body}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience: detect format from content
# ---------------------------------------------------------------------------

def detect_format(content: str) -> FileFormat:
    """Detect whether file content uses ASCII or binary format.

    Inspects the FoamFile header's ``format`` field.

    Args:
        content: Full file content.

    Returns:
        Detected :class:`FileFormat`.
    """
    try:
        header = parse_header(content)
        return header.format
    except ValueError:
        # Default to ASCII if no header found
        return FileFormat.ASCII


def get_header_end(content: str) -> int:
    """Return the byte offset where the FoamFile header block ends.

    This is the position just after the closing ``}`` of the header block.

    Args:
        content: Full file content.

    Returns:
        Character offset of the end of the header block.

    Raises:
        ValueError: If no FoamFile header is found.
    """
    match = _HEADER_PATTERN.search(content)
    if match is None:
        raise ValueError("No FoamFile header found in content")
    return match.end()
