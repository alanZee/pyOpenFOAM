"""
foamDictionary — query and modify OpenFOAM dictionary entries.

Mirrors OpenFOAM's ``foamDictionary`` utility.  Supports:

- **Query**: retrieve a single value by key path or dump the entire dictionary.
- **Modify**: set a value for a given key and write back to file.

Key paths use ``/``-separated syntax (e.g., ``"subDict/key"``), matching
the ``FoamDict.get_path`` interface.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

from pyfoam.io.dictionary import FoamDict, FoamList, parse_dict_file

__all__ = ["foam_dictionary"]


def foam_dictionary(
    case_path: Union[str, Path],
    dict_path: Union[str, Path],
    key: Optional[str] = None,
    value: Optional[Any] = None,
) -> Any:
    """Query or modify an OpenFOAM dictionary entry.

    Behaviour:

    - ``key=None, value=None`` — return the entire :class:`FoamDict`.
    - ``key="path", value=None`` — return the value at *key* (``/``-separated
      path).  Raises ``KeyError`` if the path does not exist.
    - ``key="path", value=<val>`` — set the value at *key* and write the
      modified dictionary back to the file on disk.

    Args:
        case_path: Root of the OpenFOAM case directory.
        dict_path: Path to the dictionary file *relative to case_path*
            (e.g., ``"system/controlDict"``).
        key: ``/``-separated key path to query or set.
        value: Value to write.  Only used when *key* is not ``None``.

    Returns:
        The queried value (scalar, string, :class:`FoamDict`, or
        :class:`FoamList`), or the entire dictionary when *key* is ``None``.

    Raises:
        FileNotFoundError: If the dictionary file does not exist.
        KeyError: If *key* is given but not found in the dictionary.
    """
    case_dir = Path(case_path)
    file_path = case_dir / dict_path

    if not file_path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {file_path}")

    d = parse_dict_file(file_path)

    # --- Full dump ---
    if key is None:
        return d

    # --- Read mode ---
    if value is None:
        return _get_nested(d, key)

    # --- Write mode ---
    _set_nested(d, key, value)
    _write_back(file_path, d)
    return value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_nested(d: FoamDict, key: str) -> Any:
    """Get a value by ``/``-separated path, raising KeyError on miss."""
    try:
        return d[key]
    except KeyError:
        raise KeyError(f"Key '{key}' not found in dictionary")


def _set_nested(d: FoamDict, key: str, value: Any) -> None:
    """Set a value by ``/``-separated path.

    Creates intermediate sub-dictionaries as needed.
    """
    parts = key.split("/")
    current = d
    for part in parts[:-1]:
        if part not in current:
            current[part] = FoamDict(parent=current)
        sub = current[part]
        if not isinstance(sub, FoamDict):
            raise KeyError(
                f"Cannot set '{key}': intermediate key '{part}' is not a sub-dictionary"
            )
        current = sub
    current[parts[-1]] = value


def _write_back(file_path: Path, d: FoamDict) -> None:
    """Serialize a :class:`FoamDict` back to an OpenFOAM dictionary file.

    Attempts to preserve the existing FoamFile header if one is present.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")

    # 尝试保留原始 header
    header_str = ""
    body_start = 0
    try:
        from pyfoam.io.foam_file import split_header_body
        header, _ = split_header_body(content)
        # 重新定位 header 结束位置
        match = re.search(r"FoamFile\s*\{[^}]*\}", content, re.DOTALL)
        if match:
            header_str = content[: match.end()]
            body_start = match.end()
    except ValueError:
        pass  # 无 header

    body = _serialize_dict(d)
    if header_str:
        new_content = f"{header_str}\n\n{body}\n"
    else:
        new_content = f"{body}\n"

    file_path.write_text(new_content, encoding="utf-8")


def _serialize_dict(d: FoamDict, indent: int = 0) -> str:
    """Serialize a :class:`FoamDict` to OpenFOAM dictionary text."""
    pad = "    " * indent
    lines: list[str] = []
    for key, value in d.items():
        if isinstance(value, FoamDict):
            lines.append(f"{pad}{key}")
            lines.append(f"{pad}{{")
            lines.append(_serialize_dict(value, indent + 1))
            lines.append(f"{pad}}};")
        elif isinstance(value, FoamList):
            items_str = " ".join(_format_value(v) for v in value)
            lines.append(f"{pad}{key} ({items_str});")
        else:
            lines.append(f"{pad}{key} {_format_value(value)};")
    return "\n".join(lines)


def _format_value(v: Any) -> str:
    """Format a single value for OpenFOAM dictionary output."""
    if isinstance(v, str):
        # 字符串若含空格或特殊字符则加引号
        if " " in v or "\t" in v or '"' in v:
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return v
    if isinstance(v, float):
        return f"{v}"
    if isinstance(v, int):
        return str(v)
    return str(v)
