"""
OpenFOAM field file reading and writing.

Handles volScalarField, volVectorField, surfaceScalarField, etc.

Field file structure::

    FoamFile { ... }

    dimensions      [0 2 -2 0 0 0 0];

    internalField   uniform 1.0;
    // or
    internalField   nonuniform List<scalar> 100
    (
        1.0
        2.0
        ...
    );

    boundaryField
    {
        patchName
        {
            type        fixedValue;
            value       uniform 0;
        }
    }

Supports both ASCII and binary formats.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.binary_io import BinaryReader, BinaryWriter
from pyfoam.io.dictionary import FoamDict, FoamList, parse_dict
from pyfoam.io.foam_file import (
    FoamFileHeader,
    FileFormat,
    read_foam_file,
    split_header_body,
    write_foam_file,
)

__all__ = [
    "FieldData",
    "BoundaryField",
    "BoundaryPatch",
    "read_field",
    "write_field",
    "read_dimensions",
    "parse_internal_field",
    "parse_boundary_field",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class BoundaryPatch:
    """A single boundary patch with its type and values.

    Attributes:
        name: Patch name.
        patch_type: Boundary condition type (e.g., ``fixedValue``).
        value: Patch values (uniform scalar/vector or nonuniform list).
        n_faces: Number of faces in this patch.
        start_face: Starting face index.
    """

    def __init__(
        self,
        name: str,
        patch_type: str = "",
        value: Any = None,
        n_faces: int = 0,
        start_face: int = 0,
    ) -> None:
        self.name = name
        self.patch_type = patch_type
        self.value = value
        self.n_faces = n_faces
        self.start_face = start_face

    def __repr__(self) -> str:
        return (
            f"BoundaryPatch(name={self.name!r}, type={self.patch_type!r}, "
            f"n_faces={self.n_faces})"
        )


class BoundaryField:
    """Collection of boundary patches for a field.

    Attributes:
        patches: List of :class:`BoundaryPatch` objects.
    """

    def __init__(self, patches: Optional[list[BoundaryPatch]] = None) -> None:
        self.patches: list[BoundaryPatch] = patches or []

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self):
        return iter(self.patches)

    def __getitem__(self, name: str) -> BoundaryPatch:
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(f"Patch '{name}' not found")

    def __contains__(self, name: str) -> bool:
        return any(p.name == name for p in self.patches)

    def __repr__(self) -> str:
        return f"BoundaryField(patches={len(self.patches)})"


class FieldData:
    """Complete field data read from an OpenFOAM field file.

    Attributes:
        header: FoamFile header.
        dimensions: 7-element dimension vector [mass length time temp qty current luminous].
        internal_field: Internal field values (torch.Tensor for nonuniform, scalar/vector for uniform).
        boundary_field: Boundary field data.
        is_uniform: True if the internal field is uniform.
        scalar_type: Type tag (``"scalar"``, ``"vector"``, ``"symmTensor"``, ``"tensor"``).
    """

    def __init__(
        self,
        header: FoamFileHeader,
        dimensions: list[float],
        internal_field: Any,
        boundary_field: BoundaryField,
        *,
        is_uniform: bool = True,
        scalar_type: str = "scalar",
    ) -> None:
        self.header = header
        self.dimensions = dimensions
        self.internal_field = internal_field
        self.boundary_field = boundary_field
        self.is_uniform = is_uniform
        self.scalar_type = scalar_type

    def __repr__(self) -> str:
        return (
            f"FieldData(class={self.header.class_name!r}, "
            f"uniform={self.is_uniform}, type={self.scalar_type!r})"
        )


# ---------------------------------------------------------------------------
# Dimension parsing
# ---------------------------------------------------------------------------

_DIM_PATTERN = re.compile(r"dimensions\s+\[([^\]]+)\]")


def read_dimensions(content: str) -> list[float]:
    """Read the dimensions vector from field file content.

    Args:
        content: File body content (after header).

    Returns:
        7-element list of dimension exponents.

    Raises:
        ValueError: If dimensions not found or malformed.
    """
    match = _DIM_PATTERN.search(content)
    if match is None:
        raise ValueError("No 'dimensions' entry found")
    values = match.group(1).strip().split()
    if len(values) != 7:
        raise ValueError(
            f"Dimensions must have 7 values, got {len(values)}"
        )
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# Internal field parsing
# ---------------------------------------------------------------------------

_INTERNAL_UNIFORM_PATTERN = re.compile(
    r"internalField\s+uniform\s+(.+?)\s*;",
    re.DOTALL,
)

_INTERNAL_NONUNIFORM_PATTERN = re.compile(
    r"internalField\s+nonuniform\s+(?:(\w+)<(\w+)>\s+)?(\d+)",
    re.DOTALL,
)


def _parse_uniform_value(text: str, scalar_type: str) -> Any:
    """Parse a uniform value (scalar, vector, tensor).

    Args:
        text: Value text (e.g., ``"1.0"`` or ``"(1 0 0)"``).
        scalar_type: Type tag.

    Returns:
        Parsed value (float for scalar, tuple for vector/tensor).
    """
    text = text.strip()
    if scalar_type == "scalar":
        return float(text)
    elif scalar_type == "vector":
        # Parse (x y z)
        text = text.strip("()")
        return tuple(float(v) for v in text.split())
    elif scalar_type in ("symmTensor", "tensor"):
        # Parse (xx xy xz yy yz zz) or (xx xy xz yx yy yz zx zy zz)
        text = text.strip("()")
        return tuple(float(v) for v in text.split())
    return text


def _parse_vector_list(text: str, n: int) -> torch.Tensor:
    """Parse nonuniform vector list values from text.

    Each vector is ``(x y z)`` on its own line or separated by whitespace.

    Args:
        text: Content between ``(`` and ``)``.
        n: Expected number of vectors.

    Returns:
        Tensor of shape ``(n, 3)``.
    """
    text = text.strip()
    values = []
    # Match all (x y z) tuples
    for match in re.finditer(r"\(\s*([^)]+)\)", text):
        coords = match.group(1).split()
        values.append([float(c) for c in coords])
    if len(values) != n:
        raise ValueError(
            f"Expected {n} vectors, got {len(values)}"
        )
    return torch.tensor(values, dtype=get_default_dtype(), device=get_device())


def _parse_tensor_list(text: str, n: int, tensor_size: int) -> torch.Tensor:
    """Parse nonuniform tensor list values from text.

    Args:
        text: Content between ``(`` and ``)``.
        n: Expected number of tensors.
        tensor_size: Number of components per tensor (6 for symmTensor, 9 for tensor).

    Returns:
        Tensor of shape ``(n, tensor_size)``.
    """
    text = text.strip()
    values = []
    for match in re.finditer(r"\(\s*([^)]+)\)", text):
        components = match.group(1).split()
        values.append([float(c) for c in components])
    if len(values) != n:
        raise ValueError(
            f"Expected {n} tensors, got {len(values)}"
        )
    return torch.tensor(values, dtype=get_default_dtype(), device=get_device())


def _parse_scalar_list(text: str, n: int) -> torch.Tensor:
    """Parse nonuniform scalar list values from text.

    Args:
        text: Content between ``(`` and ``)``.
        n: Expected number of scalars.

    Returns:
        1-D tensor of length *n*.
    """
    text = text.strip().strip("()")
    values = [float(v) for v in text.split() if v.strip()]
    if len(values) != n:
        raise ValueError(
            f"Expected {n} scalars, got {len(values)}"
        )
    return torch.tensor(values, dtype=get_default_dtype(), device=get_device())


def _tensor_size_for_type(scalar_type: str) -> int:
    """Return the number of components for a given type tag."""
    _SIZES = {
        "scalar": 1,
        "vector": 3,
        "symmTensor": 6,
        "tensor": 9,
    }
    return _SIZES.get(scalar_type, 1)


def parse_internal_field(
    content: str,
    scalar_type: str = "scalar",
    *,
    is_binary: bool = False,
) -> tuple[Any, bool]:
    """Parse the internalField from file body content.

    Args:
        content: File body content (after header).
        scalar_type: Type tag.
        is_binary: If True, expect binary data encoding.

    Returns:
        Tuple of (field_values, is_uniform).
    """
    # Try uniform first
    match = _INTERNAL_UNIFORM_PATTERN.search(content)
    if match:
        value = _parse_uniform_value(match.group(1), scalar_type)
        return value, True

    # Try nonuniform
    match = _INTERNAL_NONUNIFORM_PATTERN.search(content)
    if match:
        n = int(match.group(3))
        # Find the data block between ( and )
        paren_start = content.find("(", match.end())
        if paren_start == -1:
            raise ValueError("Cannot find opening '(' for nonuniform data")

        if is_binary:
            # Binary: read raw bytes between ( and )
            paren_end = content.find(")", paren_start)
            if paren_end == -1:
                raise ValueError("Cannot find closing ')' for binary data")
            binary_data = content[paren_start + 1:paren_end].encode("latin-1")
            reader = BinaryReader(binary_data)
            tensor_size = _tensor_size_for_type(scalar_type)
            if tensor_size == 1:
                arr = reader.read_doubles(n)
                return torch.tensor(arr, dtype=get_default_dtype(), device=get_device()), False
            else:
                arr = reader.read_doubles(n * tensor_size)
                return torch.tensor(
                    arr.reshape(n, tensor_size), dtype=get_default_dtype(), device=get_device()
                ), False
        else:
            # ASCII: parse text between ( and )
            paren_end = _find_matching_paren(content, paren_start)
            data_text = content[paren_start + 1:paren_end]
            tensor_size = _tensor_size_for_type(scalar_type)
            if tensor_size == 1:
                return _parse_scalar_list(data_text, n), False
            elif tensor_size == 3:
                return _parse_vector_list(data_text, n), False
            else:
                return _parse_tensor_list(data_text, n, tensor_size), False

    raise ValueError("Cannot parse internalField")


def _find_matching_paren(text: str, start: int) -> int:
    """Find the matching closing parenthesis.

    Args:
        text: Full text.
        start: Position of opening ``(``.

    Returns:
        Position of matching ``)``.
    """
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unmatched parenthesis")


# ---------------------------------------------------------------------------
# Boundary field parsing
# ---------------------------------------------------------------------------

def parse_boundary_field(content: str) -> BoundaryField:
    """Parse the boundaryField section from file body content.

    Args:
        content: File body content (after header).

    Returns:
        :class:`BoundaryField` with all patches.
    """
    # Find boundaryField block
    match = re.search(r"boundaryField\s*\{", content)
    if match is None:
        return BoundaryField()

    # Find matching closing brace
    brace_start = match.end() - 1
    brace_end = _find_matching_brace(content, brace_start)
    block = content[brace_start + 1:brace_end]

    # Parse as dictionary
    try:
        d = parse_dict(block)
    except (ValueError, IndexError):
        return BoundaryField()

    patches = []
    for name, patch_dict in d.items():
        if isinstance(patch_dict, FoamDict):
            patch_type = patch_dict.get("type", "")
            value = patch_dict.get("value", None)
            n_faces = int(patch_dict.get("nFaces", 0))
            start_face = int(patch_dict.get("startFace", 0))
            patches.append(BoundaryPatch(
                name=name,
                patch_type=patch_type,
                value=value,
                n_faces=n_faces,
                start_face=start_face,
            ))
    return BoundaryField(patches)


def _find_matching_brace(text: str, start: int) -> int:
    """Find the matching closing brace.

    Args:
        text: Full text.
        start: Position of opening ``{``.

    Returns:
        Position of matching ``}``.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unmatched brace")


# ---------------------------------------------------------------------------
# Field reading / writing
# ---------------------------------------------------------------------------

# Detect type tag from internalField line or class name
_TYPE_FROM_CLASS = {
    "volScalarField": "scalar",
    "volVectorField": "vector",
    "volSymmTensorField": "symmTensor",
    "volTensorField": "tensor",
    "surfaceScalarField": "scalar",
    "surfaceVectorField": "vector",
    "pointScalarField": "scalar",
    "pointVectorField": "vector",
}

_TYPE_PATTERN = re.compile(r"nonuniform\s+(\w+)<(\w+)>")


def read_field(path: Union[str, Path]) -> FieldData:
    """Read an OpenFOAM field file.

    Args:
        path: Path to the field file.

    Returns:
        :class:`FieldData` with all parsed field information.
    """
    header, body = read_foam_file(path)

    # Determine scalar type
    scalar_type = _TYPE_FROM_CLASS.get(header.class_name, "scalar")
    type_match = _TYPE_PATTERN.search(body)
    if type_match:
        scalar_type = type_match.group(2)

    dimensions = read_dimensions(body)
    internal_field, is_uniform = parse_internal_field(
        body, scalar_type, is_binary=header.is_binary
    )
    boundary_field = parse_boundary_field(body)

    return FieldData(
        header=header,
        dimensions=dimensions,
        internal_field=internal_field,
        boundary_field=boundary_field,
        is_uniform=is_uniform,
        scalar_type=scalar_type,
    )


def _format_uniform_value(value: Any, scalar_type: str) -> str:
    """Format a uniform value for writing.

    Args:
        value: The uniform value.
        scalar_type: Type tag.

    Returns:
        Formatted string.
    """
    if scalar_type == "scalar":
        return str(value)
    elif scalar_type == "vector":
        if isinstance(value, (list, tuple)):
            return f"({' '.join(str(v) for v in value)})"
        return str(value)
    elif scalar_type in ("symmTensor", "tensor"):
        if isinstance(value, (list, tuple)):
            return f"({' '.join(str(v) for v in value)})"
        return str(value)
    return str(value)


def _format_dimensions(dims: list[float]) -> str:
    """Format dimensions for writing.

    Args:
        dims: 7-element dimension vector.

    Returns:
        Formatted string like ``[0 2 -2 0 0 0 0]``.
    """
    # Format integers without decimal point
    parts = []
    for d in dims:
        if d == int(d):
            parts.append(str(int(d)))
        else:
            parts.append(str(d))
    return f"[{' '.join(parts)}]"


def _format_boundary_patch(
    patch: BoundaryPatch,
    scalar_type: str,
    *,
    indent: str = "    ",
) -> str:
    """Format a boundary patch for writing.

    Args:
        patch: The patch to format.
        scalar_type: Type tag.
        indent: Indentation string.

    Returns:
        Formatted patch block.
    """
    lines = [f"{indent}{patch.name}", f"{indent}{{"]
    lines.append(f"{indent}    type        {patch.patch_type};")
    if patch.value is not None:
        if isinstance(patch.value, (int, float)):
            lines.append(f"{indent}    value       uniform {patch.value};")
        elif isinstance(patch.value, (list, tuple)):
            val_str = _format_uniform_value(patch.value, scalar_type)
            lines.append(f"{indent}    value       uniform {val_str};")
        elif isinstance(patch.value, str):
            lines.append(f"{indent}    value       {patch.value};")
        else:
            lines.append(f"{indent}    value       uniform {patch.value};")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def write_field(
    path: Union[str, Path],
    field_data: FieldData,
    *,
    overwrite: bool = False,
) -> None:
    """Write an OpenFOAM field file.

    Args:
        path: Output file path.
        field_data: Field data to write.
        overwrite: If False, raise if file exists.
    """
    body_parts: list[str] = []

    # Dimensions
    body_parts.append(f"dimensions      {_format_dimensions(field_data.dimensions)};\n")

    # Internal field
    if field_data.is_uniform:
        val_str = _format_uniform_value(field_data.internal_field, field_data.scalar_type)
        body_parts.append(f"internalField   uniform {val_str};\n")
    else:
        tensor = field_data.internal_field
        if isinstance(tensor, torch.Tensor):
            n = tensor.shape[0]
            tensor_size = _tensor_size_for_type(field_data.scalar_type)
            body_parts.append(
                f"internalField   nonuniform List<{field_data.scalar_type}> {n}"
            )
            body_parts.append("(")
            if tensor_size == 1:
                for val in tensor:
                    body_parts.append(f"{val.item()}")
            else:
                for row in tensor:
                    vals = " ".join(str(v) for v in row.tolist())
                    body_parts.append(f"({vals})")
            body_parts.append(");\n")
        else:
            body_parts.append(f"internalField   uniform {tensor};\n")

    # Boundary field
    if field_data.boundary_field and len(field_data.boundary_field) > 0:
        body_parts.append("boundaryField")
        body_parts.append("{")
        for patch in field_data.boundary_field:
            body_parts.append(_format_boundary_patch(patch, field_data.scalar_type))
        body_parts.append("}\n")

    body = "\n".join(body_parts)
    write_foam_file(path, field_data.header, body, overwrite=overwrite)
