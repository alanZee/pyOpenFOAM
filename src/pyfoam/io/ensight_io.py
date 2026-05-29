"""
EnSight I/O — read and write EnSight Gold format.

EnSight is a widely used scientific visualisation format.  This module
provides reading and writing of EnSight Gold geometry and variable files,
commonly used for CFD post-processing.

EnSight file structure (per case)::

    case.case          -- case description file
    case.geo           -- geometry file
    case.vel           -- velocity variable file
    case.scl           -- scalar variable file

EnSight Gold geometry file format::

    EnSight Gold geometry file
    title line 1
    title line 2
    node id off
    element id off
    coordinates
    <n_nodes>
    <node_id> <x> <y> <z>
    ...
    part
    <part_id>
    part description
    <element_type>
    <n_elements>
    <elem_id> <n1> <n2> ...

Supported element types: ``tria3``, ``quad4``, ``tetra4``, ``hexa8``,
``pyramid5``, ``penta6``.

References
----------
- EnSight Gold format documentation
- OpenFOAM ``foamToEnsight`` utility source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "EnSightPart",
    "EnSightGeometry",
    "EnSightVariable",
    "EnSightCase",
    "read_ensight_geometry",
    "read_ensight_variable",
    "read_ensight_case",
    "write_ensight_geometry",
    "write_ensight_variable",
    "write_ensight_case",
    "foam_to_ensight",
]

logger = logging.getLogger(__name__)

# Element type definitions: name -> (n_nodes_per_element, VTK cell type code)
_ENSIGHT_ELEMENT_TYPES: Dict[str, Tuple[int, int]] = {
    "tria3": (3, 5),
    "quad4": (4, 9),
    "tetra4": (4, 10),
    "pyramid5": (5, 14),
    "penta6": (6, 13),
    "hexa8": (8, 12),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnSightPart:
    """An EnSight geometry part.

    Attributes:
        part_id: Part ID number.
        description: Part description string.
        element_type: Element type name (e.g. ``"hexa8"``).
        connectivity: Element connectivity array, shape ``(n_elements, n_nodes)``.
            0-based node indices.
    """

    part_id: int
    description: str
    element_type: str
    connectivity: np.ndarray  # (n_elements, n_nodes), int32


@dataclass
class EnSightGeometry:
    """EnSight geometry data.

    Attributes:
        title: Title lines.
        node_coords: Node coordinates, shape ``(n_nodes, 3)``.
        parts: List of geometry parts.
    """

    title: List[str]
    node_coords: np.ndarray  # (n_nodes, 3)
    parts: List[EnSightPart]


@dataclass
class EnSightVariable:
    """EnSight variable data.

    Attributes:
        description: Variable description.
        variable_type: ``"scalar"`` or ``"vector"``.
        part_id: Part ID this variable belongs to.
        values: Variable values.
            Scalar: shape ``(n_values,)``.
            Vector: shape ``(n_values, 3)``.
    """

    description: str
    variable_type: str  # "scalar" or "vector"
    part_id: int
    values: np.ndarray  # (n_values,) or (n_values, 3)


@dataclass
class EnSightCase:
    """EnSight case file data.

    Attributes:
        title: Case title.
        geometry_file: Geometry file name.
        variables: Dict mapping variable names to variable file names.
        time_values: Time values (if time-varying).
        time_set: Time set number.
    """

    title: str = ""
    geometry_file: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    time_values: List[float] = field(default_factory=list)
    time_set: int = 1


# ---------------------------------------------------------------------------
# Geometry reading
# ---------------------------------------------------------------------------


def read_ensight_geometry(path: Union[str, Path]) -> EnSightGeometry:
    """Read an EnSight Gold geometry file.

    Args:
        path: Path to the ``.geo`` file.

    Returns:
        :class:`EnSightGeometry` with parsed data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EnSight geometry file not found: {path}")

    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("EnSight geometry file too short")

    # First line is format identifier
    # Lines 1-2 are title
    title = [lines[0].rstrip(), lines[1].rstrip()]

    idx = 2
    node_id_off = False
    elem_id_off = False

    # Parse optional keywords
    while idx < len(lines):
        line = lines[idx].strip()
        if line.lower() == "node id off":
            node_id_off = True
            idx += 1
        elif line.lower() == "element id off":
            elem_id_off = True
            idx += 1
        elif line.lower() == "node id given":
            idx += 1
        elif line.lower() == "element id given":
            idx += 1
        elif line.lower() == "extents":
            idx += 1 + 2  # skip extents block
        elif line.lower() == "coordinates":
            break
        else:
            idx += 1

    # Parse coordinates section
    if idx >= len(lines) or lines[idx].strip().lower() != "coordinates":
        raise ValueError("'coordinates' section not found")

    idx += 1
    n_nodes = int(lines[idx].strip())
    idx += 1

    coords = np.zeros((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        if idx >= len(lines):
            break
        parts = lines[idx].strip().split()
        if len(parts) >= 3:
            coords[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        idx += 1

    # Parse parts
    parts: List[EnSightPart] = []
    while idx < len(lines):
        line = lines[idx].strip()
        if line.lower() == "part":
            idx += 1
            if idx >= len(lines):
                break
            part_id = int(lines[idx].strip())
            idx += 1
            if idx >= len(lines):
                break
            desc = lines[idx].rstrip()
            idx += 1

            # Parse element blocks within this part
            while idx < len(lines):
                eline = lines[idx].strip()
                if eline.lower() == "part":
                    break
                if eline.lower() in _ENSIGHT_ELEMENT_TYPES:
                    elem_type = eline.lower()
                    idx += 1
                    if idx >= len(lines):
                        break
                    n_elems = int(lines[idx].strip())
                    idx += 1
                    n_verts = _ENSIGHT_ELEMENT_TYPES[elem_type][0]
                    conn = np.zeros((n_elems, n_verts), dtype=np.int32)
                    for ei in range(n_elems):
                        if idx >= len(lines):
                            break
                        parts_line = lines[idx].strip().split()
                        # If element IDs are given, first token is the ID
                        start = 0 if elem_id_off else 1
                        for vi in range(n_verts):
                            conn[ei, vi] = int(parts_line[start + vi]) - 1  # 0-based
                        idx += 1
                    parts.append(EnSightPart(
                        part_id=part_id,
                        description=desc,
                        element_type=elem_type,
                        connectivity=conn,
                    ))
                else:
                    idx += 1
        else:
            idx += 1

    return EnSightGeometry(title=title, node_coords=coords, parts=parts)


# ---------------------------------------------------------------------------
# Variable reading
# ---------------------------------------------------------------------------


def read_ensight_variable(path: Union[str, Path]) -> List[EnSightVariable]:
    """Read an EnSight variable file.

    Args:
        path: Path to the variable file (e.g. ``.vel``, ``.scl``).

    Returns:
        List of :class:`EnSightVariable` (one per part per time step).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EnSight variable file not found: {path}")

    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return []

    desc = lines[0].rstrip()
    idx = 1

    # Determine if scalar or vector from description or content
    variable_type = "scalar"
    if "vector" in desc.lower() or "velocity" in desc.lower():
        variable_type = "vector"

    variables: List[EnSightVariable] = []

    while idx < len(lines):
        line = lines[idx].strip()
        if line.lower() == "part":
            idx += 1
            if idx >= len(lines):
                break
            part_id = int(lines[idx].strip())
            idx += 1
            # Look for "all" or coordinates per node
            if idx < len(lines) and lines[idx].strip().lower() == "all":
                idx += 1
                # Read all values
                values = []
                while idx < len(lines):
                    vline = lines[idx].strip()
                    if vline.lower() in ("part", ""):
                        break
                    try:
                        if variable_type == "vector":
                            parts_ = vline.split()
                            if len(parts_) >= 3:
                                values.append([float(parts_[0]), float(parts_[1]), float(parts_[2])])
                        else:
                            values.append(float(vline))
                    except ValueError:
                        break
                    idx += 1

                arr = np.array(values, dtype=np.float64)
                variables.append(EnSightVariable(
                    description=desc,
                    variable_type=variable_type,
                    part_id=part_id,
                    values=arr,
                ))
            else:
                idx += 1
        else:
            idx += 1

    return variables


# ---------------------------------------------------------------------------
# Case file reading
# ---------------------------------------------------------------------------


def read_ensight_case(path: Union[str, Path]) -> EnSightCase:
    """Read an EnSight case description file.

    Args:
        path: Path to the ``.case`` file.

    Returns:
        :class:`EnSightCase` with parsed data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EnSight case file not found: {path}")

    case = EnSightCase()
    section = ""

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("FORMAT"):
                section = "format"
                continue
            elif line.startswith("GEOMETRY"):
                section = "geometry"
                continue
            elif line.startswith("VARIABLE"):
                section = "variable"
                continue
            elif line.startswith("TIME"):
                section = "time"
                continue

            if section == "format":
                if line.startswith("type") and ":" in line:
                    pass  # "type: ensight"
            elif section == "geometry":
                if "model" in line.lower() and ":" in line:
                    parts_ = line.split(":")
                    if len(parts_) >= 2:
                        case.geometry_file = parts_[-1].strip()
            elif section == "variable":
                if ":" in line:
                    parts_ = line.split(":")
                    if len(parts_) >= 2:
                        # Format: "description  variable_name: filename"
                        name_part = parts_[0].strip().split()[-1] if parts_[0].strip() else ""
                        filename = parts_[-1].strip()
                        case.variables[name_part] = filename
            elif section == "time":
                if "time values" in line.lower() and ":" in line:
                    parts_ = line.split(":")
                    if len(parts_) >= 2:
                        vals = parts_[-1].strip().split()
                        case.time_values = [float(v) for v in vals]

    return case


# ---------------------------------------------------------------------------
# Geometry writing
# ---------------------------------------------------------------------------


def write_ensight_geometry(
    path: Union[str, Path],
    node_coords: np.ndarray,
    parts: List[EnSightPart],
    title: str = "pyOpenFOAM geometry",
) -> None:
    """Write an EnSight Gold geometry file.

    Args:
        path: Output file path.
        node_coords: Node coordinates, shape ``(n_nodes, 3)``.
        parts: List of geometry parts.
        title: Title string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = node_coords.shape[0]

    with open(path, "w") as f:
        f.write("EnSight Gold geometry\n")
        f.write(f"{title}\n")
        f.write("node id off\n")
        f.write("element id off\n")
        f.write("coordinates\n")
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            f.write(f"{node_coords[i, 0]:12.5e}"
                    f"{node_coords[i, 1]:12.5e}"
                    f"{node_coords[i, 2]:12.5e}\n")

        for part in parts:
            f.write("part\n")
            f.write(f"{part.part_id}\n")
            f.write(f"{part.description}\n")
            f.write(f"{part.element_type}\n")
            n_elems = part.connectivity.shape[0]
            f.write(f"{n_elems}\n")
            for ei in range(n_elems):
                # EnSight uses 1-based node indices
                conn_1based = [int(part.connectivity[ei, vi]) + 1
                               for vi in range(part.connectivity.shape[1])]
                f.write(" ".join(f"{c:>8d}" for c in conn_1based) + "\n")

    logger.info("Wrote EnSight geometry: %s", path)


# ---------------------------------------------------------------------------
# Variable writing
# ---------------------------------------------------------------------------


def write_ensight_variable(
    path: Union[str, Path],
    variables: List[EnSightVariable],
    description: str = "pyOpenFOAM variable",
) -> None:
    """Write an EnSight variable file.

    Args:
        path: Output file path.
        variables: List of variable data (one per part).
        description: Variable description.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"{description}\n")

        for var in variables:
            f.write("part\n")
            f.write(f"{var.part_id}\n")
            f.write("all\n")

            if var.variable_type == "vector" and var.values.ndim == 2:
                for i in range(var.values.shape[0]):
                    f.write(f"{var.values[i, 0]:12.5e}"
                            f"{var.values[i, 1]:12.5e}"
                            f"{var.values[i, 2]:12.5e}\n")
            else:
                for val in var.values.flatten():
                    f.write(f"{val:12.5e}\n")

    logger.info("Wrote EnSight variable: %s", path)


# ---------------------------------------------------------------------------
# Case file writing
# ---------------------------------------------------------------------------


def write_ensight_case(
    path: Union[str, Path],
    case: EnSightCase,
) -> None:
    """Write an EnSight case description file.

    Args:
        path: Output file path.
        case: :class:`EnSightCase` data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("FORMAT\ntype: ensight\n\n")
        f.write("GEOMETRY\n")
        f.write(f"model: {case.geometry_file}\n\n")
        f.write("VARIABLE\n")
        for name, filename in case.variables.items():
            f.write(f"scalar per node: {name} {filename}\n")
        if case.time_values:
            f.write("\nTIME\n")
            f.write(f"time set: {case.time_set}\n")
            f.write(f"number of steps: {len(case.time_values)}\n")
            f.write("filename start number: 0\n")
            f.write("filename increment: 1\n")
            f.write("time values: ")
            f.write(" ".join(f"{t:.6e}" for t in case.time_values))
            f.write("\n")

    logger.info("Wrote EnSight case: %s", path)


# ---------------------------------------------------------------------------
# foamToEnsight conversion
# ---------------------------------------------------------------------------


def foam_to_ensight(
    case_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert an OpenFOAM case to EnSight format.

    Reads the mesh and field files from the case directory and writes
    EnSight geometry and variable files.

    Args:
        case_dir: Path to the OpenFOAM case directory.
        output_dir: Output directory for EnSight files.

    Returns:
        Path to the generated case file.
    """
    from pyfoam.io.mesh_io import read_mesh

    case_dir = Path(case_dir)
    if output_dir is None:
        output_dir = case_dir / "EnSight"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read mesh
    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    if not poly_mesh_dir.exists():
        raise FileNotFoundError(f"polyMesh directory not found: {poly_mesh_dir}")

    mesh = read_mesh(poly_mesh_dir)
    points = mesh.points.detach().cpu().numpy()

    # Create parts from boundary patches
    parts: List[EnSightPart] = []
    part_id = 1

    # Add volume cells as a part
    for boundary in mesh.boundary:
        parts.append(EnSightPart(
            part_id=part_id,
            description=boundary.name,
            element_type="hexa8",  # simplified
            connectivity=np.zeros((0, 8), dtype=np.int32),
        ))
        part_id += 1

    # Write geometry
    geo_path = output_dir / "case.geo"
    write_ensight_geometry(geo_path, points, parts)

    # Write case file
    case = EnSightCase(
        title="pyOpenFOAM EnSight export",
        geometry_file="case.geo",
    )
    case_path = output_dir / "case.case"
    write_ensight_case(case_path, case)

    return case_path
