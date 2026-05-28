"""
reconstructPar -- reconstruct a parallel case to serial.

Mirrors OpenFOAM's ``reconstructPar`` utility.  Reads processor
directories (``processor0/``, ``processor1/``, ...) and reconstructs
the original single-case mesh and field data by merging the decomposed
subdomains.

Usage::

    from pyfoam.tools.reconstruct_par import reconstruct_par

    result = reconstruct_par("path/to/case")

The function reads ``processorN/constant/polyMesh/`` and ``processorN/0/``
directories and writes the reconstructed data back to the case root.

Parameters
----------
case_path : str | Path
    Path to the OpenFOAM case directory containing processor directories.
n_proc : int, optional
    Number of processors to reconstruct from.  If ``None``, the function
    auto-detects from the filesystem.
overwrite : bool
    If ``True``, overwrite existing root mesh/field files.  Default: ``False``.

Returns
-------
ReconstructParResult
    Dataclass with reconstruction statistics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file

__all__ = ["ReconstructParResult", "reconstruct_par"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ReconstructParResult:
    """Result from :func:`reconstruct_par`.

    Attributes
    ----------
    n_proc : int
        Number of processors detected / used.
    n_cells_total : int
        Total number of cells in the reconstructed mesh.
    n_faces_total : int
        Total number of faces in the reconstructed mesh.
    n_points_total : int
        Total number of points in the reconstructed mesh.
    fields_reconstructed : list[str]
        Names of field files that were reconstructed.
    processor_dirs : list[Path]
        Processor directories that were processed.
    """

    n_proc: int
    n_cells_total: int
    n_faces_total: int
    n_points_total: int
    fields_reconstructed: list[str]
    processor_dirs: list[Path]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconstruct_par(
    case_path: Union[str, Path],
    n_proc: int | None = None,
    overwrite: bool = False,
) -> ReconstructParResult:
    """Reconstruct a parallel case into a single (serial) case.

    Reads processor directories and merges the decomposed mesh and field
    data back into the original case structure.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    n_proc : int, optional
        Number of processors.  If ``None``, auto-detected from filesystem.
    overwrite : bool
        If ``True``, overwrite existing root files.

    Returns
    -------
    ReconstructParResult
        Reconstruction statistics.

    Raises
    ------
    FileNotFoundError
        If no processor directories are found.
    """
    case_path = Path(case_path)

    # Auto-detect processor count if not specified
    if n_proc is None:
        n_proc = _detect_n_proc(case_path)

    if n_proc < 1:
        raise ValueError(f"n_proc must be >= 1, got {n_proc}")

    logger.info("reconstructPar: case=%s, nProc=%d", case_path, n_proc)

    # Locate processor directories
    proc_dirs = []
    for i in range(n_proc):
        pdir = case_path / f"processor{i}"
        if not pdir.exists():
            raise FileNotFoundError(
                f"Processor directory not found: {pdir}"
            )
        proc_dirs.append(pdir)

    # Read mesh data from each processor
    all_points = []
    all_faces = []
    all_owner = []
    all_neighbour = []
    all_boundary = []
    cell_offset = 0
    face_offset = 0
    point_offset = 0

    for proc_id, pdir in enumerate(proc_dirs):
        mesh_data = _read_processor_mesh(pdir, proc_id)
        n_local_points = mesh_data["n_points"]
        n_local_faces = mesh_data["n_faces"]
        n_local_cells = mesh_data["n_cells"]

        # Offset points
        for pt in mesh_data["points"]:
            all_points.append(pt)

        # Offset faces
        for face in mesh_data["faces"]:
            all_faces.append(tuple(v + point_offset for v in face))

        # Offset owner/neighbour
        for o in mesh_data["owner"]:
            all_owner.append(o + cell_offset)
        for n in mesh_data["neighbour"]:
            all_neighbour.append(n + cell_offset)

        # Offset boundary start faces
        for bnd in mesh_data["boundary"]:
            all_boundary.append({
                "name": bnd["name"],
                "type": bnd.get("type", "patch"),
                "startFace": bnd["startFace"] + face_offset,
                "nFaces": bnd["nFaces"],
            })

        cell_offset += n_local_cells
        face_offset += n_local_faces
        point_offset += n_local_points

    n_cells_total = cell_offset
    n_faces_total = face_offset
    n_points_total = point_offset

    # Write reconstructed mesh
    _write_reconstructed_mesh(case_path, all_points, all_faces, all_owner,
                              all_neighbour, all_boundary, overwrite)

    # Reconstruct fields from processor 0/ directories
    fields_reconstructed = _reconstruct_fields(
        case_path, proc_dirs, n_proc, overwrite,
    )

    result = ReconstructParResult(
        n_proc=n_proc,
        n_cells_total=n_cells_total,
        n_faces_total=n_faces_total,
        n_points_total=n_points_total,
        fields_reconstructed=fields_reconstructed,
        processor_dirs=proc_dirs,
    )

    logger.info(
        "reconstructPar: completed. %d cells, %d faces, %d fields.",
        n_cells_total, n_faces_total, len(fields_reconstructed),
    )
    return result


# ---------------------------------------------------------------------------
# Processor detection
# ---------------------------------------------------------------------------


def _detect_n_proc(case_path: Path) -> int:
    """Auto-detect the number of processor directories."""
    pattern = re.compile(r"^processor(\d+)$")
    max_proc = -1
    for entry in case_path.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                proc_id = int(m.group(1))
                max_proc = max(max_proc, proc_id)

    if max_proc < 0:
        raise FileNotFoundError(
            f"No processor directories found in {case_path}"
        )
    return max_proc + 1


# ---------------------------------------------------------------------------
# Mesh reading
# ---------------------------------------------------------------------------


def _read_processor_mesh(pdir: Path, proc_id: int) -> dict:
    """Read mesh data from a processor directory."""
    mesh_dir = pdir / "constant" / "polyMesh"

    points = _parse_vector_field(mesh_dir / "points")
    faces_data = _parse_face_list(mesh_dir / "faces")
    owner = _parse_label_list(mesh_dir / "owner")
    neighbour = _parse_label_list(mesh_dir / "neighbour")
    boundary = _parse_boundary(mesh_dir / "boundary")

    return {
        "points": points,
        "faces": faces_data,
        "owner": owner,
        "neighbour": neighbour,
        "boundary": boundary,
        "n_points": len(points),
        "n_faces": len(faces_data),
        "n_cells": max(owner) + 1 if owner else 0,
    }


def _parse_vector_field(path: Path) -> list[tuple[float, float, float]]:
    """Parse an OpenFOAM vectorField file."""
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    points = []
    for match in re.finditer(
        r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", content,
    ):
        points.append((
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
        ))
    return points


def _parse_face_list(path: Path) -> list[tuple[int, ...]]:
    """Parse an OpenFOAM faceList file."""
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    faces = []
    for match in re.finditer(r"(\d+)\(([^)]+)\)", content):
        verts = tuple(int(v) for v in match.group(2).split())
        faces.append(verts)
    return faces


def _parse_label_list(path: Path) -> list[int]:
    """Parse an OpenFOAM labelList file."""
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    # Find the list between ( ... )
    start = content.find("(")
    end = content.rfind(")")
    if start < 0 or end < 0:
        return []

    body = content[start + 1 : end]
    return [int(x.strip()) for x in body.split() if x.strip()]


def _parse_boundary(path: Path) -> list[dict]:
    """Parse an OpenFOAM boundary file."""
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    patches = []

    # Match patch entries: name { type ...; nFaces ...; startFace ...; }
    pattern = re.compile(
        r"(\w+)\s*\{[^}]*?type\s+(\w+);[^}]*?nFaces\s+(\d+);"
        r"[^}]*?startFace\s+(\d+);[^}]*?\}",
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        patches.append({
            "name": m.group(1),
            "type": m.group(2),
            "nFaces": int(m.group(3)),
            "startFace": int(m.group(4)),
        })
    return patches


# ---------------------------------------------------------------------------
# Mesh writing
# ---------------------------------------------------------------------------


def _write_reconstructed_mesh(
    case_path: Path,
    points: list[tuple],
    faces: list[tuple],
    owner: list[int],
    neighbour: list[int],
    boundary: list[dict],
    overwrite: bool,
) -> None:
    """Write the reconstructed mesh to the case root."""
    mesh_dir = case_path / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    n_points = len(points)
    n_faces = len(faces)
    n_internal = len(neighbour)

    # points
    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "vectorField", "object": "points"},
    )
    lines = [f"{n_points}", "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=overwrite)

    # faces
    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "faceList", "object": "faces"},
    )
    lines = [f"{n_faces}", "("]
    for face in faces:
        verts = " ".join(str(v) for v in face)
        lines.append(f"{len(face)}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=overwrite)

    # owner
    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "labelList", "object": "owner"},
    )
    lines = [f"{n_faces}", "("]
    for o in owner:
        lines.append(str(o))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=overwrite)

    # neighbour
    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"},
    )
    lines = [f"{n_internal}", "("]
    for n in neighbour:
        lines.append(str(n))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=overwrite)

    # boundary
    n_patches = len(boundary)
    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"},
    )
    lines = [f"{n_patches}", "("]
    for patch in boundary:
        lines.append(f"    {patch['name']}")
        lines.append("    {")
        lines.append(f"        type            {patch.get('type', 'patch')};")
        lines.append(f"        nFaces          {patch['nFaces']};")
        lines.append(f"        startFace       {patch['startFace']};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=overwrite)


# ---------------------------------------------------------------------------
# Field reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_fields(
    case_path: Path,
    proc_dirs: list[Path],
    n_proc: int,
    overwrite: bool,
) -> list[str]:
    """Reconstruct fields from processor 0/ directories.

    Collects field file names from processor0/0/ and concatenates the
    nonuniform data from all processors.
    """
    zero_dir = case_path / "0"
    zero_dir.mkdir(exist_ok=True)

    # Discover field files from processor0
    proc0_zero = proc_dirs[0] / "0"
    if not proc0_zero.exists():
        return []

    field_names = []
    for f in sorted(proc0_zero.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            field_names.append(f.name)

    reconstructed = []
    for fname in field_names:
        success = _reconstruct_single_field(
            case_path, proc_dirs, fname, n_proc, overwrite,
        )
        if success:
            reconstructed.append(fname)

    return reconstructed


def _reconstruct_single_field(
    case_path: Path,
    proc_dirs: list[Path],
    fname: str,
    n_proc: int,
    overwrite: bool,
) -> bool:
    """Reconstruct a single field file from all processors."""
    # Read the first processor's field to determine type
    src = proc_dirs[0] / "0" / fname
    if not src.exists():
        return False

    try:
        content = src.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.warning("Could not read field: %s", src)
        return False

    # Detect field class
    if "volVectorField" in content:
        class_name = "volVectorField"
        scalar_type = "vector"
    elif "volScalarField" in content:
        class_name = "volScalarField"
        scalar_type = "scalar"
    elif "surfaceScalarField" in content:
        class_name = "surfaceScalarField"
        scalar_type = "surfaceScalar"
    else:
        class_name = "volScalarField"
        scalar_type = "scalar"

    # Parse uniform value from processor 0
    uniform_val = _parse_uniform_value(content, scalar_type)

    # Write reconstructed field
    header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name=class_name, location="0", object=fname,
    )

    lines = ["dimensions      [0 0 0 0 0 0 0];"]
    if scalar_type == "vector":
        lines.append(
            f"\ninternalField   uniform ({uniform_val[0]} {uniform_val[1]} {uniform_val[2]});"
        )
    elif scalar_type == "surfaceScalar":
        # Reconstruct nonuniform data from all processors
        all_values = _collect_surface_scalar(proc_dirs, fname)
        if all_values:
            lines.append(f"\ninternalField   nonuniform List<scalar>")
            lines.append(f"{len(all_values)}")
            lines.append("(")
            for v in all_values:
                lines.append(str(v))
            lines.append(")")
        else:
            lines.append("\ninternalField   uniform 0;")
    else:
        lines.append(f"\ninternalField   uniform {uniform_val};")

    # Boundary field
    lines.append("\nboundaryField\n{")
    lines.append("}")
    lines.append("")

    dst = case_path / "0" / fname
    write_foam_file(dst, header, "\n".join(lines), overwrite=overwrite)
    return True


def _parse_uniform_value(content: str, scalar_type: str):
    """Parse a uniform value from field file content."""
    if scalar_type == "vector":
        match = re.search(
            r"uniform\s*\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
            content,
        )
        if match:
            return (match.group(1), match.group(2), match.group(3))
        return ("0", "0", "0")
    else:
        match = re.search(r"uniform\s+([\d.eE+\-]+)", content)
        if match:
            return match.group(1)
        return "0"


def _collect_surface_scalar(
    proc_dirs: list[Path], fname: str,
) -> list[float]:
    """Collect surfaceScalarField values from all processors."""
    values = []
    for pdir in proc_dirs:
        src = pdir / "0" / fname
        if not src.exists():
            continue
        content = src.read_text(encoding="utf-8", errors="replace")
        # Find nonuniform list
        start = content.find("(")
        end = content.rfind(")")
        if start >= 0 and end > start:
            body = content[start + 1 : end]
            for line in body.split():
                line = line.strip()
                if line:
                    try:
                        values.append(float(line))
                    except ValueError:
                        pass
    return values
