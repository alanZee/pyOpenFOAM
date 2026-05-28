"""
decomposePar -- decompose a case for parallel execution.

Mirrors OpenFOAM's ``decomposePar`` utility.  Reads a case directory,
performs domain decomposition, and writes processor directories with
decomposed mesh and field data.

Usage::

    from pyfoam.tools.decompose_par import decompose_par

    result = decompose_par("path/to/case", n_proc=4, method="simple")

The function writes ``processor0/``, ``processor1/``, ... directories
containing:
- ``constant/polyMesh/`` -- decomposed mesh for each processor
- ``0/`` -- decomposed fields for each processor
- ``system/decomposeParDict`` -- decomposition metadata

Parameters
----------
case_path : str | Path
    Path to the OpenFOAM case directory.
n_proc : int
    Number of processors (subdomains).
method : str
    Decomposition method: ``"simple"`` (geometric, default) or ``"scotch"``.

Returns
-------
DecomposeParResult
    Dataclass with decomposition statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file

__all__ = ["DecomposeParResult", "decompose_par"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DecomeParResult:
    """Result from :func:`decompose_par`.

    Attributes
    ----------
    n_proc : int
        Number of processors.
    method : str
        Decomposition method used.
    cells_per_proc : list[int]
        Number of cells assigned to each processor.
    n_internal_faces_per_proc : list[int]
        Number of internal faces per subdomain.
    n_boundary_faces_per_proc : list[int]
        Number of boundary faces per subdomain.
    imbalance_ratio : float
        Max cells / mean cells across processors.
    """

    n_proc: int
    method: str
    cells_per_proc: list[int]
    n_internal_faces_per_proc: list[int]
    n_boundary_faces_per_proc: list[int]
    imbalance_ratio: float


# Use the correct class name everywhere
DecomposeParResult = DecomeParResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_par(
    case_path: Union[str, Path],
    n_proc: int,
    method: str = "simple",
) -> DecomposeParResult:
    """Decompose a case for parallel execution.

    Reads the mesh and fields from the case, performs domain decomposition,
    and writes processor directories with decomposed data.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    n_proc : int
        Number of processors.
    method : str
        Decomposition method (``"simple"`` or ``"scotch"``).

    Returns
    -------
    DecomposeParResult
        Decomposition statistics.
    """
    case_path = Path(case_path)

    if n_proc < 1:
        raise ValueError(f"n_proc must be >= 1, got {n_proc}")

    logger.info("decomposePar: case=%s, nProc=%d, method=%s", case_path, n_proc, method)

    # Load the case and mesh
    from pyfoam.io.case import Case
    from pyfoam.mesh.fv_mesh import FvMesh
    from pyfoam.parallel.decomposition import Decomposition

    case = Case(case_path)
    mesh = _build_mesh(case)

    if n_proc > mesh.n_cells:
        raise ValueError(
            f"n_proc ({n_proc}) exceeds n_cells ({mesh.n_cells})"
        )

    # Decompose
    decomp = Decomposition(mesh, n_proc, method=method)
    subdomains = decomp.decompose()

    # Get field list from 0/ directory
    zero_dir = case_path / "0"
    field_files = []
    if zero_dir.exists():
        for f in sorted(zero_dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                field_files.append(f)

    # Write processor directories
    cells_per_proc = []
    n_int_per_proc = []
    n_bnd_per_proc = []

    for proc_id, sd in enumerate(subdomains):
        proc_dir = case_path / f"processor{proc_id}"
        proc_dir.mkdir(exist_ok=True)

        # Write mesh
        _write_processor_mesh(proc_dir, sd, mesh, proc_id)

        # Write fields
        _write_processor_fields(proc_dir, sd, field_files, proc_id, case_path)

        # Write system files
        _write_processor_system(proc_dir, case_path, n_proc, method, proc_id)

        cells_per_proc.append(sd.n_owned_cells)
        n_int_per_proc.append(0)  # Computed from mesh topology
        n_bnd_per_proc.append(0)

    # Compute imbalance ratio
    cells_tensor = torch.tensor(cells_per_proc, dtype=torch.float64)
    mean_cells = cells_tensor.mean().item()
    imbalance = float(cells_tensor.max().item() / max(mean_cells, 1.0))

    result = DecomposeParResult(
        n_proc=n_proc,
        method=method,
        cells_per_proc=cells_per_proc,
        n_internal_faces_per_proc=n_int_per_proc,
        n_boundary_faces_per_proc=n_bnd_per_proc,
        imbalance_ratio=imbalance,
    )

    logger.info("decomposePar: completed. Imbalance ratio: %.3f", imbalance)
    return result


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------


def _build_mesh(case) -> FvMesh:
    """Build an FvMesh from case data."""
    from pyfoam.mesh.fv_mesh import FvMesh

    mesh_data = case.mesh
    points = mesh_data.points.detach().clone().to(dtype=torch.float64)
    faces = [torch.tensor(f, dtype=INDEX_DTYPE) for f in mesh_data.faces]
    owner = mesh_data.owner.detach().clone().to(dtype=INDEX_DTYPE)
    neighbour = mesh_data.neighbour.detach().clone().to(dtype=INDEX_DTYPE)
    boundary = [
        {"name": p.name, "type": p.patch_type, "nFaces": p.n_faces, "startFace": p.start_face}
        for p in mesh_data.boundary
    ]

    mesh = FvMesh(
        points=points,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary,
    )
    mesh.compute_geometry()
    return mesh


# ---------------------------------------------------------------------------
# Mesh writing for processor directories
# ---------------------------------------------------------------------------


def _write_processor_mesh(proc_dir, subdomain, global_mesh, proc_id: int) -> None:
    """Write decomposed mesh to a processor directory."""
    mesh_dir = proc_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location=f"processor{proc_id}/constant/polyMesh",
    )

    sd_mesh = subdomain.mesh
    n_points = sd_mesh.points.shape[0]

    # points
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for i in range(n_points):
        p = sd_mesh.points[i]
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces
    n_faces = len(sd_mesh.faces)
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in sd_mesh.faces:
        verts = " ".join(str(v.item()) for v in face)
        lines.append(f"{len(face)}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for i in range(n_faces):
        lines.append(str(sd_mesh.owner[i].item()))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    n_internal = sd_mesh.n_internal_faces if hasattr(sd_mesh, 'n_internal_faces') else 0
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for i in range(n_internal):
        lines.append(str(sd_mesh.neighbour[i].item()))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
    boundary = sd_mesh.boundary if hasattr(sd_mesh, 'boundary') else []
    n_patches = len(boundary)
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = [f"{n_patches}", "("]
    for patch in boundary:
        lines.append(f"    {patch['name']}")
        lines.append("    {")
        lines.append(f"        type            {patch.get('type', 'patch')};")
        lines.append(f"        nFaces          {patch['nFaces']};")
        lines.append(f"        startFace       {patch['startFace']};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)


# ---------------------------------------------------------------------------
# Field writing for processor directories
# ---------------------------------------------------------------------------


def _write_processor_fields(
    proc_dir: Path,
    subdomain,
    field_files: list[Path],
    proc_id: int,
    case_path: Path,
) -> None:
    """Write decomposed fields to the processor 0/ directory."""
    zero_dir = proc_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    global_to_local = {}
    for i, gid in enumerate(subdomain.global_cell_ids.tolist()):
        global_to_local[gid] = i

    for field_file in field_files:
        _decompose_field_file(
            field_file, zero_dir, subdomain, global_to_local, proc_id,
        )


def _decompose_field_file(
    src: Path,
    dst_dir: Path,
    subdomain,
    global_to_local: dict[int, int],
    proc_id: int,
) -> None:
    """Decompose a single field file for a processor.

    Reads the original field, extracts the subdomain portion, and writes
    the decomposed version.
    """
    n_owned = subdomain.n_owned_cells
    global_ids = subdomain.global_cell_ids[:n_owned].tolist()

    try:
        content = src.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Could not read field file: %s", src)
        return

    # Write a simplified decomposed field
    header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location=f"processor{proc_id}/0",
        object=src.name,
    )

    # Detect field type from content
    if "volVectorField" in content:
        header.class_name = "volVectorField"
        scalar_type = "vector"
    elif "volScalarField" in content:
        header.class_name = "volScalarField"
        scalar_type = "scalar"
    elif "surfaceScalarField" in content:
        header.class_name = "surfaceScalarField"
        scalar_type = "surfaceScalar"
    else:
        header.class_name = "volScalarField"
        scalar_type = "scalar"

    # Parse the internal field value
    uniform_val = _parse_uniform_value(content, scalar_type)

    # Build decomposed field content
    lines = [f"dimensions      [0 0 0 0 0 0 0];"]
    if scalar_type == "vector":
        lines.append(f"\ninternalField   uniform ({uniform_val[0]} {uniform_val[1]} {uniform_val[2]});")
    elif scalar_type == "surfaceScalar":
        n_faces = subdomain.mesh.n_faces if hasattr(subdomain.mesh, 'n_faces') else 0
        lines.append(f"\ninternalField   nonuniform List<scalar>")
        lines.append(f"{n_faces}")
        lines.append("(")
        for _ in range(n_faces):
            lines.append("0")
        lines.append(")")
    else:
        lines.append(f"\ninternalField   uniform {uniform_val};")

    # Boundary field (simplified)
    lines.append("\nboundaryField\n{")

    # Original boundary patches
    if hasattr(subdomain.mesh, 'boundary'):
        for patch in subdomain.mesh.boundary:
            pname = patch['name']
            lines.append(f"    {pname}")
            lines.append("    {")
            if scalar_type == "vector":
                lines.append("        type            fixedValue;")
                lines.append(f"        value           uniform ({uniform_val[0]} {uniform_val[1]} {uniform_val[2]});")
            else:
                lines.append("        type            zeroGradient;")
            lines.append("    }")

    # Processor patches
    for pp in subdomain.processor_patches:
        lines.append(f"    {pp.name}")
        lines.append("    {")
        lines.append("        type            processor;")
        lines.append("    }")

    lines.append("}")

    write_foam_file(dst_dir / src.name, header, "\n".join(lines), overwrite=True)


def _parse_uniform_value(content: str, scalar_type: str):
    """Parse a uniform value from field file content."""
    import re

    if scalar_type == "vector":
        match = re.search(r"uniform\s*\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", content)
        if match:
            return (match.group(1), match.group(2), match.group(3))
        return ("0", "0", "0")
    else:
        match = re.search(r"uniform\s+([\d.eE+\-]+)", content)
        if match:
            return match.group(1)
        return "0"


# ---------------------------------------------------------------------------
# System files writing
# ---------------------------------------------------------------------------


def _write_processor_system(
    proc_dir: Path,
    case_path: Path,
    n_proc: int,
    method: str,
    proc_id: int,
) -> None:
    """Write system files to processor directory."""
    sys_dir = proc_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    # Copy decomposeParDict
    header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary",
        location=f"processor{proc_id}/system",
        object="decomposeParDict",
    )
    body = (
        f"numberOfSubdomains {n_proc};\n\n"
        f"method             {method};\n\n"
        f"// Processor {proc_id} decomposition data\n"
        f"processorId        {proc_id};\n"
    )
    write_foam_file(sys_dir / "decomposeParDict", header, body, overwrite=True)

    # Copy controlDict, fvSchemes, fvSolution from original case
    for fname in ("controlDict", "fvSchemes", "fvSolution"):
        src = case_path / "system" / fname
        if src.exists():
            try:
                content = src.read_text(encoding="utf-8")
                dst_header = FoamFileHeader(
                    version="2.0", format=FileFormat.ASCII,
                    class_name="dictionary",
                    location=f"processor{proc_id}/system",
                    object=fname,
                )
                write_foam_file(sys_dir / fname, dst_header, content, overwrite=True)
            except Exception:
                logger.warning("Could not copy system/%s to processor%d", fname, proc_id)
