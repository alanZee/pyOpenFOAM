"""
Parallel I/O — processor directory structure for parallel CFD.

OpenFOAM parallel cases use the directory layout::

    case/
    ├── processor0/
    │   ├── constant/polyMesh/
    │   │   ├── points
    │   │   ├── faces
    │   │   ├── owner
    │   │   ├── neighbour
    │   │   └── boundary
    │   └── 0/
    │       ├── p
    │       ├── U
    │       └── ...
    ├── processor1/
    │   └── ...
    └── ...

This module writes subdomain meshes and fields into processor directories.

Usage::

    from pyfoam.parallel.parallel_io import ParallelWriter, ParallelReader

    writer = ParallelWriter(case_dir, n_processors)
    writer.write_mesh(subdomains)
    writer.write_field("p", field_values, subdomains, time=0)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.decomposition import SubDomain

__all__ = ["ParallelWriter", "ParallelReader"]


# ---------------------------------------------------------------------------
# ParallelWriter
# ---------------------------------------------------------------------------


class ParallelWriter:
    """Write parallel case data to processor directories.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory.
    n_processors : int
        Number of processors.
    """

    def __init__(self, case_dir: str | Path, n_processors: int) -> None:
        self._case_dir = Path(case_dir)
        self._n_processors = n_processors
        self._device = get_device()
        self._dtype = get_default_dtype()

    # ------------------------------------------------------------------
    # Directory structure
    # ------------------------------------------------------------------

    def create_processor_dirs(self) -> list[Path]:
        """Create processor directory structure.

        Returns:
            List of processor directory paths.
        """
        dirs: list[Path] = []
        for i in range(self._n_processors):
            proc_dir = self._case_dir / f"processor{i}"
            mesh_dir = proc_dir / "constant" / "polyMesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            dirs.append(proc_dir)
        return dirs

    # ------------------------------------------------------------------
    # Mesh writing
    # ------------------------------------------------------------------

    def write_mesh(self, subdomains: list[SubDomain]) -> None:
        """Write subdomain meshes to processor directories.

        Args:
            subdomains: List of :class:`SubDomain` objects.
        """
        for sub in subdomains:
            proc_dir = self._case_dir / f"processor{sub.processor_id}"
            mesh_dir = proc_dir / "constant" / "polyMesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)

            mesh = sub.mesh

            self._write_points(mesh_dir, mesh.points)
            self._write_faces(mesh_dir, mesh.faces)
            self._write_owner(mesh_dir, mesh.owner)
            self._write_neighbour(mesh_dir, mesh.neighbour)
            self._write_boundary(mesh_dir, mesh.boundary)

    def _write_points(self, mesh_dir: Path, points: torch.Tensor) -> None:
        """Write points file."""
        n_points = points.shape[0]
        with open(mesh_dir / "points", "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       vectorField;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      points;\n")
            f.write("}\n\n")
            f.write(f"{n_points}\n(\n")
            pts_np = points.cpu().numpy()
            for i in range(n_points):
                f.write(f"({pts_np[i, 0]:.10g} {pts_np[i, 1]:.10g} {pts_np[i, 2]:.10g})\n")
            f.write(")\n")

    def _write_faces(self, mesh_dir: Path, faces: list[torch.Tensor]) -> None:
        """Write faces file."""
        n_faces = len(faces)
        with open(mesh_dir / "faces", "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       faceList;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      faces;\n")
            f.write("}\n\n")
            f.write(f"{n_faces}\n(\n")
            for face in faces:
                indices = face.cpu().numpy().tolist()
                f.write(f"({' '.join(str(i) for i in indices)})\n")
            f.write(")\n")

    def _write_owner(self, mesh_dir: Path, owner: torch.Tensor) -> None:
        """Write owner file."""
        n = owner.shape[0]
        with open(mesh_dir / "owner", "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      owner;\n")
            f.write("}\n\n")
            f.write(f"{n}\n(\n")
            own_np = owner.cpu().numpy()
            for i in range(n):
                f.write(f"{own_np[i]}\n")
            f.write(")\n")

    def _write_neighbour(self, mesh_dir: Path, neighbour: torch.Tensor) -> None:
        """Write neighbour file."""
        n = neighbour.shape[0]
        with open(mesh_dir / "neighbour", "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      neighbour;\n")
            f.write("}\n\n")
            f.write(f"{n}\n(\n")
            nbr_np = neighbour.cpu().numpy()
            for i in range(n):
                f.write(f"{nbr_np[i]}\n")
            f.write(")\n")

    def _write_boundary(self, mesh_dir: Path, boundary: list[dict]) -> None:
        """Write boundary file."""
        n_patches = len(boundary)
        with open(mesh_dir / "boundary", "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       polyBoundaryMesh;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      boundary;\n")
            f.write("}\n\n")
            f.write(f"{n_patches}\n(\n")
            for patch in boundary:
                f.write(f"    {patch['name']}\n")
                f.write("    {\n")
                f.write(f"        type {patch.get('type', 'patch')};\n")
                f.write(f"        nFaces {patch['nFaces']};\n")
                f.write(f"        startFace {patch['startFace']};\n")
                f.write("    }\n")
            f.write(")\n")

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def write_field(
        self,
        field_name: str,
        field_values: torch.Tensor,
        subdomains: list[SubDomain],
        time: int | float = 0,
        dimensions: str = "0 0 0 0 0 0 0",
    ) -> None:
        """Write field values to processor directories.

        Args:
            field_name: Name of the field (e.g. ``"p"``).
            field_values: Global field tensor ``(n_global_cells,)``.
            subdomains: List of subdomains with cell mapping.
            time: Time step directory name.
            dimensions: Dimension string (OpenFOAM format).
        """
        for sub in subdomains:
            proc_dir = self._case_dir / f"processor{sub.processor_id}"
            time_dir = proc_dir / str(time)
            time_dir.mkdir(parents=True, exist_ok=True)

            # Extract local values using global cell mapping
            local_values = field_values[sub.global_cell_ids[: sub.n_owned_cells]]

            filepath = time_dir / field_name
            with open(filepath, "w") as f:
                f.write("FoamFile\n{\n")
                f.write("    version     2.0;\n")
                f.write("    format      ascii;\n")
                f.write("    class       volScalarField;\n")
                f.write(f"    location    \"{time}\";\n")
                f.write(f"    object      {field_name};\n")
                f.write("}\n\n")
                f.write(f"dimensions [{dimensions}];\n\n")
                f.write(f"internalField nonuniform {sub.n_owned_cells}\n(\n")
                vals_np = local_values.cpu().numpy()
                for i in range(sub.n_owned_cells):
                    f.write(f"{vals_np[i]:.10g}\n")
                f.write(");\n\n")
                f.write("boundaryField\n{\n")
                for patch in sub.mesh.boundary:
                    f.write(f"    {patch['name']}\n")
                    f.write("    {\n")
                    f.write(f"        type {patch.get('type', 'patch')};\n")
                    f.write("    }\n")
                f.write("}\n")


# ---------------------------------------------------------------------------
# ParallelReader
# ---------------------------------------------------------------------------


class ParallelReader:
    """Read parallel case data from processor directories.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory.
    """

    def __init__(self, case_dir: str | Path) -> None:
        self._case_dir = Path(case_dir)
        self._device = get_device()
        self._dtype = get_default_dtype()

    def read_processor_mesh(self, processor_id: int) -> dict:
        """Read mesh data from a processor directory.

        Args:
            processor_id: Processor index.

        Returns:
            Dict with keys: ``points``, ``faces``, ``owner``, ``neighbour``, ``boundary``.
        """
        mesh_dir = self._case_dir / f"processor{processor_id}" / "constant" / "polyMesh"

        return {
            "points": self._read_points(mesh_dir / "points"),
            "faces": self._read_faces(mesh_dir / "faces"),
            "owner": self._read_label_list(mesh_dir / "owner"),
            "neighbour": self._read_label_list(mesh_dir / "neighbour"),
            "boundary": self._read_boundary(mesh_dir / "boundary"),
        }

    def read_field(
        self,
        processor_id: int,
        field_name: str,
        time: int | float = 0,
    ) -> torch.Tensor:
        """Read internal field values from a processor directory.

        Args:
            processor_id: Processor index.
            field_name: Name of the field.
            time: Time step.

        Returns:
            Field values tensor.
        """
        filepath = (
            self._case_dir / f"processor{processor_id}" / str(time) / field_name
        )
        return self._read_internal_field(filepath)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_points(self, filepath: Path) -> torch.Tensor:
        """Read points file."""
        with open(filepath) as f:
            lines = f.readlines()

        # Find the start of data (after the closing brace of FoamFile)
        data_start = 0
        brace_count = 0
        for i, line in enumerate(lines):
            if "{" in line:
                brace_count += 1
            if "}" in line:
                brace_count -= 1
                if brace_count == 0:
                    data_start = i + 1
                    break

        # Skip blank lines
        while data_start < len(lines) and lines[data_start].strip() == "":
            data_start += 1

        n_points = int(lines[data_start].strip())
        points = []
        for i in range(data_start + 2, data_start + 2 + n_points):
            line = lines[i].strip().strip("()")
            parts = line.split()
            points.append([float(p) for p in parts])

        return torch.tensor(points, dtype=self._dtype, device=self._device)

    def _read_faces(self, filepath: Path) -> list[torch.Tensor]:
        """Read faces file."""
        with open(filepath) as f:
            lines = f.readlines()

        data_start = 0
        brace_count = 0
        for i, line in enumerate(lines):
            if "{" in line:
                brace_count += 1
            if "}" in line:
                brace_count -= 1
                if brace_count == 0:
                    data_start = i + 1
                    break

        while data_start < len(lines) and lines[data_start].strip() == "":
            data_start += 1

        n_faces = int(lines[data_start].strip())
        faces = []
        for i in range(data_start + 2, data_start + 2 + n_faces):
            line = lines[i].strip().strip("()")
            parts = line.split()
            faces.append(torch.tensor([int(p) for p in parts], dtype=INDEX_DTYPE, device=self._device))

        return faces

    def _read_label_list(self, filepath: Path) -> torch.Tensor:
        """Read labelList (owner/neighbour)."""
        with open(filepath) as f:
            lines = f.readlines()

        data_start = 0
        brace_count = 0
        for i, line in enumerate(lines):
            if "{" in line:
                brace_count += 1
            if "}" in line:
                brace_count -= 1
                if brace_count == 0:
                    data_start = i + 1
                    break

        while data_start < len(lines) and lines[data_start].strip() == "":
            data_start += 1

        n = int(lines[data_start].strip())
        values = []
        for i in range(data_start + 2, data_start + 2 + n):
            values.append(int(lines[i].strip()))

        return torch.tensor(values, dtype=INDEX_DTYPE, device=self._device)

    def _read_boundary(self, filepath: Path) -> list[dict]:
        """Read boundary file."""
        with open(filepath) as f:
            lines = f.readlines()

        data_start = 0
        brace_count = 0
        for i, line in enumerate(lines):
            if "{" in line:
                brace_count += 1
            if "}" in line:
                brace_count -= 1
                if brace_count == 0:
                    data_start = i + 1
                    break

        while data_start < len(lines) and lines[data_start].strip() == "":
            data_start += 1

        n_patches = int(lines[data_start].strip())
        patches = []
        idx = data_start + 2
        for _ in range(n_patches):
            name = lines[idx].strip()
            idx += 1
            idx += 1  # skip {
            type_line = lines[idx].strip()
            ptype = type_line.split()[1].rstrip(";")
            idx += 1
            nfaces_line = lines[idx].strip()
            n_faces = int(nfaces_line.split()[1].rstrip(";"))
            idx += 1
            startface_line = lines[idx].strip()
            start_face = int(startface_line.split()[1].rstrip(";"))
            idx += 1
            idx += 1  # skip }
            patches.append({
                "name": name,
                "type": ptype,
                "nFaces": n_faces,
                "startFace": start_face,
            })

        return patches

    def _read_internal_field(self, filepath: Path) -> torch.Tensor:
        """Read internal field values."""
        with open(filepath) as f:
            content = f.read()

        # Find "internalField" section
        marker = "internalField"
        idx = content.find(marker)
        if idx == -1:
            raise ValueError(f"No internalField found in {filepath}")

        # Parse "nonuniform N" or "uniform value"
        rest = content[idx + len(marker):].strip()
        if rest.startswith("uniform"):
            val = float(rest.split()[1].rstrip(";"))
            return torch.tensor([val], dtype=self._dtype, device=self._device)

        # nonuniform N(...)
        n_start = rest.find("nonuniform") + len("nonuniform")
        n_end = rest.find("\n", n_start)
        n = int(rest[n_start:n_end].strip())

        paren_start = rest.find("(") + 1
        paren_end = rest.find(")")
        values_str = rest[paren_start:paren_end]
        values = [float(v) for v in values_str.split() if v]

        return torch.tensor(values, dtype=self._dtype, device=self._device)
