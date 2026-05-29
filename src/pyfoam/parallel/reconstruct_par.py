"""
ReconstructPar — reconstruct a parallel case into a single case.

Combines processor directories (``processor0/``, ``processor1/``, ...) into
a single unified case directory, merging mesh and field data from all
subdomains.  This mirrors OpenFOAM's ``reconstructPar`` utility.

Workflow:

1. Discover processor directories and time steps
2. Read each processor's mesh and reconstruct a global mesh
3. For each time step, merge internal fields from all processors
4. Write the reconstructed case

Usage::

    recon = ReconstructPar(case_dir)
    recon.discover()
    recon.reconstruct_mesh()
    recon.reconstruct_fields(time=0)

References
----------
- OpenFOAM ``reconstructPar`` utility source
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE

__all__ = ["ReconstructPar", "ReconstructResult"]

logger = logging.getLogger(__name__)


@dataclass
class ReconstructResult:
    """Result of a reconstruction operation.

    Attributes:
        n_processors: Number of processor directories found.
        n_time_steps: Number of time steps reconstructed.
        n_global_cells: Total number of cells in the reconstructed mesh.
        time_steps: List of reconstructed time step names.
    """

    n_processors: int = 0
    n_time_steps: int = 0
    n_global_cells: int = 0
    time_steps: List[str] = field(default_factory=list)


class ReconstructPar:
    """Reconstruct a parallel case into a single case directory.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory containing ``processorN/`` subdirectories.
    """

    def __init__(self, case_dir: str | Path) -> None:
        self._case_dir = Path(case_dir)
        self._device = get_device()
        self._dtype = get_default_dtype()
        self._n_processors: int = 0
        self._processor_dirs: List[Path] = []
        self._time_steps: List[str] = []

    @property
    def case_dir(self) -> Path:
        """Root case directory."""
        return self._case_dir

    @property
    def n_processors(self) -> int:
        """Number of processor directories discovered."""
        return self._n_processors

    @property
    def time_steps(self) -> List[str]:
        """Time steps discovered."""
        return self._time_steps

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> int:
        """Discover processor directories and common time steps.

        Scans for ``processorN/`` directories (N = 0, 1, ...) and finds
        time steps present in all processors.

        Returns:
            Number of processors found.

        Raises:
            FileNotFoundError: If no processor directories exist.
        """
        proc_dirs = sorted(self._case_dir.glob("processor[0-9]*"))
        proc_dirs = [d for d in proc_dirs if d.is_dir()]

        if not proc_dirs:
            raise FileNotFoundError(
                f"No processor directories found in {self._case_dir}"
            )

        self._processor_dirs = proc_dirs
        self._n_processors = len(proc_dirs)

        # Find time steps common to all processors
        all_times: Optional[set] = None
        for pd in proc_dirs:
            proc_times = set()
            for entry in pd.iterdir():
                if entry.is_dir():
                    try:
                        float(entry.name)
                        proc_times.add(entry.name)
                    except ValueError:
                        continue
            if all_times is None:
                all_times = proc_times
            else:
                all_times &= proc_times

        self._time_steps = sorted(all_times or [], key=lambda x: float(x))
        logger.info(
            "Discovered %d processors, %d time steps",
            self._n_processors,
            len(self._time_steps),
        )
        return self._n_processors

    # ------------------------------------------------------------------
    # Mesh reconstruction
    # ------------------------------------------------------------------

    def reconstruct_mesh(self) -> Dict[str, Any]:
        """Reconstruct the global mesh from processor subdomain meshes.

        Reads points, faces, owner, neighbour, and boundary from each
        processor and merges them into a global mesh.

        Returns:
            Dictionary with keys: ``points``, ``faces``, ``owner``,
            ``neighbour``, ``boundary``, ``n_cells``.
        """
        if not self._processor_dirs:
            self.discover()

        all_points: List[torch.Tensor] = []
        all_faces: List[List[int]] = []
        all_owner: List[int] = []
        all_neighbour: List[int] = []
        all_boundary: List[Dict[str, Any]] = []

        point_offset = 0
        face_offset = 0
        cell_offset = 0

        for i, proc_dir in enumerate(self._processor_dirs):
            mesh_dir = proc_dir / "constant" / "polyMesh"
            if not mesh_dir.exists():
                logger.warning("No mesh in processor%d", i)
                continue

            # Read processor mesh
            proc_points = self._read_points_ascii(mesh_dir / "points")
            proc_faces = self._read_faces_ascii(mesh_dir / "faces")
            proc_owner = self._read_label_list_ascii(mesh_dir / "owner")
            proc_neighbour = self._read_label_list_ascii(mesh_dir / "neighbour")
            proc_boundary = self._read_boundary_ascii(mesh_dir / "boundary")

            n_proc_cells = int(proc_owner.max().item()) + 1 if proc_owner.numel() > 0 else 0

            # Accumulate points
            all_points.append(proc_points)

            # Remap face vertex indices
            for face_verts in proc_faces:
                remapped = [v + point_offset for v in face_verts]
                all_faces.append(remapped)

            # Remap owner/neighbour cell indices
            for o in proc_owner.tolist():
                all_owner.append(o + cell_offset)
            for n in proc_neighbour.tolist():
                all_neighbour.append(n + cell_offset)

            # Remap boundary patches
            for patch in proc_boundary:
                all_boundary.append({
                    "name": patch["name"],
                    "type": patch.get("type", "patch"),
                    "startFace": patch["startFace"] + face_offset,
                    "nFaces": patch["nFaces"],
                })

            point_offset += proc_points.shape[0]
            face_offset += len(proc_faces)
            cell_offset += n_proc_cells

        # Concatenate points
        if all_points:
            merged_points = torch.cat(all_points, dim=0)
        else:
            merged_points = torch.zeros(0, 3, dtype=self._dtype, device=self._device)

        result = {
            "points": merged_points,
            "faces": all_faces,
            "owner": torch.tensor(all_owner, dtype=INDEX_DTYPE, device=self._device),
            "neighbour": torch.tensor(all_neighbour, dtype=INDEX_DTYPE, device=self._device),
            "boundary": all_boundary,
            "n_cells": cell_offset,
        }

        logger.info(
            "Reconstructed mesh: %d cells, %d faces, %d points",
            cell_offset, len(all_faces), merged_points.shape[0],
        )
        return result

    # ------------------------------------------------------------------
    # Field reconstruction
    # ------------------------------------------------------------------

    def reconstruct_fields(
        self,
        time: str | int | float = "0",
        field_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct fields for a given time step.

        Reads the internal field from each processor directory and
        concatenates them into a global field.

        Args:
            time: Time step (directory name).
            field_names: List of field names to reconstruct.
                If None, auto-detect from processor0.

        Returns:
            Dictionary mapping field names to global field tensors.
        """
        if not self._processor_dirs:
            self.discover()

        time_str = str(time)

        # Auto-detect fields from processor0
        if field_names is None:
            proc0_time = self._processor_dirs[0] / time_str
            if not proc0_time.exists():
                raise FileNotFoundError(
                    f"Time directory {time_str} not found in processor0"
                )
            field_names = []
            for entry in proc0_time.iterdir():
                if entry.is_file() and not entry.name.startswith("."):
                    field_names.append(entry.name)

        result: Dict[str, torch.Tensor] = {}

        for fname in field_names:
            proc_fields: List[torch.Tensor] = []
            for proc_dir in self._processor_dirs:
                filepath = proc_dir / time_str / fname
                if not filepath.exists():
                    logger.warning(
                        "Field %s not found in %s/processor at time %s",
                        fname, proc_dir.name, time_str,
                    )
                    continue
                field_data = self._read_internal_field_ascii(filepath)
                proc_fields.append(field_data)

            if proc_fields:
                result[fname] = torch.cat(proc_fields, dim=0)
                logger.info(
                    "Reconstructed field '%s' at t=%s: %d values",
                    fname, time_str, result[fname].shape[0],
                )

        return result

    def reconstruct_case(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
    ) -> ReconstructResult:
        """Reconstruct the full case (mesh + all time steps).

        Args:
            output_dir: Output directory. If None, writes to case_dir.
            field_names: Fields to reconstruct. If None, auto-detect.

        Returns:
            :class:`ReconstructResult` with reconstruction statistics.
        """
        if not self._processor_dirs:
            self.discover()

        out = Path(output_dir) if output_dir else self._case_dir

        # Reconstruct mesh
        mesh = self.reconstruct_mesh()

        # Write mesh to constant/polyMesh
        mesh_dir = out / "constant" / "polyMesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        self._write_points_ascii(mesh_dir / "points", mesh["points"])
        self._write_faces_ascii(mesh_dir / "faces", mesh["faces"])
        self._write_label_list_ascii(mesh_dir / "owner", mesh["owner"])
        self._write_label_list_ascii(mesh_dir / "neighbour", mesh["neighbour"])
        self._write_boundary_ascii(mesh_dir / "boundary", mesh["boundary"])

        # Reconstruct each time step
        for ts in self._time_steps:
            fields = self.reconstruct_fields(time=ts, field_names=field_names)
            time_dir = out / ts
            time_dir.mkdir(parents=True, exist_ok=True)

            for fname, fdata in fields.items():
                filepath = time_dir / fname
                self._write_field_ascii(filepath, fname, fdata, ts)

        return ReconstructResult(
            n_processors=self._n_processors,
            n_time_steps=len(self._time_steps),
            n_global_cells=mesh["n_cells"],
            time_steps=list(self._time_steps),
        )

    # ------------------------------------------------------------------
    # ASCII readers (processor directory format)
    # ------------------------------------------------------------------

    def _find_data_start(self, lines: List[str]) -> int:
        """Find the start of data after FoamFile header."""
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
        return data_start

    def _read_points_ascii(self, filepath: Path) -> torch.Tensor:
        """Read points file (ASCII)."""
        with open(filepath) as f:
            lines = f.readlines()
        ds = self._find_data_start(lines)
        n = int(lines[ds].strip())
        points = []
        for i in range(ds + 2, ds + 2 + n):
            line = lines[i].strip().strip("()")
            parts = line.split()
            points.append([float(p) for p in parts])
        return torch.tensor(points, dtype=self._dtype, device=self._device)

    def _read_faces_ascii(self, filepath: Path) -> List[List[int]]:
        """Read faces file (ASCII)."""
        with open(filepath) as f:
            lines = f.readlines()
        ds = self._find_data_start(lines)
        n = int(lines[ds].strip())
        faces = []
        for i in range(ds + 2, ds + 2 + n):
            line = lines[i].strip().strip("()")
            parts = line.split()
            faces.append([int(p) for p in parts])
        return faces

    def _read_label_list_ascii(self, filepath: Path) -> torch.Tensor:
        """Read labelList (owner/neighbour) file (ASCII)."""
        with open(filepath) as f:
            lines = f.readlines()
        ds = self._find_data_start(lines)
        n = int(lines[ds].strip())
        values = []
        for i in range(ds + 2, ds + 2 + n):
            values.append(int(lines[i].strip()))
        return torch.tensor(values, dtype=INDEX_DTYPE, device=self._device)

    def _read_boundary_ascii(self, filepath: Path) -> List[Dict[str, Any]]:
        """Read boundary file (ASCII)."""
        with open(filepath) as f:
            lines = f.readlines()
        ds = self._find_data_start(lines)
        n_patches = int(lines[ds].strip())
        patches: List[Dict[str, Any]] = []
        idx = ds + 2
        for _ in range(n_patches):
            name = lines[idx].strip()
            idx += 1
            idx += 1  # skip {
            type_line = lines[idx].strip()
            ptype = type_line.split()[1].rstrip(";")
            idx += 1
            n_faces = int(lines[idx].strip().split()[1].rstrip(";"))
            idx += 1
            start_face = int(lines[idx].strip().split()[1].rstrip(";"))
            idx += 1
            idx += 1  # skip }
            patches.append({
                "name": name,
                "type": ptype,
                "nFaces": n_faces,
                "startFace": start_face,
            })
        return patches

    def _read_internal_field_ascii(self, filepath: Path) -> torch.Tensor:
        """Read internal field values from a file."""
        with open(filepath) as f:
            content = f.read()

        marker = "internalField"
        idx = content.find(marker)
        if idx == -1:
            raise ValueError(f"No internalField in {filepath}")

        rest = content[idx + len(marker):].strip()
        if rest.startswith("uniform"):
            val = float(rest.split()[1].rstrip(";"))
            # Need to know the number of cells; parse from the file
            # For uniform fields, return a single value
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

    # ------------------------------------------------------------------
    # ASCII writers
    # ------------------------------------------------------------------

    def _write_header(self, f, class_name: str, obj: str, location: str) -> None:
        """Write FoamFile header."""
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write(f"    class       {class_name};\n")
        f.write(f'    location    "{location}";\n')
        f.write(f"    object      {obj};\n")
        f.write("}\n\n")

    def _write_points_ascii(self, filepath: Path, points: torch.Tensor) -> None:
        n = points.shape[0]
        with open(filepath, "w") as f:
            self._write_header(f, "vectorField", "points", "constant/polyMesh")
            f.write(f"{n}\n(\n")
            pts_np = points.cpu().numpy()
            for i in range(n):
                f.write(f"({pts_np[i, 0]:.10g} {pts_np[i, 1]:.10g} {pts_np[i, 2]:.10g})\n")
            f.write(")\n")

    def _write_faces_ascii(self, filepath: Path, faces: List[List[int]]) -> None:
        n = len(faces)
        with open(filepath, "w") as f:
            self._write_header(f, "faceList", "faces", "constant/polyMesh")
            f.write(f"{n}\n(\n")
            for face in faces:
                f.write(f"({' '.join(str(i) for i in face)})\n")
            f.write(")\n")

    def _write_label_list_ascii(self, filepath: Path, values: torch.Tensor) -> None:
        n = values.shape[0]
        with open(filepath, "w") as f:
            self._write_header(f, "labelList", filepath.stem, "constant/polyMesh")
            f.write(f"{n}\n(\n")
            vals_np = values.cpu().numpy()
            for i in range(n):
                f.write(f"{int(vals_np[i])}\n")
            f.write(")\n")

    def _write_boundary_ascii(self, filepath: Path, boundary: List[Dict[str, Any]]) -> None:
        with open(filepath, "w") as f:
            self._write_header(f, "polyBoundaryMesh", "boundary", "constant/polyMesh")
            f.write(f"{len(boundary)}\n(\n")
            for patch in boundary:
                f.write(f"    {patch['name']}\n")
                f.write("    {\n")
                f.write(f"        type {patch.get('type', 'patch')};\n")
                f.write(f"        nFaces {patch['nFaces']};\n")
                f.write(f"        startFace {patch['startFace']};\n")
                f.write("    }\n")
            f.write(")\n")

    def _write_field_ascii(
        self,
        filepath: Path,
        field_name: str,
        data: torch.Tensor,
        location: str,
    ) -> None:
        n = data.shape[0]
        with open(filepath, "w") as f:
            self._write_header(f, "volScalarField", field_name, location)
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(f"internalField nonuniform {n}\n(\n")
            vals_np = data.cpu().numpy()
            for i in range(n):
                f.write(f"{vals_np[i]:.10g}\n")
            f.write(");\n\n")
            f.write("boundaryField\n{\n")
            f.write("}\n")
