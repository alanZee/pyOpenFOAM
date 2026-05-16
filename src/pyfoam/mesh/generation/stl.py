"""
STL file reader for surface mesh import.

Reads both ASCII and binary STL files and provides triangle mesh data
for use with snappyHexMesh and other surface-based operations.

Example::

    from pyfoam.mesh.generation.stl import STLReader

    reader = STLReader("surface.stl")
    triangles = reader.read()
    print(f"Read {len(triangles)} triangles")
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["STLReader", "STLSurface"]


class STLSurface:
    """Triangle surface mesh from STL file.

    Attributes:
        vertices: (n_vertices, 3) vertex coordinates.
        triangles: (n_triangles, 3) vertex indices per triangle.
        normals: (n_triangles, 3) face normals.
        name: Surface name (from solid name or filename).
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        normals: torch.Tensor,
        name: str = "surface",
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals
        self.name = name

    @property
    def n_vertices(self) -> int:
        """Number of unique vertices."""
        return self.vertices.shape[0]

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        return self.triangles.shape[0]

    def __repr__(self) -> str:
        return (
            f"STLSurface(name={self.name!r}, "
            f"n_vertices={self.n_vertices}, "
            f"n_triangles={self.n_triangles})"
        )


class STLReader:
    """Read STL files (ASCII and binary formats).

    Parameters:
        path: Path to the STL file.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)

    def read(self) -> STLSurface:
        """Read the STL file and return a surface mesh.

        Returns:
            STLSurface with vertices, triangles, and normals.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"STL file not found: {self._path}")

        with open(self._path, "rb") as f:
            header = f.read(80)

        # Check if binary or ASCII
        if self._is_ascii():
            return self._read_ascii()
        else:
            return self._read_binary()

    def _is_ascii(self) -> bool:
        """Check if the STL file is ASCII format."""
        try:
            with open(self._path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                return first_line.startswith("solid")
        except (UnicodeDecodeError, IOError):
            return False

    def _read_ascii(self) -> STLSurface:
        """Read an ASCII STL file."""
        device = get_device()
        dtype = get_default_dtype()

        vertices = []
        normals = []
        triangles = []
        vertex_map = {}
        current_normal = None

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split()

                if not parts:
                    continue

                if parts[0] == "solid":
                    name = " ".join(parts[1:]) if len(parts) > 1 else "surface"
                elif parts[0] == "facet" and parts[1] == "normal":
                    nx, ny, nz = float(parts[2]), float(parts[3]), float(parts[4])
                    current_normal = [nx, ny, nz]
                elif parts[0] == "vertex":
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    # Deduplicate vertices
                    key = (round(x, 10), round(y, 10), round(z, 10))
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append([x, y, z])
                elif parts[0] == "endfacet":
                    if current_normal is not None:
                        normals.append(current_normal)
                        current_normal = None

        # Build triangles from vertex sequences
        # Re-read to get triangle connectivity
        triangles = []
        current_tri = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if parts[0] == "vertex":
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    key = (round(x, 10), round(y, 10), round(z, 10))
                    current_tri.append(vertex_map[key])
                elif parts[0] == "endfacet":
                    if len(current_tri) == 3:
                        triangles.append(current_tri)
                    current_tri = []

        vertices_tensor = torch.tensor(vertices, dtype=dtype, device=device)
        triangles_tensor = torch.tensor(triangles, dtype=torch.int64, device=device)
        normals_tensor = torch.tensor(normals, dtype=dtype, device=device)

        name = self._path.stem
        return STLSurface(
            vertices=vertices_tensor,
            triangles=triangles_tensor,
            normals=normals_tensor,
            name=name,
        )

    def _read_binary(self) -> STLSurface:
        """Read a binary STL file."""
        device = get_device()
        dtype = get_default_dtype()

        with open(self._path, "rb") as f:
            # Header (80 bytes)
            header = f.read(80)

            # Number of triangles (4 bytes, little-endian uint32)
            n_triangles_raw = f.read(4)
            n_triangles = struct.unpack("<I", n_triangles_raw)[0]

            vertices = []
            normals = []
            vertex_map = {}

            for _ in range(n_triangles):
                # Normal (3 floats)
                nx, ny, nz = struct.unpack("<fff", f.read(12))
                normals.append([nx, ny, nz])

                # 3 vertices
                tri_indices = []
                for _ in range(3):
                    x, y, z = struct.unpack("<fff", f.read(12))
                    key = (round(x, 10), round(y, 10), round(z, 10))
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append([x, y, z])
                    tri_indices.append(vertex_map[key])

                # Attribute byte count (2 bytes, unused)
                f.read(2)

        # Build triangles
        triangles = []
        with open(self._path, "rb") as f:
            f.read(80)  # Skip header
            n_tri = struct.unpack("<I", f.read(4))[0]
            for _ in range(n_tri):
                f.read(12)  # Skip normal
                tri_indices = []
                for _ in range(3):
                    x, y, z = struct.unpack("<fff", f.read(12))
                    key = (round(x, 10), round(y, 10), round(z, 10))
                    tri_indices.append(vertex_map[key])
                triangles.append(tri_indices)
                f.read(2)  # Skip attribute

        vertices_tensor = torch.tensor(vertices, dtype=dtype, device=device)
        triangles_tensor = torch.tensor(triangles, dtype=torch.int64, device=device)
        normals_tensor = torch.tensor(normals, dtype=dtype, device=device)

        name = self._path.stem
        return STLSurface(
            vertices=vertices_tensor,
            triangles=triangles_tensor,
            normals=normals_tensor,
            name=name,
        )


def write_stl_ascii(
    path: Union[str, Path],
    surface: STLSurface,
    *,
    overwrite: bool = False,
) -> None:
    """Write a surface mesh to ASCII STL format.

    Args:
        path: Output file path.
        surface: STLSurface to write.
        overwrite: If False, raise if file exists.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    vertices = surface.vertices.cpu().numpy()
    triangles = surface.triangles.cpu().numpy()
    normals = surface.normals.cpu().numpy()

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"solid {surface.name}\n")
        for i in range(surface.n_triangles):
            nx, ny, nz = normals[i]
            f.write(f"  facet normal {nx} {ny} {nz}\n")
            f.write("    outer loop\n")
            for j in range(3):
                vidx = triangles[i, j]
                x, y, z = vertices[vidx]
                f.write(f"      vertex {x} {y} {z}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {surface.name}\n")
