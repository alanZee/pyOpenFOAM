"""
ExtrudeToRegion — extrude faces into a separate mesh region.

Mirrors OpenFOAM's ``extrudeToRegion`` utility.  Extrudes selected
boundary faces of a mesh into a new separate mesh region, creating
a multi-region mesh suitable for conjugate heat transfer or similar
coupled simulations.

Physics
-------
For each selected boundary face, the extrusion creates a column of
prismatic cells extending from the boundary into the new region.
The new region has its own cell zone and can be assigned independent
material properties.

The extrusion follows the boundary face normal::

    x_new = x_face + t * n_face

where ``t`` is the cumulative layer thickness and ``n_face`` is the
outward-pointing face normal.

References
----------
- OpenFOAM ``extrudeToRegion`` utility source
- OpenFOAM multi-region mesh handling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh

__all__ = [
    "ExtrudeToRegion",
    "RegionExtrudeSpec",
]


@dataclass
class RegionExtrudeSpec:
    """Specification for region extrusion.

    Attributes:
        source_patch: Name of the source boundary patch to extrude.
        region_name: Name of the new region.
        n_layers: Number of extrusion layers.
        total_thickness: Total extrusion distance.
        expansion_ratio: Layer-to-layer expansion ratio (default: 1.0 = uniform).
    """

    source_patch: str
    region_name: str = "region1"
    n_layers: int = 5
    total_thickness: float = 0.1
    expansion_ratio: float = 1.0

    def layer_thicknesses(self) -> list[float]:
        """Compute per-layer thicknesses with optional expansion."""
        n = self.n_layers
        r = self.expansion_ratio

        if abs(r - 1.0) < 1e-10:
            t0 = self.total_thickness / n
            return [t0] * n

        t0 = self.total_thickness * (1.0 - r) / (1.0 - r ** n)
        return [t0 * r ** i for i in range(n)]


class ExtrudeToRegion:
    """Extrude boundary faces into a separate mesh region.

    Takes an existing mesh and extrudes specified boundary patches into
    new prismatic cells forming a separate region.

    Parameters:
        mesh: Source mesh.
        specs: List of extrusion specifications (one per patch/region).
    """

    def __init__(
        self,
        mesh: PolyMesh,
        specs: list[RegionExtrudeSpec],
    ) -> None:
        self._mesh = mesh
        self._specs = specs

    def generate(self) -> dict[str, PolyMesh]:
        """Generate the multi-region mesh.

        Returns:
            Dictionary mapping region name to PolyMesh:
            - ``"source"``: the original mesh (with extruded faces removed)
            - Additional keys for each extruded region.
        """
        device = get_device()
        dtype = get_default_dtype()

        result: dict[str, PolyMesh] = {}

        for spec in self._specs:
            region_mesh = self._extrude_patch(spec, device, dtype)
            result[spec.region_name] = region_mesh

        # Also return the source mesh
        result["source"] = self._mesh

        return result

    def _extrude_patch(
        self,
        spec: RegionExtrudeSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PolyMesh:
        """Extrude faces from one patch into a new region mesh."""
        mesh = self._mesh
        src_points = mesh.points.to(device=device, dtype=dtype)
        src_faces = mesh.faces
        src_owner = mesh.owner
        n_src_points = src_points.shape[0]
        n_src_cells = mesh.n_cells

        # Find the source patch
        patch_info = None
        for p in mesh.boundary:
            if p["name"] == spec.source_patch:
                patch_info = p
                break

        if patch_info is None:
            raise ValueError(
                f"Patch '{spec.source_patch}' not found. "
                f"Available: {[p['name'] for p in mesh.boundary]}"
            )

        start_face = patch_info["startFace"]
        n_patch_faces = patch_info["nFaces"]

        # Face normals (outward from owner cell)
        if hasattr(mesh, "face_normals"):
            face_normals = mesh.face_normals.to(device=device, dtype=dtype)
        else:
            # Compute simple face normals from first 3 vertices
            face_normals = torch.zeros((len(src_faces), 3), dtype=dtype, device=device)
            for fi in range(len(src_faces)):
                pts = src_points[src_faces[fi].tolist()]
                if pts.shape[0] >= 3:
                    e1 = pts[1] - pts[0]
                    e2 = pts[2] - pts[0]
                    n = torch.linalg.cross(e1, e2)
                    face_normals[fi] = n / n.norm().clamp(min=1e-30)

        # Layer thicknesses
        thicknesses = spec.layer_thicknesses()
        n_layers = len(thicknesses)

        # --- Generate new region mesh ---
        new_points_list = []
        new_faces = []
        new_owner = []
        new_neighbour = []

        # Point offset for new points
        pt_offset = 0

        # For each patch face, create a column of prismatic cells
        for fi_local in range(n_patch_faces):
            fi = start_face + fi_local
            face_pts = src_faces[fi].tolist()
            n_pts_face = len(face_pts)

            # Face normal
            normal = face_normals[fi]

            # Face centre for offset reference
            face_centre = src_points[face_pts].mean(dim=0)

            # Generate layer points
            layer_pts = []  # layer_pts[layer_i] = list of point indices
            for layer in range(n_layers + 1):
                # Cumulative offset
                cum_offset = sum(thicknesses[:layer])
                offset = normal * cum_offset

                layer_indices = []
                for p_idx in face_pts:
                    new_pt = src_points[p_idx] + offset
                    new_points_list.append(new_pt)
                    layer_indices.append(pt_offset)
                    pt_offset += 1

                layer_pts.append(layer_indices)

            # Create cells (prism layers)
            for layer in range(n_layers):
                cell_idx = fi_local * n_layers + layer

                # Bottom face of this cell
                bottom = layer_pts[layer]
                # Top face of this cell
                top = layer_pts[layer + 1]

                # Owner face (bottom) — outward normal points down
                new_faces.append(torch.tensor(bottom, dtype=INDEX_DTYPE))
                new_owner.append(cell_idx)

                # Neighbour face (top) — shared with next layer or boundary
                if layer < n_layers - 1:
                    # Internal: top face shared with next cell
                    new_faces.append(torch.tensor(top, dtype=INDEX_DTYPE))
                    new_owner.append(cell_idx)
                    new_neighbour.append(cell_idx + 1)

                # Side faces connecting bottom to top
                for j in range(n_pts_face):
                    j_next = (j + 1) % n_pts_face
                    side_face = [
                        bottom[j], bottom[j_next],
                        top[j_next], top[j],
                    ]
                    new_faces.append(torch.tensor(side_face, dtype=INDEX_DTYPE))
                    new_owner.append(cell_idx)
                    # Side faces are boundary (no neighbour)

        n_internal = len(new_neighbour)
        n_total = len(new_faces)
        n_boundary = n_total - n_internal

        # New points tensor
        if new_points_list:
            new_points = torch.stack(new_points_list, dim=0)
        else:
            new_points = torch.zeros((0, 3), dtype=dtype, device=device)

        # Boundary patches
        boundary = []
        # Bottom faces (connected to source mesh)
        n_bottom = n_patch_faces
        # Top faces (outer boundary)
        n_top = n_patch_faces
        # Side faces
        n_side = n_boundary - n_bottom - n_top

        start = n_internal
        if n_bottom > 0:
            boundary.append({
                "name": f"{spec.region_name}_source",
                "type": "wall",
                "startFace": start,
                "nFaces": n_bottom,
            })
            start += n_bottom
        if n_top > 0:
            boundary.append({
                "name": f"{spec.region_name}_outer",
                "type": "wall",
                "startFace": start,
                "nFaces": n_top,
            })
            start += n_top
        if n_side > 0:
            boundary.append({
                "name": f"{spec.region_name}_sides",
                "type": "wall",
                "startFace": start,
                "nFaces": n_side,
            })

        owner_t = torch.tensor(new_owner, dtype=INDEX_DTYPE, device=device)
        nbr_t = torch.tensor(new_neighbour, dtype=INDEX_DTYPE, device=device)

        return PolyMesh(
            points=new_points,
            faces=new_faces,
            owner=owner_t,
            neighbour=nbr_t,
            boundary=boundary,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def extrude_to_region(
    mesh: PolyMesh,
    source_patch: str,
    region_name: str = "region1",
    n_layers: int = 5,
    total_thickness: float = 0.1,
    expansion_ratio: float = 1.0,
) -> dict[str, PolyMesh]:
    """Extrude a boundary patch into a separate mesh region.

    Args:
        mesh: Source mesh.
        source_patch: Name of boundary patch to extrude.
        region_name: Name for the new region.
        n_layers: Number of extrusion layers.
        total_thickness: Total extrusion distance.
        expansion_ratio: Layer expansion ratio.

    Returns:
        Dictionary with ``"source"`` and the new region name as keys.
    """
    spec = RegionExtrudeSpec(
        source_patch=source_patch,
        region_name=region_name,
        n_layers=n_layers,
        total_thickness=total_thickness,
        expansion_ratio=expansion_ratio,
    )
    extruder = ExtrudeToRegion(mesh=mesh, specs=[spec])
    return extruder.generate()
