"""
ExtrudeMesh — extrude a 2D mesh into 3D.

Mirrors OpenFOAM's ``extrudeMesh`` utility.  Takes a 2D mesh (in the
x-y plane) and extrudes it in a specified direction to create a 3D mesh.

Physics
-------
For a 2D face with vertices {v0, v1, ...}, the extrusion creates
prismatic cells by sweeping each face along the extrusion vector.
With ``n_layers`` layers, each face generates ``n_layers`` cells.

The extrusion vector for layer ``i`` is::

    delta_i = layer_thickness(i) * n_extrude

where ``n_extrude`` is the unit extrusion direction and the layer
thickness follows an optional expansion ratio::

    layer_thickness(i) = t0 * exp_ratio^i

    t0 = total_thickness * (1 - exp_ratio) / (1 - exp_ratio^n_layers)

References
----------
- OpenFOAM ``extrudeMesh`` utility source
- OpenFOAM ``extrudeModel`` classes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh

__all__ = [
    "ExtrudeMesh",
    "ExtrudeModel",
    "LinearExtrude",
    "WedgeExtrude",
    "RotationalExtrude",
]


# ---------------------------------------------------------------------------
# Extrusion models
# ---------------------------------------------------------------------------


@dataclass
class ExtrudeModel:
    """Base extrusion model parameters.

    Attributes:
        n_layers: Number of extrusion layers.
        total_thickness: Total extrusion distance.
        expansion_ratio: Layer-to-layer expansion ratio.
        direction: Extrusion direction ``(dx, dy, dz)`` (need not be unit).
    """

    n_layers: int = 1
    total_thickness: float = 1.0
    expansion_ratio: float = 1.0
    direction: tuple = (0.0, 0.0, 1.0)

    def layer_thicknesses(self) -> list[float]:
        """Compute per-layer thicknesses with optional expansion.

        Returns:
            List of thickness values summing to ``total_thickness``.
        """
        n = self.n_layers
        r = self.expansion_ratio

        if abs(r - 1.0) < 1e-10:
            # Uniform
            t0 = self.total_thickness / n
            return [t0] * n

        # Geometric series: t0 * (1 - r^n) / (1 - r) = total_thickness
        t0 = self.total_thickness * (1.0 - r) / (1.0 - r ** n)
        return [t0 * r ** i for i in range(n)]

    def direction_unit(self) -> torch.Tensor:
        """Unit extrusion direction."""
        d = torch.tensor(self.direction, dtype=torch.float64)
        return d / d.norm().clamp(min=1e-30)


@dataclass
class LinearExtrude(ExtrudeModel):
    """Linear extrusion along a fixed direction.

    Each layer is offset by its thickness in the extrusion direction.
    """
    pass


@dataclass
class WedgeExtrude(ExtrudeModel):
    """Wedge extrusion for axisymmetric cases.

    Extrudes with a small angular displacement, creating wedge cells
    suitable for 2D axisymmetric simulations.

    Attributes:
        wedge_angle: Total wedge angle in degrees (default: 5.0).
    """
    wedge_angle: float = 5.0


@dataclass
class RotationalExtrude(ExtrudeModel):
    """Rotational extrusion around an axis.

    Extrudes faces by rotating them around a specified axis.

    Attributes:
        axis: Rotation axis ``(ax, ay, az)``.
        angle: Total rotation angle in degrees.
    """
    axis: tuple = (0.0, 0.0, 1.0)
    angle: float = 360.0


# ---------------------------------------------------------------------------
# ExtrudeMesh generator
# ---------------------------------------------------------------------------


class ExtrudeMesh:
    """Extrude a 2D mesh into 3D.

    Takes a :class:`~pyfoam.mesh.poly_mesh.PolyMesh` (or raw data)
    and extrudes it to create a 3D mesh.

    Parameters:
        source_mesh: The 2D source mesh to extrude.
        extrude_model: Extrusion model specifying layers and direction.
        patch_name: Name for the extruded boundary patch (default: ``"extruded"``).
        source_patch: Name for the original source patch (default: ``"source"``).
    """

    def __init__(
        self,
        source_mesh: PolyMesh,
        extrude_model: ExtrudeModel,
        patch_name: str = "extruded",
        source_patch: str = "source",
    ) -> None:
        self._source = source_mesh
        self._model = extrude_model
        self._patch_name = patch_name
        self._source_patch = source_patch

    def generate(self) -> PolyMesh:
        """Generate the extruded 3D mesh.

        Returns:
            PolyMesh with the extruded 3D mesh.
        """
        device = get_device()
        dtype = get_default_dtype()

        src_points = self._source.points.to(device=device, dtype=dtype)
        src_faces = self._source.faces
        src_owner = self._source.owner
        src_neighbour = self._source.neighbour
        n_src_internal = self._source.n_internal_faces
        src_boundary = self._source.boundary

        n_src_points = src_points.shape[0]
        n_src_faces = len(src_faces)
        n_src_cells = self._source.n_cells

        # Layer thicknesses
        thicknesses = self._model.layer_thicknesses()
        direction = self._model.direction_unit().to(device=device, dtype=dtype)

        n_layers = len(thicknesses)

        # Compute cumulative z-offsets for each layer
        offsets = [torch.zeros(3, dtype=dtype, device=device)]
        cum = torch.zeros(3, dtype=dtype, device=device)
        for t in thicknesses:
            cum = cum + t * direction
            offsets.append(cum.clone())

        # --- Generate points ---
        # Each layer replicates all source points with z-offset
        all_points = []
        for layer in range(n_layers + 1):
            offset = offsets[layer]
            layer_points = src_points + offset.unsqueeze(0)
            all_points.append(layer_points)
        new_points = torch.cat(all_points, dim=0)

        # --- Generate faces and owner/neighbour ---
        new_faces = []
        new_owner = []
        new_neighbour = []

        # Helper: remap point indices from layer
        def pt(layer, src_pt_idx):
            return layer * n_src_points + src_pt_idx

        # 1. Internal lateral faces (between adjacent layers)
        for layer in range(n_layers):
            cell_base = layer * n_src_cells
            next_cell_base = (layer + 1) * n_src_cells if layer + 1 < n_layers else -1

            # For each source face, create a lateral prism face
            for fi in range(n_src_faces):
                face_pts = src_faces[fi].tolist()
                n_pts_face = len(face_pts)

                # Create lateral face connecting this layer to next
                lateral_face = []
                for p in face_pts:
                    lateral_face.append(pt(layer, p))
                for p in reversed(face_pts):
                    lateral_face.append(pt(layer + 1, p))

                own = cell_base + (int(src_owner[fi].item()) if int(src_owner[fi].item()) < n_src_cells else 0)

                if fi < n_src_internal:
                    # Internal source face -> prism cell internal face
                    nbr_src = int(src_neighbour[fi].item())
                    # Create two lateral faces: one for owner-side, one for neighbour-side
                    # Owner-side lateral
                    own_cell = cell_base + int(src_owner[fi].item())
                    nbr_cell = cell_base + nbr_src

                    # Top face of owner prism cell (at next layer boundary)
                    top_face = [pt(layer + 1, p) for p in face_pts]
                    new_faces.append(torch.tensor(top_face, dtype=INDEX_DTYPE))
                    new_owner.append(own_cell)
                    new_neighbour.append(nbr_cell)

                else:
                    # Boundary source face -> side boundary face
                    src_own = int(src_owner[fi].item())
                    own_cell = cell_base + src_own if src_own < n_src_cells else cell_base

                    # Lateral face for boundary
                    new_faces.append(torch.tensor(lateral_face, dtype=INDEX_DTYPE))
                    new_owner.append(own_cell)

        # 2. Top and bottom faces for each cell in each layer
        for layer in range(n_layers):
            cell_base = layer * n_src_cells

            # For each source internal face, we already created the horizontal
            # face between prism cells. Now create top/bottom for boundary source faces.
            for fi in range(n_src_faces):
                if fi >= n_src_internal:
                    # Boundary face
                    face_pts = src_faces[fi].tolist()
                    src_own = int(src_owner[fi].item())
                    if src_own >= n_src_cells:
                        continue
                    own_cell = cell_base + src_own

                    # Bottom face of this prism cell
                    bottom_face = [pt(layer, p) for p in face_pts]
                    # Top face
                    top_face = [pt(layer + 1, p) for p in reversed(face_pts)]

                    if layer == 0:
                        new_faces.append(torch.tensor(bottom_face, dtype=INDEX_DTYPE))
                        new_owner.append(own_cell)

                    if layer == n_layers - 1:
                        new_faces.append(torch.tensor(top_face, dtype=INDEX_DTYPE))
                        new_owner.append(own_cell)

        # 3. Source boundary: bottom at layer 0, top at last layer
        for patch in src_boundary:
            start = patch["startFace"]
            n_f = patch["nFaces"]
            for j in range(n_f):
                fi = start + j
                if fi >= n_src_faces:
                    continue
                face_pts = src_faces[fi].tolist()
                src_own = int(src_owner[fi].item())
                if src_own >= n_src_cells:
                    continue

                # Bottom face (layer 0)
                bottom = [pt(0, p) for p in face_pts]
                new_faces.append(torch.tensor(bottom, dtype=INDEX_DTYPE))
                new_owner.append(src_own)

                # Top face (last layer)
                top = [pt(n_layers, p) for p in reversed(face_pts)]
                new_faces.append(torch.tensor(top, dtype=INDEX_DTYPE))
                new_owner.append((n_layers - 1) * n_src_cells + src_own)

        # Build boundary
        # Count boundary faces added (those with no neighbour)
        n_internal = len(new_neighbour)
        n_boundary = len(new_faces) - n_internal
        n_total_cells = n_layers * n_src_cells

        boundary = []
        if n_boundary > 0:
            boundary.append({
                "name": self._patch_name,
                "type": "wall",
                "startFace": n_internal,
                "nFaces": n_boundary,
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
# Convenience functions
# ---------------------------------------------------------------------------


def extrude_mesh(
    source_mesh: PolyMesh,
    direction: tuple = (0.0, 0.0, 1.0),
    n_layers: int = 10,
    total_thickness: float = 1.0,
    expansion_ratio: float = 1.0,
) -> PolyMesh:
    """Extrude a 2D mesh into 3D with a linear extrusion model.

    Args:
        source_mesh: 2D source mesh.
        direction: Extrusion direction vector.
        n_layers: Number of layers.
        total_thickness: Total extrusion distance.
        expansion_ratio: Layer expansion ratio.

    Returns:
        Extruded 3D PolyMesh.
    """
    model = LinearExtrude(
        n_layers=n_layers,
        total_thickness=total_thickness,
        expansion_ratio=expansion_ratio,
        direction=direction,
    )
    extruder = ExtrudeMesh(source_mesh=source_mesh, extrude_model=model)
    return extruder.generate()
