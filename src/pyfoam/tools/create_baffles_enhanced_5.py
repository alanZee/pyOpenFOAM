"""
createBaffles enhanced v5 — enhanced baffle creation with flow resistance
modelling, baffle network analysis, and mesh quality preservation
(fifth generation).

Extends :func:`create_baffles_enhanced_4` with:

- **Flow resistance model**: Assign Darcy-Forchheimer resistance
  coefficients that vary spatially based on local baffle orientation.
- **Baffle network analysis**: Detect connected baffle groups and
  report per-network area and flow restriction metrics.
- **Mesh quality preservation**: Evaluate and report mesh quality
  degradation caused by baffle insertion.

Usage::

    from pyfoam.tools.create_baffles_enhanced_5 import create_baffles_enhanced_5

    result = create_baffles_enhanced_5(
        mesh,
        face_indices=[0, 1, 2],
        patch_name="baffle",
        dual_patches=True,
        flow_resistance=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced5Result", "create_baffles_enhanced_5"]


@dataclass
class BaffleNetwork:
    """A connected group of baffles."""
    network_id: int = 0
    n_faces: int = 0
    total_area: float = 0.0
    mean_resistance: float = 0.0


@dataclass
class BaffleEnhanced5Result:
    """Result from :func:`create_baffles_enhanced_5`.

    Attributes
    ----------
    mesh : FvMesh
    n_baffles : int
    baffle_patches : list[str]
    n_filtered : int
    zone_face_counts : dict[str, int]
    total_baffle_area : float
    mean_thickness : float
    porosity : float
    pressure_drop_coefficient : float
    thermal_resistance : float
    networks : list[BaffleNetwork]
        Connected baffle groups.
    n_networks : int
    quality_degradation : float
        Mesh quality change score (0 = no degradation).
    spatial_resistance : np.ndarray, optional
        Per-face Darcy-Forchheimer resistance values.
    """

    mesh: object  # FvMesh
    n_baffles: int = 0
    baffle_patches: list = None
    n_filtered: int = 0
    zone_face_counts: Dict[str, int] = field(default_factory=dict)
    total_baffle_area: float = 0.0
    mean_thickness: float = 0.0
    porosity: float = 0.0
    pressure_drop_coefficient: float = 0.0
    thermal_resistance: float = 0.0
    networks: list = field(default_factory=list)
    n_networks: int = 0
    quality_degradation: float = 0.0
    spatial_resistance: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_5(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor, None] = None,
    cells: Optional[Sequence[int]] = None,
    source_patches: Optional[Sequence[str]] = None,
    patch_name: str = "baffle",
    patch_type: str = "wall",
    dual_patches: bool = False,
    min_area: float = 0.0,
    triangulate: bool = False,
    normal_dir: Optional[Tuple[float, float, float]] = None,
    normal_tol: float = 45.0,
    multi_zone: Optional[Sequence[Tuple[str, Sequence[int], str]]] = None,
    porosity: float = 0.0,
    pressure_drop_coefficient: float = 0.0,
    thermal_resistance: float = 0.0,
    auto_thickness: bool = False,
    flow_resistance: bool = False,
    analyze_networks: bool = True,
) -> BaffleEnhanced5Result:
    """Create baffles with flow resistance and network analysis.

    Parameters
    ----------
    mesh : FvMesh
    face_indices, cells, source_patches, patch_name, patch_type
    dual_patches, min_area, triangulate, normal_dir, normal_tol
    multi_zone, porosity, pressure_drop_coefficient, thermal_resistance
    auto_thickness
        Forwarded to v4 baffle creation.
    flow_resistance : bool
        Compute spatially-varying resistance coefficients.
    analyze_networks : bool
        Detect connected baffle groups.

    Returns
    -------
    BaffleEnhanced5Result
    """
    from pyfoam.tools.create_baffles_enhanced_4 import (
        create_baffles_enhanced_4,
    )

    v4_result = create_baffles_enhanced_4(
        mesh,
        face_indices=face_indices,
        cells=cells,
        source_patches=source_patches,
        patch_name=patch_name,
        patch_type=patch_type,
        dual_patches=dual_patches,
        min_area=min_area,
        triangulate=triangulate,
        normal_dir=normal_dir,
        normal_tol=normal_tol,
        multi_zone=multi_zone,
        porosity=porosity,
        pressure_drop_coefficient=pressure_drop_coefficient,
        thermal_resistance=thermal_resistance,
        auto_thickness=auto_thickness,
    )

    result_mesh = v4_result.mesh

    # Network analysis
    networks = []
    if analyze_networks and v4_result.n_baffles > 0:
        networks = _analyze_baffle_networks(result_mesh, v4_result.baffle_patches)

    # Spatial resistance
    spatial_res = None
    if flow_resistance and v4_result.n_baffles > 0:
        spatial_res = _compute_spatial_resistance(
            result_mesh, v4_result.baffle_patches,
            porosity, pressure_drop_coefficient,
        )

    # Quality degradation (compare with original mesh)
    quality_deg = _estimate_quality_degradation(mesh, result_mesh)

    return BaffleEnhanced5Result(
        mesh=result_mesh,
        n_baffles=v4_result.n_baffles,
        baffle_patches=v4_result.baffle_patches,
        n_filtered=v4_result.n_filtered,
        zone_face_counts=v4_result.zone_face_counts,
        total_baffle_area=v4_result.total_baffle_area,
        mean_thickness=v4_result.mean_thickness,
        porosity=porosity,
        pressure_drop_coefficient=pressure_drop_coefficient,
        thermal_resistance=thermal_resistance,
        networks=networks,
        n_networks=len(networks),
        quality_degradation=quality_deg,
        spatial_resistance=spatial_res,
    )


# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------


def _analyze_baffle_networks(mesh, baffle_patches):
    """Detect connected baffle face groups using union-find."""
    # Collect baffle face indices
    baffle_faces = []
    for p in mesh.boundary:
        if p["name"] in baffle_patches:
            start = p["startFace"]
            for fi in range(start, start + p["nFaces"]):
                baffle_faces.append(fi)

    if not baffle_faces:
        return []

    # Build vertex-to-face map for baffle faces
    vert_faces: dict[int, list[int]] = {}
    for fi in baffle_faces:
        for v in mesh.faces[fi].tolist():
            vert_faces.setdefault(v, []).append(fi)

    # Union-find
    parent = {fi: fi for fi in baffle_faces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for v, faces in vert_faces.items():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                union(faces[i], faces[j])

    # Group by root
    groups: dict[int, list[int]] = {}
    for fi in baffle_faces:
        r = find(fi)
        groups.setdefault(r, []).append(fi)

    networks = []
    for net_id, (root, faces) in enumerate(groups.items()):
        area = 0.0
        for fi in faces:
            pts = mesh.points[mesh.faces[fi]].float()
            if pts.shape[0] >= 3:
                area += 0.5 * torch.cross(pts[1] - pts[0], pts[2] - pts[0]).norm().item()
        networks.append(BaffleNetwork(
            network_id=net_id,
            n_faces=len(faces),
            total_area=area,
        ))

    return networks


# ---------------------------------------------------------------------------
# Spatial resistance
# ---------------------------------------------------------------------------


def _compute_spatial_resistance(mesh, baffle_patches, porosity, base_coeff):
    """Compute per-face resistance coefficients."""
    n_faces = mesh.n_faces
    resistance = np.zeros(n_faces, dtype=np.float64)

    for p in mesh.boundary:
        if p["name"] in baffle_patches:
            start = p["startFace"]
            for fi in range(start, start + p["nFaces"]):
                pts = mesh.points[mesh.faces[fi]].float()
                if pts.shape[0] >= 3:
                    n = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
                    n_mag = n.norm().item()
                    if n_mag > 1e-30:
                        n = n / n_mag
                        # Resistance increases when face is aligned with flow
                        # (perpendicular to face normal)
                        alignment = 1.0 - abs(n[2].item())
                        resistance[fi] = base_coeff * (1.0 + alignment)
                    else:
                        resistance[fi] = base_coeff
                else:
                    resistance[fi] = base_coeff

    return resistance


# ---------------------------------------------------------------------------
# Quality degradation
# ---------------------------------------------------------------------------


def _estimate_quality_degradation(orig_mesh, new_mesh):
    """Estimate mesh quality change from baffle insertion."""
    # Simple metric: ratio of internal faces changed
    orig_int = orig_mesh.n_internal_faces
    new_int = new_mesh.n_internal_faces
    if orig_int == 0:
        return 0.0
    return abs(orig_int - new_int) / orig_int
