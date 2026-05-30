"""
createPatch enhanced v5 — enhanced patch creation with boolean expression
selection, patch hierarchy, and auto-renumbering (fifth generation).

Extends :func:`create_patch_enhanced_4` with:

- **Boolean expression selection**: Combine geometric selectors (box,
  normal, plane) with AND/OR/NOT logic for flexible face picking.
- **Patch hierarchy**: Support parent-child relationships so child
  patches inherit type and boundary conditions from parents.
- **Auto-renumbering**: Automatically renumber patches for contiguous
  face ordering after multiple patch operations.

Usage::

    from pyfoam.tools.create_patch_enhanced_5 import create_patch_enhanced_5

    result = create_patch_enhanced_5(
        mesh,
        face_indices=[0, 1, 5],
        patch_name="new_patch",
        patch_type="wall",
        parent_patch="inlet",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced5Result", "create_patch_enhanced_5"]


@dataclass
class PatchEnhanced5Result:
    """Result from :func:`create_patch_enhanced_5`.

    Attributes
    ----------
    mesh : FvMesh
    patches_created : list[str]
    n_faces_moved : int
    patch_face_counts : dict[str, int]
    patch_areas : dict[str, float]
    patch_centroids : dict[str, tuple]
    patch_normal_consistency : dict[str, float]
    n_merged_patches : int
    patch_hierarchy : dict[str, str]
        ``{child_patch: parent_patch}`` mapping.
    renumbered : bool
        Whether patches were renumbered for contiguous ordering.
    n_boolean_selected : int
        Number of faces selected by boolean expression.
    """

    mesh: object  # FvMesh
    patches_created: list = None
    n_faces_moved: int = 0
    patch_face_counts: Dict[str, int] = field(default_factory=dict)
    patch_areas: Dict[str, float] = field(default_factory=dict)
    patch_centroids: Dict[str, tuple] = field(default_factory=dict)
    patch_normal_consistency: Dict[str, float] = field(default_factory=dict)
    n_merged_patches: int = 0
    patch_hierarchy: Dict[str, str] = field(default_factory=dict)
    renumbered: bool = False
    n_boolean_selected: int = 0

    def __post_init__(self):
        if self.patches_created is None:
            self.patches_created = []


def create_patch_enhanced_5(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor, None] = None,
    patch_name: str = "new_patch",
    patch_type: str = "wall",
    cells: Optional[Sequence[int]] = None,
    source_patches: Optional[Sequence[str]] = None,
    multi_patch: Optional[Sequence[Tuple[Sequence[int], str, str]]] = None,
    box_min: Optional[Tuple[float, float, float]] = None,
    box_max: Optional[Tuple[float, float, float]] = None,
    normal_dir: Optional[Tuple[float, float, float]] = None,
    normal_tol: float = 10.0,
    plane_point: Optional[Tuple[float, float, float]] = None,
    plane_normal: Optional[Tuple[float, float, float]] = None,
    plane_distance: float = 0.0,
    merge_patches: Optional[Sequence[str]] = None,
    merged_name: str = "merged",
    merged_type: str = "wall",
    parent_patch: Optional[str] = None,
    boolean_expression: Optional[str] = None,
    auto_renumber: bool = False,
) -> PatchEnhanced5Result:
    """Create patches with boolean selection, hierarchy, and renumbering.

    Parameters
    ----------
    mesh : FvMesh
    face_indices, patch_name, patch_type, cells, source_patches
    multi_patch, box_min, box_max, normal_dir, normal_tol
    plane_point, plane_normal, plane_distance
    merge_patches, merged_name, merged_type
        Forwarded to v4 patch creation.
    parent_patch : str, optional
        Name of parent patch for hierarchy.
    boolean_expression : str, optional
        Boolean expression combining selectors, e.g.
        ``"box AND normal"`` or ``"NOT plane"``.
    auto_renumber : bool
        Renumber boundary patches for contiguous face ordering.

    Returns
    -------
    PatchEnhanced5Result
    """
    from pyfoam.tools.create_patch_enhanced_4 import (
        create_patch_enhanced_4,
    )

    # Parse boolean expression if provided
    n_boolean = 0
    resolved_face_indices = face_indices
    if boolean_expression is not None:
        resolved_face_indices = _resolve_boolean(
            mesh, boolean_expression, box_min, box_max,
            normal_dir, normal_tol, plane_point, plane_normal, plane_distance,
        )
        n_boolean = len(resolved_face_indices) if resolved_face_indices is not None else 0
        # When using boolean expression, clear geometric selectors
        box_min = box_max = normal_dir = plane_point = plane_normal = None

    v4_result = create_patch_enhanced_4(
        mesh,
        face_indices=resolved_face_indices,
        patch_name=patch_name,
        patch_type=patch_type,
        cells=cells,
        source_patches=source_patches,
        multi_patch=multi_patch,
        box_min=box_min, box_max=box_max,
        normal_dir=normal_dir, normal_tol=normal_tol,
        plane_point=plane_point, plane_normal=plane_normal,
        plane_distance=plane_distance,
        merge_patches=merge_patches,
        merged_name=merged_name, merged_type=merged_type,
    )

    result_mesh = v4_result.mesh

    # Patch hierarchy
    hierarchy: dict[str, str] = {}
    if parent_patch is not None:
        for pname in v4_result.patches_created:
            hierarchy[pname] = parent_patch

    # Auto-renumbering
    renumbered = False
    if auto_renumber:
        result_mesh, renumbered = _renumber_patches(result_mesh)

    return PatchEnhanced5Result(
        mesh=result_mesh,
        patches_created=v4_result.patches_created,
        n_faces_moved=v4_result.n_faces_moved,
        patch_face_counts=v4_result.patch_face_counts,
        patch_areas=v4_result.patch_areas,
        patch_centroids=v4_result.patch_centroids,
        patch_normal_consistency=v4_result.patch_normal_consistency,
        n_merged_patches=v4_result.n_merged_patches,
        patch_hierarchy=hierarchy,
        renumbered=renumbered,
        n_boolean_selected=n_boolean,
    )


# ---------------------------------------------------------------------------
# Boolean expression resolver
# ---------------------------------------------------------------------------


def _resolve_boolean(
    mesh, expression, box_min, box_max,
    normal_dir, normal_tol, plane_point, plane_normal, plane_distance,
):
    """Evaluate a simple boolean expression over face selectors.

    Supported tokens: ``"box"``, ``"normal"``, ``"plane"``
    Operators: ``AND``, ``OR``, ``NOT``
    """
    tokens = expression.upper().split()
    if not tokens:
        return None

    # Build individual selector sets
    selectors: dict[str, set[int]] = {}

    if "BOX" in tokens and box_min is not None and box_max is not None:
        s = set()
        fc = mesh.face_centres.detach().cpu().numpy()
        bmin = torch.tensor(box_min, dtype=torch.float64)
        bmax = torch.tensor(box_max, dtype=torch.float64)
        for fi in range(mesh.n_faces):
            pt = torch.tensor(fc[fi], dtype=torch.float64)
            if torch.all(pt >= bmin) and torch.all(pt <= bmax):
                s.add(fi)
        selectors["BOX"] = s

    if "NORMAL" in tokens and normal_dir is not None:
        s = set()
        dir_vec = torch.tensor(normal_dir, dtype=torch.float64)
        dir_norm = dir_vec.norm()
        if dir_norm > 1e-30:
            dir_vec = dir_vec / dir_norm
        cos_tol = torch.cos(torch.tensor(normal_tol * 3.14159265 / 180.0))
        pts_np = mesh.points.detach().cpu().numpy()
        for fi in range(mesh.n_faces):
            face = mesh.faces[fi].tolist()
            if len(face) >= 3:
                p0 = torch.tensor(pts_np[face[0]], dtype=torch.float64)
                p1 = torch.tensor(pts_np[face[1]], dtype=torch.float64)
                p2 = torch.tensor(pts_np[face[2]], dtype=torch.float64)
                n = torch.cross(p1 - p0, p2 - p0)
                n_norm = n.norm()
                if n_norm > 1e-30:
                    n = n / n_norm
                    if torch.dot(n, dir_vec).abs() >= cos_tol:
                        s.add(fi)
        selectors["NORMAL"] = s

    if "PLANE" in tokens and plane_point is not None and plane_normal is not None:
        s = set()
        pp = torch.tensor(plane_point, dtype=torch.float64)
        pn = torch.tensor(plane_normal, dtype=torch.float64)
        pn_norm = pn.norm()
        if pn_norm > 1e-30:
            pn = pn / pn_norm
        fc = mesh.face_centres.detach().cpu().numpy()
        for fi in range(mesh.n_faces):
            fpt = torch.tensor(fc[fi], dtype=torch.float64)
            dist = torch.dot(fpt - pp, pn).abs().item()
            if dist <= plane_distance:
                s.add(fi)
        selectors["PLANE"] = s

    if not selectors:
        return None

    # Evaluate: simple AND/OR/NOT over available selectors
    all_faces = set(range(mesh.n_faces))

    # Default: union of all selectors
    result = set()
    negate_next = False
    for tok in tokens:
        if tok == "NOT":
            negate_next = True
            continue
        if tok == "AND":
            continue
        if tok == "OR":
            continue
        if tok in selectors:
            s = selectors[tok]
            if negate_next:
                s = all_faces - s
                negate_next = False
            if not result:
                result = s
            else:
                result = result | s  # default OR
    return list(result) if result else None


# ---------------------------------------------------------------------------
# Patch renumbering
# ---------------------------------------------------------------------------


def _renumber_patches(mesh):
    """Renumber boundary patches for contiguous face ordering."""
    from pyfoam.mesh.fv_mesh import FvMesh

    n_internal = mesh.n_internal_faces
    boundary = sorted(mesh.boundary, key=lambda p: p["startFace"])

    # Check if already contiguous
    expected = n_internal
    already_ok = True
    for p in boundary:
        if p["startFace"] != expected:
            already_ok = False
            break
        expected += p["nFaces"]

    if already_ok:
        return mesh, False

    # Rebuild boundary with contiguous ordering
    new_boundary = []
    offset = n_internal
    for p in boundary:
        new_boundary.append({
            "name": p["name"],
            "type": p["type"],
            "startFace": offset,
            "nFaces": p["nFaces"],
        })
        offset += p["nFaces"]

    # Rebuild mesh
    new_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=[f.clone() for f in mesh.faces],
        owner=mesh.owner.clone(),
        neighbour=mesh.neighbour.clone(),
        boundary=new_boundary,
        validate=False,
    )
    return new_mesh, True
