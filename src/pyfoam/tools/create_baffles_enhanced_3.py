"""
createBaffles enhanced v3 — enhanced baffle creation with normal-based
face selection and multi-zone baffles (third generation).

Extends :func:`create_baffles_enhanced_2` with:

- **Normal-based selection**: Create baffles from faces whose normal
  is within an angle tolerance of a given direction.
- **Multi-zone baffles**: Assign different baffle zones to different
  face sets with independent names and types.
- **Baffle statistics**: Reports per-zone face counts and total area.

Usage::

    from pyfoam.tools.create_baffles_enhanced_3 import create_baffles_enhanced_3

    result = create_baffles_enhanced_3(
        mesh,
        face_indices=[0, 1, 2],
        patch_name="baffle",
        dual_patches=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced3Result", "create_baffles_enhanced_3"]


@dataclass
class BaffleEnhanced3Result:
    """Result from :func:`create_baffles_enhanced_3`.

    Attributes
    ----------
    mesh : FvMesh
        The mesh with baffles created.
    n_baffles : int
        Number of baffle face pairs created.
    baffle_patches : list[str]
        Names of the baffle patches created.
    n_filtered : int
        Number of faces filtered out by area threshold.
    zone_face_counts : dict[str, int]
        ``{zone_name: n_faces}`` for each baffle zone.
    total_baffle_area : float
        Total area of all baffle faces.
    """

    mesh: object  # FvMesh
    n_baffles: int = 0
    baffle_patches: list = None
    n_filtered: int = 0
    zone_face_counts: Dict[str, int] = field(default_factory=dict)
    total_baffle_area: float = 0.0

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_3(
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
) -> BaffleEnhanced3Result:
    """Create baffles with normal selection and multi-zone support.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor, optional
        Indices of internal faces to convert to baffles.
    cells : sequence of int, optional
        Cell indices — all internal faces of these cells become baffles.
    source_patches : sequence of str, optional
        Names of existing boundary patches whose faces become baffles.
    patch_name : str
        Base name for the baffle patch.
    patch_type : str
        OpenFOAM patch type.
    dual_patches : bool
        If True, create separate ``"{patch_name}_left"`` and
        ``"{patch_name}_right"`` patches.
    min_area : float
        Minimum face area to create a baffle.
    triangulate : bool
        If True, triangulate non-triangular baffle faces.
    normal_dir : tuple, optional
        ``(nx, ny, nz)`` — select only faces whose normal is within
        ``normal_tol`` degrees of this direction.
    normal_tol : float
        Angle tolerance in degrees for normal-based selection.
    multi_zone : sequence of (name, indices, type), optional
        Create multiple baffle zones in one call.

    Returns
    -------
    BaffleEnhanced3Result
        Mesh with baffles and diagnostics.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    n_internal = mesh.n_internal_faces

    # Build zone specs
    zone_specs: list[tuple[str, set[int], str]] = []

    if multi_zone is not None:
        for zname, zindices, ztype in multi_zone:
            if isinstance(zindices, torch.Tensor):
                fs = set(int(i) for i in zindices.tolist())
            else:
                fs = set(int(i) for i in zindices)
            zone_specs.append((zname, fs, ztype))
    else:
        # Determine face set
        if face_indices is not None:
            if isinstance(face_indices, torch.Tensor):
                baffle_set = set(int(i) for i in face_indices.tolist())
            else:
                baffle_set = set(int(i) for i in face_indices)
        elif cells is not None:
            cell_set = set(int(c) for c in cells)
            baffle_set = set()
            owner = mesh.owner.detach().cpu().numpy()
            neighbour = mesh.neighbour.detach().cpu().numpy()
            for fi in range(n_internal):
                if int(owner[fi]) in cell_set or int(neighbour[fi]) in cell_set:
                    baffle_set.add(fi)
        elif source_patches is not None:
            source_set = set(source_patches)
            baffle_set = set()
            for p in mesh.boundary:
                if p["name"] in source_set:
                    start = p["startFace"]
                    for fi in range(start, start + p["nFaces"]):
                        baffle_set.add(fi)
        else:
            raise ValueError(
                "One of 'face_indices', 'cells', or 'source_patches' must be provided."
            )

        # Normal-based filtering
        if normal_dir is not None:
            dir_vec = torch.tensor(normal_dir, dtype=torch.float64)
            dir_norm = dir_vec.norm()
            if dir_norm > 1e-30:
                dir_vec = dir_vec / dir_norm
            cos_tol = torch.cos(torch.tensor(normal_tol * 3.14159265 / 180.0))
            normal_filtered = set()
            for fi in baffle_set:
                pts = mesh.points[mesh.faces[fi]].float()
                if pts.shape[0] >= 3:
                    n = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
                    n_norm = n.norm()
                    if n_norm > 1e-30:
                        n = n / n_norm
                        if torch.dot(n.double(), dir_vec).abs() >= cos_tol:
                            normal_filtered.add(fi)
                    else:
                        normal_filtered.add(fi)
                else:
                    normal_filtered.add(fi)
            baffle_set = normal_filtered

        zone_specs.append((patch_name, baffle_set, patch_type))

    # Validate faces
    for _, fs, _ in zone_specs:
        for fi in fs:
            if fi >= n_internal:
                raise ValueError(
                    f"Face {fi} is not an internal face (n_internal_faces={n_internal})."
                )

    # Area filtering across all zones
    n_filtered = 0
    total_baffle_area = 0.0
    if min_area > 0:
        new_specs = []
        for zname, fs, ztype in zone_specs:
            filtered = set()
            for fi in fs:
                pts = mesh.points[mesh.faces[fi]].float()
                if pts.shape[0] >= 3:
                    cross = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
                    area = 0.5 * cross.norm().item()
                    if area >= min_area:
                        filtered.add(fi)
                        total_baffle_area += area
                    else:
                        n_filtered += 1
                else:
                    filtered.add(fi)
            new_specs.append((zname, filtered, ztype))
        zone_specs = new_specs
    else:
        for _, fs, _ in zone_specs:
            for fi in fs:
                pts = mesh.points[mesh.faces[fi]].float()
                if pts.shape[0] >= 3:
                    cross = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
                    total_baffle_area += 0.5 * cross.norm().item()

    # Build unified baffle set
    all_baffle_set: set[int] = set()
    face_to_zone: dict[int, str] = {}
    for zname, fs, _ in zone_specs:
        for fi in fs:
            all_baffle_set.add(fi)
            face_to_zone[fi] = zname

    if not all_baffle_set:
        clone = FvMesh(
            points=mesh.points.clone(),
            faces=[f.clone() for f in mesh.faces],
            owner=mesh.owner.clone(),
            neighbour=mesh.neighbour.clone(),
            boundary=[dict(b) for b in mesh.boundary],
            validate=False,
        )
        return BaffleEnhanced3Result(
            mesh=clone, n_baffles=0, baffle_patches=[],
            n_filtered=n_filtered, total_baffle_area=0.0,
        )

    # Build baffles
    new_int_faces: list = []
    new_int_owner: list = []
    new_int_neighbour: list = []
    new_bnd_faces: list = []
    new_bnd_owner: list = []

    for fi in range(mesh.n_faces):
        if fi < n_internal:
            if fi in all_baffle_set:
                own = int(mesh.owner[fi].item())
                nbr = int(mesh.neighbour[fi].item())
                face = mesh.faces[fi]
                if triangulate and face.shape[0] > 3:
                    pts_list = face.tolist()
                    for k in range(1, len(pts_list) - 1):
                        tri = torch.tensor(
                            [pts_list[0], pts_list[k], pts_list[k + 1]],
                            dtype=INDEX_DTYPE, device=dev,
                        )
                        new_bnd_faces.append(tri)
                        new_bnd_owner.append(own)
                        new_bnd_faces.append(torch.flip(tri, dims=[0]))
                        new_bnd_owner.append(nbr)
                else:
                    new_bnd_faces.append(face.clone())
                    new_bnd_owner.append(own)
                    new_bnd_faces.append(torch.flip(face, dims=[0]).clone())
                    new_bnd_owner.append(nbr)
            else:
                new_int_faces.append(mesh.faces[fi].clone())
                new_int_owner.append(int(mesh.owner[fi].item()))
                new_int_neighbour.append(int(mesh.neighbour[fi].item()))
        else:
            new_bnd_faces.append(mesh.faces[fi].clone())
            new_bnd_owner.append(int(mesh.owner[fi].item()))

    n_new_internal = len(new_int_neighbour)
    all_faces = new_int_faces + new_bnd_faces
    all_owner = new_int_owner + new_bnd_owner

    boundary: list = []
    bnd_start = n_new_internal
    n_baffle_faces = len(new_bnd_faces) - sum(p["nFaces"] for p in mesh.boundary)
    baffle_patch_names: list[str] = []

    if dual_patches:
        n_per_side = n_baffle_faces // 2
        left_name = f"{patch_name}_left"
        right_name = f"{patch_name}_right"
        boundary.append({
            "name": left_name, "type": patch_type,
            "startFace": bnd_start, "nFaces": n_per_side,
        })
        bnd_start += n_per_side
        boundary.append({
            "name": right_name, "type": patch_type,
            "startFace": bnd_start, "nFaces": n_per_side,
        })
        bnd_start += n_per_side
        baffle_patch_names = [left_name, right_name]
    else:
        if n_baffle_faces > 0:
            boundary.append({
                "name": patch_name, "type": patch_type,
                "startFace": bnd_start, "nFaces": n_baffle_faces,
            })
            bnd_start += n_baffle_faces
            baffle_patch_names = [patch_name]

    for patch in mesh.boundary:
        orig_start = patch["startFace"]
        orig_end = orig_start + patch["nFaces"]
        removed = sum(1 for fi in range(orig_start, orig_end) if fi in all_baffle_set)
        new_n = patch["nFaces"] - removed
        if new_n > 0:
            boundary.append({
                "name": patch["name"],
                "type": patch["type"],
                "startFace": bnd_start,
                "nFaces": new_n,
            })
            bnd_start += new_n

    # Zone face counts
    zone_face_counts: dict[str, int] = {}
    for zname, fs, _ in zone_specs:
        zone_face_counts[zname] = zone_face_counts.get(zname, 0) + len(fs)

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(new_int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return BaffleEnhanced3Result(
        mesh=result_mesh,
        n_baffles=len(all_baffle_set),
        baffle_patches=baffle_patch_names,
        n_filtered=n_filtered,
        zone_face_counts=zone_face_counts,
        total_baffle_area=total_baffle_area,
    )
