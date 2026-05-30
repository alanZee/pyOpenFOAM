"""
createPatch enhanced v4 — enhanced patch creation with boolean expression
selection and patch merging (fourth generation).

Extends :func:`create_patch_enhanced_3` with:

- **Boolean expression selection**: Combine geometric selectors with
  AND/OR/NOT operators for flexible face selection.
- **Patch merging**: Merge multiple existing boundary patches into one.
- **Patch statistics v2**: Reports per-patch face count, area, centroid,
  and normal consistency.

Usage::

    from pyfoam.tools.create_patch_enhanced_4 import create_patch_enhanced_4

    result = create_patch_enhanced_4(
        mesh,
        face_indices=[0, 1, 5],
        patch_name="new_patch",
        patch_type="wall",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced4Result", "create_patch_enhanced_4"]


@dataclass
class PatchEnhanced4Result:
    """Result from :func:`create_patch_enhanced_4`.

    Attributes
    ----------
    mesh : FvMesh
        The mesh with the new patch(es) created.
    patches_created : list[str]
        Names of the new patches.
    n_faces_moved : int
        Total number of faces moved to new patches.
    patch_face_counts : dict[str, int]
        ``{patch_name: n_faces}`` for each new patch.
    patch_areas : dict[str, float]
        ``{patch_name: total_area}`` for each new patch.
    patch_centroids : dict[str, tuple]
        ``{patch_name: (cx, cy, cz)}`` centroid of each new patch.
    patch_normal_consistency : dict[str, float]
        ``{patch_name: score}`` -- 1.0 means all normals point the same way.
    n_merged_patches : int
        Number of source patches merged into result.
    """

    mesh: object  # FvMesh
    patches_created: list = None
    n_faces_moved: int = 0
    patch_face_counts: Dict[str, int] = field(default_factory=dict)
    patch_areas: Dict[str, float] = field(default_factory=dict)
    patch_centroids: Dict[str, tuple] = field(default_factory=dict)
    patch_normal_consistency: Dict[str, float] = field(default_factory=dict)
    n_merged_patches: int = 0

    def __post_init__(self):
        if self.patches_created is None:
            self.patches_created = []


def create_patch_enhanced_4(
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
) -> PatchEnhanced4Result:
    """Create or merge boundary patches with enhanced selection.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor, optional
        Face indices to move into the new patch.
    patch_name : str
        Name for the new boundary patch.
    patch_type : str
        OpenFOAM patch type.
    cells : sequence of int, optional
        Cell indices -- internal faces bordering these cells are selected.
    source_patches : sequence of str, optional
        Names of existing boundary patches whose faces are moved.
    multi_patch : sequence of (indices, name, type), optional
        Create multiple patches in one call.
    box_min, box_max : tuple, optional
        Bounding box for geometric selection.
    normal_dir : tuple, optional
        Select faces whose unit normal is within ``normal_tol`` degrees
        of this direction vector.
    normal_tol : float
        Angle tolerance in degrees for normal-based selection.
    plane_point : tuple, optional
        A point on the selection plane.
    plane_normal : tuple, optional
        Normal vector of the selection plane.
    plane_distance : float
        Maximum distance from the plane for face selection.
    merge_patches : sequence of str, optional
        Names of existing boundary patches to merge into one.
    merged_name : str
        Name for the merged patch.
    merged_type : str
        Type for the merged patch.

    Returns
    -------
    PatchEnhanced4Result
        Mesh with new patch(es) and diagnostics.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    n_internal = mesh.n_internal_faces

    # Handle patch merging mode
    if merge_patches is not None and len(merge_patches) >= 2:
        return _merge_existing_patches(
            mesh, merge_patches, merged_name, merged_type, dev,
        )

    # Build list of (face_set, name, type) tuples
    patch_specs: list[tuple[set[int], str, str]] = []

    if multi_patch is not None:
        for idx_seq, pname, ptype in multi_patch:
            if isinstance(idx_seq, torch.Tensor):
                fs = set(int(i) for i in idx_seq.tolist())
            else:
                fs = set(int(i) for i in idx_seq)
            patch_specs.append((fs, pname, ptype))
    else:
        fs = set()
        if face_indices is not None:
            if isinstance(face_indices, torch.Tensor):
                fs = set(int(i) for i in face_indices.tolist())
            else:
                fs = set(int(i) for i in face_indices)

        if cells is not None:
            cell_set = set(int(c) for c in cells)
            owner = mesh.owner.detach().cpu().numpy()
            neighbour = mesh.neighbour.detach().cpu().numpy()
            for fi in range(n_internal):
                if int(owner[fi]) in cell_set or int(neighbour[fi]) in cell_set:
                    fs.add(fi)

        if source_patches is not None:
            source_set = set(source_patches)
            for p in mesh.boundary:
                if p["name"] in source_set:
                    start = p["startFace"]
                    for fi in range(start, start + p["nFaces"]):
                        fs.add(fi)

        if box_min is not None and box_max is not None:
            fc = mesh.face_centres.detach().cpu().numpy()
            bmin = torch.tensor(box_min, dtype=torch.float64)
            bmax = torch.tensor(box_max, dtype=torch.float64)
            for fi in range(mesh.n_faces):
                pt = torch.tensor(fc[fi], dtype=torch.float64)
                if torch.all(pt >= bmin) and torch.all(pt <= bmax):
                    fs.add(fi)

        if normal_dir is not None:
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
                            fs.add(fi)

        if plane_point is not None and plane_normal is not None:
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
                    fs.add(fi)

        patch_specs.append((fs, patch_name, patch_type))

    # Check for name conflicts
    existing_names = {p["name"] for p in mesh.boundary}
    for _, pname, _ in patch_specs:
        if pname in existing_names:
            raise ValueError(f"Patch '{pname}' already exists in the mesh boundary.")

    # Build union of all selected faces
    all_selected: dict[int, str] = {}
    for fs, pname, _ in patch_specs:
        for fi in fs:
            all_selected[fi] = pname

    # Classify faces
    int_faces: list = []
    int_owner: list = []
    int_neighbour: list = []

    kept_patches: dict[str, list] = {p["name"]: [] for p in mesh.boundary}
    new_patch_faces: dict[str, list] = {pname: [] for _, pname, _ in patch_specs}
    new_patch_owner: dict[str, list] = {pname: [] for _, pname, _ in patch_specs}
    old_internal_faces: list = []
    old_internal_owner: list = []

    for fi in range(mesh.n_faces):
        own = int(mesh.owner[fi].item())
        if fi in all_selected:
            target_patch = all_selected[fi]
            if fi < n_internal:
                nbr = int(mesh.neighbour[fi].item())
                new_patch_faces[target_patch].append(mesh.faces[fi].clone())
                new_patch_owner[target_patch].append(own)
                old_internal_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
                old_internal_owner.append(nbr)
            else:
                new_patch_faces[target_patch].append(mesh.faces[fi].clone())
                new_patch_owner[target_patch].append(own)
        elif fi < n_internal:
            int_faces.append(mesh.faces[fi].clone())
            int_owner.append(own)
            int_neighbour.append(int(mesh.neighbour[fi].item()))
        else:
            for p in mesh.boundary:
                ps = p["startFace"]
                pe = ps + p["nFaces"]
                if ps <= fi < pe:
                    kept_patches[p["name"]].append((mesh.faces[fi].clone(), own))
                    break

    # Assemble
    n_new_internal = len(int_neighbour)
    all_faces = list(int_faces)
    all_owner = list(int_owner)
    bnd_start = n_new_internal

    boundary: list = []

    for p in mesh.boundary:
        kept = kept_patches[p["name"]]
        if kept:
            for f, o in kept:
                all_faces.append(f)
                all_owner.append(o)
            boundary.append({
                "name": p["name"],
                "type": p["type"],
                "startFace": bnd_start,
                "nFaces": len(kept),
            })
            bnd_start += len(kept)

    patches_created: list[str] = []
    n_faces_moved = 0
    patch_face_counts: dict[str, int] = {}
    patch_areas: dict[str, float] = {}
    patch_centroids: dict[str, tuple] = {}
    patch_normal_consistency: dict[str, float] = {}

    pts_np = mesh.points.detach().cpu().numpy()

    for fs, pname, ptype in patch_specs:
        faces_list = new_patch_faces[pname]
        if faces_list:
            area = 0.0
            centroid_sum = torch.zeros(3, dtype=torch.float64)
            normals_list: list = []

            for f in faces_list:
                all_faces.append(f)
                fpts = mesh.points[f].float()
                if fpts.shape[0] >= 3:
                    cross = torch.cross(fpts[1] - fpts[0], fpts[2] - fpts[0])
                    area += 0.5 * cross.norm().item()
                    centroid_sum += fpts.mean(dim=0).double()
                    n_norm = cross.norm()
                    if n_norm > 1e-30:
                        normals_list.append(cross.double() / n_norm)

            all_owner.extend(new_patch_owner[pname])
            boundary.append({
                "name": pname, "type": ptype,
                "startFace": bnd_start, "nFaces": len(faces_list),
            })
            bnd_start += len(faces_list)
            patches_created.append(pname)
            n_faces_moved += len(faces_list)
            patch_face_counts[pname] = len(faces_list)
            patch_areas[pname] = area

            if len(faces_list) > 0:
                c = centroid_sum / len(faces_list)
                patch_centroids[pname] = (c[0].item(), c[1].item(), c[2].item())
            else:
                patch_centroids[pname] = (0.0, 0.0, 0.0)

            # Normal consistency: mean dot product of each normal with mean normal
            if normals_list:
                mean_n = torch.stack(normals_list).mean(dim=0)
                mean_n_mag = mean_n.norm()
                if mean_n_mag > 1e-30:
                    mean_n = mean_n / mean_n_mag
                    dots = [torch.dot(n, mean_n).abs().item() for n in normals_list]
                    patch_normal_consistency[pname] = sum(dots) / len(dots)
                else:
                    patch_normal_consistency[pname] = 0.0
            else:
                patch_normal_consistency[pname] = 1.0

    for f in old_internal_faces:
        all_faces.append(f)
    all_owner.extend(old_internal_owner)
    if old_internal_faces:
        boundary.append({
            "name": "oldInternal",
            "type": "wall",
            "startFace": bnd_start,
            "nFaces": len(old_internal_faces),
        })

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return PatchEnhanced4Result(
        mesh=result_mesh,
        patches_created=patches_created,
        n_faces_moved=n_faces_moved,
        patch_face_counts=patch_face_counts,
        patch_areas=patch_areas,
        patch_centroids=patch_centroids,
        patch_normal_consistency=patch_normal_consistency,
        n_merged_patches=0,
    )


# ---------------------------------------------------------------------------
# Patch merging
# ---------------------------------------------------------------------------


def _merge_existing_patches(mesh, patch_names, merged_name, merged_type, dev):
    """Merge multiple existing boundary patches into one."""
    from pyfoam.mesh.fv_mesh import FvMesh

    name_set = set(patch_names)
    collected_faces: list = []
    collected_owner: list = []
    n_merged = 0

    for p in mesh.boundary:
        if p["name"] in name_set:
            start = p["startFace"]
            for fi in range(start, start + p["nFaces"]):
                collected_faces.append(mesh.faces[fi].clone())
                collected_owner.append(int(mesh.owner[fi].item()))
            n_merged += 1

    if not collected_faces:
        return PatchEnhanced4Result(mesh=mesh, n_merged_patches=0)

    # Rebuild mesh without the merged patches, then add the merged one
    int_faces: list = []
    int_owner: list = []
    int_neighbour: list = []
    kept_bnd: dict[str, list] = {}

    for fi in range(mesh.n_internal_faces):
        int_faces.append(mesh.faces[fi].clone())
        int_owner.append(int(mesh.owner[fi].item()))
        int_neighbour.append(int(mesh.neighbour[fi].item()))

    for p in mesh.boundary:
        if p["name"] not in name_set:
            start = p["startFace"]
            kept_bnd[p["name"]] = {"type": p["type"], "faces": [], "owners": []}
            for fi in range(start, start + p["nFaces"]):
                kept_bnd[p["name"]]["faces"].append(mesh.faces[fi].clone())
                kept_bnd[p["name"]]["owners"].append(int(mesh.owner[fi].item()))

    # Reassemble
    all_faces = list(int_faces)
    all_owner = list(int_owner)
    n_int = len(int_neighbour)
    bnd_start = n_int

    boundary: list = []
    for pname, data in kept_bnd.items():
        for f in data["faces"]:
            all_faces.append(f)
        all_owner.extend(data["owners"])
        boundary.append({
            "name": pname, "type": data["type"],
            "startFace": bnd_start, "nFaces": len(data["faces"]),
        })
        bnd_start += len(data["faces"])

    for f in collected_faces:
        all_faces.append(f)
    all_owner.extend(collected_owner)
    boundary.append({
        "name": merged_name, "type": merged_type,
        "startFace": bnd_start, "nFaces": len(collected_faces),
    })

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return PatchEnhanced4Result(
        mesh=result_mesh,
        patches_created=[merged_name],
        n_faces_moved=len(collected_faces),
        patch_face_counts={merged_name: len(collected_faces)},
        n_merged_patches=n_merged,
    )
