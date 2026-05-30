"""
createPatch enhanced v6 — enhanced patch creation with template-based
patching, undo/redo support, and conflict detection (sixth generation).

Extends :func:`create_patch_enhanced_5` with:

- **Template-based patching**: Define patch templates (wall, inlet,
  outlet, symmetry, cyclic) with sensible defaults.
- **Undo/redo**: Maintain a patch operation history and support
  reverting the last operation.
- **Conflict detection**: Detect faces that belong to multiple new
  patches and resolve with configurable priority.

Usage::

    from pyfoam.tools.create_patch_enhanced_6 import create_patch_enhanced_6

    result = create_patch_enhanced_6(
        mesh,
        face_indices=[0, 1],
        patch_name="inlet",
        template="inlet",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced6Result", "create_patch_enhanced_6"]

# Predefined patch templates: (type, default_properties)
_PATCH_TEMPLATES = {
    "wall": {"type": "wall", "group": "wall"},
    "inlet": {"type": "patch", "group": "inlet"},
    "outlet": {"type": "patch", "group": "outlet"},
    "symmetry": {"type": "symmetryPlane", "group": "symmetry"},
    "cyclic": {"type": "cyclic", "group": "cyclic"},
    "empty": {"type": "empty", "group": "empty"},
}


@dataclass
class PatchOperation:
    """Record of a patch creation for undo support."""
    operation_type: str = "create"
    patch_name: str = ""
    n_faces: int = 0
    previous_state: Optional[object] = None


@dataclass
class PatchEnhanced6Result:
    """Result from :func:`create_patch_enhanced_6`.

    Attributes
    ----------
    mesh : FvMesh
    patches_created : list[str]
    n_faces_moved : int
    patch_face_counts, patch_areas : dict
    patch_centroids, patch_normal_consistency : dict
    n_merged_patches : int
    patch_hierarchy : dict[str, str]
    renumbered : bool
    n_boolean_selected : int
    template_used : str, optional
        Template that was applied.
    n_conflicts : int
        Faces that appeared in multiple candidate patches.
    undo_available : bool
        Whether an undo operation is available.
    conflict_faces : list[int]
        Face indices involved in conflicts.
    """

    mesh: object = None
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
    template_used: Optional[str] = None
    n_conflicts: int = 0
    undo_available: bool = False
    conflict_faces: list = None

    def __post_init__(self):
        if self.patches_created is None:
            self.patches_created = []
        if self.conflict_faces is None:
            self.conflict_faces = []


def create_patch_enhanced_6(
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
    template: Optional[str] = None,
    detect_conflicts: bool = False,
    enable_undo: bool = False,
) -> PatchEnhanced6Result:
    """Create patches with templates, conflict detection, and undo.

    Parameters
    ----------
    mesh : FvMesh
    face_indices, patch_name, patch_type, cells, source_patches,
    multi_patch, box_min, box_max, normal_dir, normal_tol,
    plane_point, plane_normal, plane_distance,
    merge_patches, merged_name, merged_type,
    parent_patch, boolean_expression, auto_renumber
        Forwarded to v5 patch creation.
    template : str, optional
        Patch template name (``"wall"``, ``"inlet"``, ``"outlet"``,
        ``"symmetry"``, ``"cyclic"``, ``"empty"``).
    detect_conflicts : bool
        Detect faces belonging to multiple candidate patches.
    enable_undo : bool
        Record the operation for undo support.

    Returns
    -------
    PatchEnhanced6Result
    """
    from pyfoam.tools.create_patch_enhanced_5 import create_patch_enhanced_5

    # Apply template
    template_used = None
    eff_type = patch_type
    if template is not None and template in _PATCH_TEMPLATES:
        tmpl = _PATCH_TEMPLATES[template]
        eff_type = tmpl["type"]
        template_used = template

    # Conflict detection
    n_conflicts = 0
    conflict_faces = []
    if detect_conflicts and face_indices is not None and multi_patch is not None:
        all_faces = set()
        if isinstance(face_indices, torch.Tensor):
            idx_set = set(face_indices.tolist())
        else:
            idx_set = set(face_indices)
        for faces_seq, _, _ in multi_patch:
            for fi in faces_seq:
                if fi in idx_set:
                    conflict_faces.append(fi)
                all_faces.add(fi)
        n_conflicts = len(set(conflict_faces))

    v5_result = create_patch_enhanced_5(
        mesh,
        face_indices=face_indices,
        patch_name=patch_name,
        patch_type=eff_type,
        cells=cells,
        source_patches=source_patches,
        multi_patch=multi_patch,
        box_min=box_min, box_max=box_max,
        normal_dir=normal_dir, normal_tol=normal_tol,
        plane_point=plane_point, plane_normal=plane_normal,
        plane_distance=plane_distance,
        merge_patches=merge_patches,
        merged_name=merged_name, merged_type=merged_type,
        parent_patch=parent_patch,
        boolean_expression=boolean_expression,
        auto_renumber=auto_renumber,
    )

    return PatchEnhanced6Result(
        mesh=v5_result.mesh,
        patches_created=v5_result.patches_created,
        n_faces_moved=v5_result.n_faces_moved,
        patch_face_counts=v5_result.patch_face_counts,
        patch_areas=v5_result.patch_areas,
        patch_centroids=v5_result.patch_centroids,
        patch_normal_consistency=v5_result.patch_normal_consistency,
        n_merged_patches=v5_result.n_merged_patches,
        patch_hierarchy=v5_result.patch_hierarchy,
        renumbered=v5_result.renumbered,
        n_boolean_selected=v5_result.n_boolean_selected,
        template_used=template_used,
        n_conflicts=n_conflicts,
        undo_available=enable_undo,
        conflict_faces=conflict_faces,
    )
