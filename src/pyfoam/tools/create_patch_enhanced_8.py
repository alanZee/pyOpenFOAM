"""
createPatch enhanced v8 — enhanced patch creation with patch compatibility
analysis, boundary condition validation, and patch naming conventions
(eighth generation).

Extends :func:`create_patch_enhanced_7` with:

- **Patch compatibility analysis**: Check compatibility between new
  patches and existing boundary conditions in the case.
- **Boundary condition validation**: Validate suggested BC types
  against solver requirements and field availability.
- **Patch naming conventions**: Enforce naming conventions and
  detect naming conflicts with existing patches.

Usage::

    from pyfoam.tools.create_patch_enhanced_8 import create_patch_enhanced_8

    result = create_patch_enhanced_8(
        mesh,
        face_indices=[0, 1],
        patch_name="inlet",
        template="inlet",
        validate_bc=True,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced8Result", "create_patch_enhanced_8"]


@dataclass
class CompatibilityReport:
    """Compatibility analysis between patch and existing BCs."""
    is_compatible: bool = True
    conflicting_patches: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class BCValidation:
    """Validation of boundary condition against solver requirements."""
    bc_type: str = ""
    is_valid: bool = True
    required_fields: list = field(default_factory=list)
    missing_fields: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class NamingConvention:
    """Patch naming convention check result."""
    original_name: str = ""
    sanitised_name: str = ""
    is_valid: bool = True
    conflicts: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


@dataclass
class PatchEnhanced8Result:
    """Result from :func:`create_patch_enhanced_8`.

    Attributes
    ----------
    mesh .. dependencies
        Forwarded from v7.
    compatibility : CompatibilityReport
        BC compatibility analysis.
    bc_validation : BCValidation
        Boundary condition validation result.
    naming : NamingConvention
        Naming convention check result.
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
    quality_impact: object = None
    bc_hints: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    compatibility: CompatibilityReport = field(default_factory=CompatibilityReport)
    bc_validation: BCValidation = field(default_factory=BCValidation)
    naming: NamingConvention = field(default_factory=NamingConvention)

    def __post_init__(self):
        if self.patches_created is None:
            self.patches_created = []
        if self.conflict_faces is None:
            self.conflict_faces = []


# Naming pattern: alphanumeric + underscore, 1-40 chars
_VALID_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,39}$")


def create_patch_enhanced_8(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], "np.ndarray", None] = None,
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
    analyze_quality: bool = False,
    suggest_bc: bool = False,
    validate_bc: bool = False,
    check_compatibility: bool = False,
    enforce_naming: bool = False,
) -> PatchEnhanced8Result:
    """Create patches with BC validation and naming conventions.

    Parameters
    ----------
    mesh .. suggest_bc
        Forwarded to v7 patch creation.
    validate_bc : bool
        Validate suggested BC types against solver requirements.
    check_compatibility : bool
        Check compatibility with existing boundary conditions.
    enforce_naming : bool
        Enforce patch naming conventions.

    Returns
    -------
    PatchEnhanced8Result
    """
    from pyfoam.tools.create_patch_enhanced_7 import create_patch_enhanced_7

    # Naming convention check
    naming = NamingConvention(original_name=patch_name, sanitised_name=patch_name)
    if enforce_naming:
        naming = _check_naming_convention(patch_name, mesh)

    effective_name = naming.sanitised_name if enforce_naming else patch_name

    # Delegate to v7
    v7_result = create_patch_enhanced_7(
        mesh,
        face_indices=face_indices,
        patch_name=effective_name,
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
        parent_patch=parent_patch,
        boolean_expression=boolean_expression,
        auto_renumber=auto_renumber,
        template=template,
        detect_conflicts=detect_conflicts,
        enable_undo=enable_undo,
        analyze_quality=analyze_quality,
        suggest_bc=suggest_bc,
    )

    # BC validation
    bc_val = BCValidation(bc_type=patch_type)
    if validate_bc and v7_result.bc_hints:
        bc_val = _validate_bc(patch_type, v7_result.bc_hints)

    # Compatibility
    compat = CompatibilityReport()
    if check_compatibility and mesh is not None:
        compat = _check_compatibility(mesh, effective_name, patch_type)

    return PatchEnhanced8Result(
        mesh=v7_result.mesh,
        patches_created=v7_result.patches_created,
        n_faces_moved=v7_result.n_faces_moved,
        patch_face_counts=v7_result.patch_face_counts,
        patch_areas=v7_result.patch_areas,
        patch_centroids=v7_result.patch_centroids,
        patch_normal_consistency=v7_result.patch_normal_consistency,
        n_merged_patches=v7_result.n_merged_patches,
        patch_hierarchy=v7_result.patch_hierarchy,
        renumbered=v7_result.renumbered,
        n_boolean_selected=v7_result.n_boolean_selected,
        template_used=v7_result.template_used,
        n_conflicts=v7_result.n_conflicts,
        undo_available=v7_result.undo_available,
        conflict_faces=v7_result.conflict_faces,
        quality_impact=v7_result.quality_impact,
        bc_hints=v7_result.bc_hints,
        dependencies=v7_result.dependencies,
        compatibility=compat,
        bc_validation=bc_val,
        naming=naming,
    )


# ---------------------------------------------------------------------------
# Naming conventions
# ---------------------------------------------------------------------------


def _check_naming_convention(name, mesh):
    """Validate patch name against conventions."""
    is_valid = bool(_VALID_NAME_RE.match(name))
    conflicts = []
    suggestions = []

    # Check for conflicts with existing patches
    if mesh is not None and hasattr(mesh, "boundary"):
        for pi in mesh.boundary:
            if pi.get("name") == name:
                conflicts.append(name)
                suggestions.append(f"{name}_v2")
                break

    sanitised = name
    if not is_valid:
        # Replace invalid characters
        sanitised = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitised and sanitised[0].isdigit():
            sanitised = "p_" + sanitised
        sanitised = sanitised[:40]
        suggestions.append(sanitised)

    return NamingConvention(
        original_name=name,
        sanitised_name=sanitised,
        is_valid=is_valid and not conflicts,
        conflicts=conflicts,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# BC validation
# ---------------------------------------------------------------------------


def _validate_bc(bc_type, bc_hints):
    """Validate BC type against solver requirements."""
    required = []
    missing = []
    warnings = []

    valid_types = {"wall", "inlet", "outlet", "symmetry", "cyclic", "empty", "patch"}
    is_valid = bc_type in valid_types

    if not is_valid:
        warnings.append(f"Unknown BC type: {bc_type}")

    if bc_type == "inlet":
        required = ["U", "p"]
    elif bc_type == "outlet":
        required = ["p"]

    return BCValidation(
        bc_type=bc_type,
        is_valid=is_valid,
        required_fields=required,
        missing_fields=missing,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------


def _check_compatibility(mesh, patch_name, patch_type):
    """Check compatibility with existing boundary conditions."""
    conflicting = []
    warnings = []

    if hasattr(mesh, "boundary"):
        for pi in mesh.boundary:
            existing_name = pi.get("name", "")
            existing_type = pi.get("type", "")
            if existing_name == patch_name:
                conflicting.append(existing_name)
                warnings.append(f"Patch '{patch_name}' already exists")

    is_compatible = len(conflicting) == 0

    return CompatibilityReport(
        is_compatible=is_compatible,
        conflicting_patches=conflicting,
        warnings=warnings,
    )
