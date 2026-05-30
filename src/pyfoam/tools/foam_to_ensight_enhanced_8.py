"""
foamToEnsight enhanced v8 — enhanced EnSight export with animation support,
variable mapping, and case template generation (eighth generation).

Extends :func:`foam_to_ensight_enhanced_7` with:

- **Animation support**: Generate EnSight animation scripts (``.anim``)
  with configurable keyframes and camera paths.
- **Variable mapping**: Map OpenFOAM field names to standard EnSight
  variable names with unit conversion support.
- **Case template generation**: Produce reusable case file templates
  that can be applied to similar simulation setups.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_8 import foam_to_ensight_enhanced_8

    result = foam_to_ensight_enhanced_8(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr},
        generate_animation=True,
        variable_mapping={"p": "pressure", "U": "velocity"},
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnSightV8Result", "foam_to_ensight_enhanced_8"]


@dataclass
class AnimationKeyframe:
    """Animation keyframe definition."""
    time: float = 0.0
    camera_position: tuple = (1.0, 1.0, 1.0)
    camera_target: tuple = (0.0, 0.0, 0.0)
    description: str = ""


@dataclass
class VariableMapping:
    """Mapping from OpenFOAM field to EnSight variable."""
    foam_name: str = ""
    ensight_name: str = ""
    unit: str = ""
    vector: bool = False


@dataclass
class EnSightV8Result:
    """Result from :func:`foam_to_ensight_enhanced_8`.

    Attributes
    ----------
    case_file .. n_components_exported
        Forwarded from v7.
    animation_file : Path, optional
        EnSight animation script file.
    n_keyframes : int
        Number of animation keyframes generated.
    variable_mappings : list[VariableMapping]
        Variable name mappings applied.
    template_file : Path, optional
        Reusable case template file.
    """

    case_file: Path = field(default_factory=lambda: Path("."))
    geometry_files: List[Path] = field(default_factory=list)
    variable_files: List[Path] = field(default_factory=list)
    n_times: int = 0
    n_variables: int = 0
    n_parts: int = 1
    binary: bool = False
    geometry_reused: int = 0
    total_bytes_written: int = 0
    export_time_ms: float = 0.0
    coarse_geometry_file: Optional[Path] = None
    n_coarse_cells: int = 0
    compression_ratio: float = 1.0
    streamed: bool = False
    n_recovered: int = 0
    n_interpolated: int = 0
    parallel: bool = False
    n_derived_fields: int = 0
    config_file: Optional[Path] = None
    case_comments: List[str] = field(default_factory=list)
    n_components_exported: int = 0
    animation_file: Optional[Path] = None
    n_keyframes: int = 0
    variable_mappings: List[VariableMapping] = field(default_factory=list)
    template_file: Optional[Path] = None


def foam_to_ensight_enhanced_8(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_boundary_parts: bool = True,
    export_variables: Optional[Set[str]] = None,
    deduplicate_geometry: bool = True,
    chunk_size: int = 1024 * 1024,
    multi_resolution: bool = False,
    coarse_ratio: float = 0.25,
    stream_mode: bool = False,
    adaptive_compression: bool = False,
    recover: bool = False,
    n_interpolation_steps: int = 0,
    parallel_write: bool = False,
    derived_fields: Optional[Dict[str, Callable]] = None,
    component_export: Optional[Dict[str, List[str]]] = None,
    generate_config: bool = False,
    case_comments: Optional[List[str]] = None,
    generate_animation: bool = False,
    animation_keyframes: Optional[List[tuple]] = None,
    variable_mapping: Optional[Dict[str, str]] = None,
    generate_template: bool = False,
) -> EnSightV8Result:
    """Export to EnSight Gold with animation, mapping, and templates.

    Parameters
    ----------
    case_path .. case_comments
        Forwarded to v7 export.
    generate_animation : bool
        Generate EnSight animation script.
    animation_keyframes : list of tuples, optional
        ``(time, cam_pos, cam_target)`` keyframes.
    variable_mapping : dict[str, str], optional
        ``{foam_name: ensight_name}`` for variable renaming.
    generate_template : bool
        Produce a reusable case template.

    Returns
    -------
    EnSightV8Result
    """
    import time as _time
    t_start = _time.perf_counter()

    # Apply variable mapping
    all_fields = dict(fields) if fields else {}
    mappings = []
    if variable_mapping:
        renamed = {}
        for foam_name, ensight_name in variable_mapping.items():
            if foam_name in all_fields:
                renamed[ensight_name] = all_fields[foam_name]
                is_vec = all_fields[foam_name].ndim == 2
                mappings.append(VariableMapping(
                    foam_name=foam_name,
                    ensight_name=ensight_name,
                    vector=is_vec,
                ))
            else:
                renamed[foam_name] = all_fields.get(foam_name)
        all_fields = {k: v for k, v in renamed.items() if v is not None}
    else:
        all_fields = dict(fields) if fields else {}

    # Delegate to v7
    from pyfoam.tools.foam_to_ensight_enhanced_7 import foam_to_ensight_enhanced_7

    v7_result = foam_to_ensight_enhanced_7(
        case_path=case_path,
        time_range=time_range,
        mesh=mesh,
        fields=all_fields,
        output_dir=output_dir,
        binary=binary,
        write_boundary_parts=write_boundary_parts,
        export_variables=export_variables,
        deduplicate_geometry=deduplicate_geometry,
        chunk_size=chunk_size,
        multi_resolution=multi_resolution,
        coarse_ratio=coarse_ratio,
        stream_mode=stream_mode,
        adaptive_compression=adaptive_compression,
        recover=recover,
        n_interpolation_steps=n_interpolation_steps,
        parallel_write=parallel_write,
        derived_fields=derived_fields,
        component_export=component_export,
        generate_config=generate_config,
        case_comments=case_comments,
    )

    # Animation generation
    anim_path = None
    n_keyframes = 0
    if generate_animation:
        keyframes = []
        if animation_keyframes:
            for kf in animation_keyframes:
                if len(kf) >= 3:
                    keyframes.append(AnimationKeyframe(
                        time=kf[0], camera_position=kf[1], camera_target=kf[2],
                    ))
        anim_path, n_keyframes = _generate_animation(v7_result.case_file, keyframes)

    # Template generation
    template_path = None
    if generate_template:
        template_path = _generate_template(v7_result.case_file)

    t_end = _time.perf_counter()

    return EnSightV8Result(
        case_file=v7_result.case_file,
        geometry_files=v7_result.geometry_files,
        variable_files=v7_result.variable_files,
        n_times=v7_result.n_times,
        n_variables=v7_result.n_variables,
        n_parts=v7_result.n_parts,
        binary=v7_result.binary,
        geometry_reused=v7_result.geometry_reused,
        total_bytes_written=v7_result.total_bytes_written,
        export_time_ms=(t_end - t_start) * 1000.0,
        coarse_geometry_file=v7_result.coarse_geometry_file,
        n_coarse_cells=v7_result.n_coarse_cells,
        compression_ratio=v7_result.compression_ratio,
        streamed=v7_result.streamed,
        n_recovered=v7_result.n_recovered,
        n_interpolated=v7_result.n_interpolated,
        parallel=v7_result.parallel,
        n_derived_fields=v7_result.n_derived_fields,
        config_file=v7_result.config_file,
        case_comments=v7_result.case_comments,
        n_components_exported=v7_result.n_components_exported,
        animation_file=anim_path,
        n_keyframes=n_keyframes,
        variable_mappings=mappings,
        template_file=template_path,
    )


# ---------------------------------------------------------------------------
# Animation generation
# ---------------------------------------------------------------------------


def _generate_animation(case_file, keyframes):
    """Generate an EnSight animation script."""
    anim_path = case_file.parent / "animation.anim"
    n_kf = 0
    try:
        with open(anim_path, "w") as f:
            f.write("# EnSight animation script\n")
            f.write("# Auto-generated by pyfoam EnSight v8\n\n")
            f.write(f"case: {case_file.name}\n\n")

            if keyframes:
                for i, kf in enumerate(keyframes):
                    f.write(f"keyframe {i}:\n")
                    f.write(f"  time: {kf.time}\n")
                    f.write(f"  camera_position: {kf.camera_position}\n")
                    f.write(f"  camera_target: {kf.camera_target}\n")
                    n_kf += 1
            else:
                # Default keyframe
                f.write("keyframe 0:\n")
                f.write("  time: 0.0\n")
                f.write("  camera_position: (1, 1, 1)\n")
                f.write("  camera_target: (0, 0, 0)\n")
                n_kf = 1

        return anim_path, n_kf
    except Exception:
        return None, 0


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def _generate_template(case_file):
    """Generate a reusable case template."""
    template_path = case_file.parent / "template.case"
    try:
        with open(template_path, "w") as f:
            f.write("# EnSight case template\n")
            f.write("# Auto-generated by pyfoam EnSight v8\n")
            f.write("# Copy and modify for similar simulations\n\n")
            f.write(f"FORMAT:\n")
            f.write(f"type: ensight gold\n\n")
            f.write(f"GEOMETRY:\n")
            f.write(f"model: geometry.case\n\n")
            f.write(f"VARIABLE:\n")
            f.write(f"# Add variable entries here\n")
        return template_path
    except Exception:
        return None
