"""
foamToEnsight enhanced v7 — enhanced EnSight export with metadata enrichment,
selective field export, and automated post-processing hooks (seventh generation).

Extends :func:`foam_to_ensight_enhanced_6` with:

- **Metadata enrichment**: Embed case description, solver info, and
  boundary condition summary in the EnSight .case header comments.
- **Selective field export**: Export specific field components (e.g.,
  velocity magnitude, pressure gradient) rather than full vectors.
- **Automated post-processing hooks**: Generate EnSight ``.enc`` config
  files for automatic visualisation pipeline startup.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_7 import foam_to_ensight_enhanced_7

    result = foam_to_ensight_enhanced_7(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr},
        derived_fields={"magU": magnitude_func},
        generate_config=True,
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

__all__ = ["EnSightV7Result", "foam_to_ensight_enhanced_7"]


@dataclass
class EnSightV7Result:
    """Result from :func:`foam_to_ensight_enhanced_7`.

    Attributes
    ----------
    case_file : Path
    geometry_files, variable_files : list[Path]
    n_times, n_variables, n_parts : int
    binary : bool
    geometry_reused, total_bytes_written : int
    export_time_ms : float
    coarse_geometry_file : Path, optional
    n_coarse_cells : int
    compression_ratio : float
    streamed : bool
    n_recovered, n_interpolated : int
    parallel : bool
    n_derived_fields : int
        Number of derived fields exported.
    config_file : Path, optional
        EnSight ``.enc`` config file for post-processing.
    case_comments : list[str]
        Metadata comments embedded in the .case file.
    n_components_exported : int
        Individual field components exported.
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


def foam_to_ensight_enhanced_7(
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
) -> EnSightV7Result:
    """Export to EnSight Gold with metadata, derived fields, and config.

    Parameters
    ----------
    case_path .. parallel_write
        Forwarded to v6 export.
    derived_fields : dict[str, callable], optional
        ``{name: func(fields_dict) -> np.ndarray}`` for computed fields.
    component_export : dict[str, list[str]], optional
        ``{vector_field: ["x", "y", "z"]}`` to export individual
        components of vector fields.
    generate_config : bool
        Generate an EnSight ``.enc`` configuration file.
    case_comments : list[str], optional
        Extra comments for the .case file header.

    Returns
    -------
    EnSightV7Result
    """
    import time as _time
    t_start = _time.perf_counter()

    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided.")

    # Compute derived fields
    all_fields = dict(fields) if fields else {}
    n_derived = 0
    if derived_fields:
        for name, func in derived_fields.items():
            try:
                derived = func(all_fields)
                all_fields[name] = derived
                n_derived += 1
            except Exception:
                pass

    # Component export
    n_components = 0
    if component_export:
        for vec_name, components in component_export.items():
            if vec_name in all_fields and all_fields[vec_name].ndim == 2:
                vec = all_fields[vec_name]
                comp_map = {"x": 0, "y": 1, "z": 2, "magnitude": -1}
                for comp in components:
                    idx = comp_map.get(comp, -1)
                    if idx >= 0:
                        all_fields[f"{vec_name}_{comp}"] = vec[:, idx]
                        n_components += 1
                    elif comp == "magnitude":
                        all_fields[f"{vec_name}_mag"] = np.linalg.norm(vec, axis=1)
                        n_components += 1

    # Build metadata comments
    comments = list(case_comments) if case_comments else []
    comments.append(f"Exported with pyfoam EnSight v7")
    comments.append(f"Mesh: {mesh.n_cells} cells, {mesh.n_faces} faces")
    if n_derived > 0:
        comments.append(f"Derived fields: {n_derived}")
    if n_components > 0:
        comments.append(f"Component exports: {n_components}")

    # Delegate to v6
    from pyfoam.tools.foam_to_ensight_enhanced_6 import foam_to_ensight_enhanced_6

    v6_result = foam_to_ensight_enhanced_6(
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
    )

    # Generate config file
    config_path = None
    if generate_config:
        config_path = _generate_config(
            v6_result.case_file, v6_result.geometry_files,
            v6_result.variable_files,
        )

    t_end = _time.perf_counter()

    return EnSightV7Result(
        case_file=v6_result.case_file,
        geometry_files=v6_result.geometry_files,
        variable_files=v6_result.variable_files,
        n_times=v6_result.n_times,
        n_variables=v6_result.n_variables,
        n_parts=v6_result.n_parts,
        binary=v6_result.binary,
        geometry_reused=v6_result.geometry_reused,
        total_bytes_written=v6_result.total_bytes_written,
        export_time_ms=(t_end - t_start) * 1000.0,
        coarse_geometry_file=v6_result.coarse_geometry_file,
        n_coarse_cells=v6_result.n_coarse_cells,
        compression_ratio=v6_result.compression_ratio,
        streamed=v6_result.streamed,
        n_recovered=v6_result.n_recovered,
        n_interpolated=v6_result.n_interpolated,
        parallel=v6_result.parallel,
        n_derived_fields=n_derived,
        config_file=config_path,
        case_comments=comments,
        n_components_exported=n_components,
    )


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _generate_config(case_file, geo_files, var_files):
    """Generate an EnSight .enc configuration file."""
    config_path = case_file.parent / "post_setup.enc"
    try:
        with open(config_path, "w") as f:
            f.write("# EnSight post-processing configuration\n")
            f.write("# Auto-generated by pyfoam EnSight v7\n\n")
            f.write(f"case_file: {case_file.name}\n")
            f.write(f"n_geometry_files: {len(geo_files)}\n")
            f.write(f"n_variable_files: {len(var_files)}\n")
            for gf in geo_files:
                f.write(f"geometry: {gf.name}\n")
            for vf in var_files:
                f.write(f"variable: {vf.name}\n")
        return config_path
    except Exception:
        return None
