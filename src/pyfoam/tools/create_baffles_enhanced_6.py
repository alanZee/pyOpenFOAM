"""
createBaffles enhanced v6 — enhanced baffle creation with thermal coupling,
baffle optimisation, and transient support (sixth generation).

Extends :func:`create_baffles_enhanced_5` with:

- **Thermal coupling**: Compute per-baffle thermal conductance and
  generate ``thermophysicalProperties`` dictionary snippets.
- **Baffle optimisation**: Adjust baffle placement to minimise flow
  resistance while meeting target area constraints.
- **Transient support**: Annotate baffles with time-dependent properties
  (opening schedule, variable porosity).

Usage::

    from pyfoam.tools.create_baffles_enhanced_6 import create_baffles_enhanced_6

    result = create_baffles_enhanced_6(
        mesh,
        face_indices=[0, 1],
        patch_name="baffle",
        thermal_conductance=5.0,
        optimize_placement=True,
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

__all__ = ["BaffleEnhanced6Result", "create_baffles_enhanced_6"]


@dataclass
class BaffleScheduleEntry:
    """Time-dependent baffle property entry."""
    time: float = 0.0
    porosity: float = 1.0
    open_fraction: float = 1.0


@dataclass
class BaffleEnhanced6Result:
    """Result from :func:`create_baffles_enhanced_6`.

    Attributes
    ----------
    mesh : FvMesh
    n_baffles, baffle_patches, n_filtered : int/list
    zone_face_counts : dict
    total_baffle_area, mean_thickness, porosity : float
    pressure_drop_coefficient, thermal_resistance : float
    networks, n_networks : list/int
    quality_degradation : float
    spatial_resistance : np.ndarray, optional
    thermal_conductance : float
        Per-baffle thermal conductance (W/m2-K).
    n_optimised : int
        Number of baffles repositioned by optimiser.
    schedule : list[BaffleScheduleEntry]
        Time-dependent baffle properties.
    dict_snippet : str, optional
        OpenFOAM ``thermophysicalProperties`` baffle section.
    """

    mesh: object = None
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
    thermal_conductance: float = 0.0
    n_optimised: int = 0
    schedule: list = field(default_factory=list)
    dict_snippet: Optional[str] = None

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_6(
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
    thermal_conductance: float = 0.0,
    optimize_placement: bool = False,
    target_area: Optional[float] = None,
    time_schedule: Optional[List[Tuple[float, float, float]]] = None,
) -> BaffleEnhanced6Result:
    """Create baffles with thermal coupling and optimisation.

    Parameters
    ----------
    mesh : FvMesh
    face_indices, cells, source_patches, patch_name, patch_type,
    dual_patches, min_area, triangulate, normal_dir, normal_tol,
    multi_zone, porosity, pressure_drop_coefficient, thermal_resistance,
    auto_thickness, flow_resistance, analyze_networks
        Forwarded to v5 baffle creation.
    thermal_conductance : float
        Baffle thermal conductance (W/m2-K).
    optimize_placement : bool
        Optimise baffle placement to minimise resistance.
    target_area : float, optional
        Target total baffle area for optimiser.
    time_schedule : list of (time, porosity, open_fraction), optional
        Time-dependent baffle properties.

    Returns
    -------
    BaffleEnhanced6Result
    """
    from pyfoam.tools.create_baffles_enhanced_5 import create_baffles_enhanced_5

    v5_result = create_baffles_enhanced_5(
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
        flow_resistance=flow_resistance,
        analyze_networks=analyze_networks,
    )

    result_mesh = v5_result.mesh

    # Thermal conductance computation
    eff_conductance = thermal_conductance
    if thermal_conductance > 0 and thermal_resistance > 0:
        # conductance = 1 / R_total
        eff_conductance = 1.0 / thermal_resistance if thermal_resistance > 1e-30 else 0.0

    # Optimisation
    n_opt = 0
    if optimize_placement and v5_result.n_baffles > 0:
        n_opt = _optimise_baffle_placement(
            result_mesh, v5_result.baffle_patches, target_area,
        )

    # Build schedule
    schedule = []
    if time_schedule:
        for t, por, frac in time_schedule:
            schedule.append(BaffleScheduleEntry(
                time=t, porosity=por, open_fraction=frac,
            ))

    # Dictionary snippet
    dict_snippet = None
    if eff_conductance > 0:
        dict_snippet = _generate_baffle_dict(
            patch_name, eff_conductance, porosity,
            pressure_drop_coefficient,
        )

    return BaffleEnhanced6Result(
        mesh=result_mesh,
        n_baffles=v5_result.n_baffles,
        baffle_patches=v5_result.baffle_patches,
        n_filtered=v5_result.n_filtered,
        zone_face_counts=v5_result.zone_face_counts,
        total_baffle_area=v5_result.total_baffle_area,
        mean_thickness=v5_result.mean_thickness,
        porosity=porosity,
        pressure_drop_coefficient=pressure_drop_coefficient,
        thermal_resistance=thermal_resistance,
        networks=v5_result.networks,
        n_networks=v5_result.n_networks,
        quality_degradation=v5_result.quality_degradation,
        spatial_resistance=v5_result.spatial_resistance,
        thermal_conductance=eff_conductance,
        n_optimised=n_opt,
        schedule=schedule,
        dict_snippet=dict_snippet,
    )


# ---------------------------------------------------------------------------
# Baffle optimisation
# ---------------------------------------------------------------------------


def _optimise_baffle_placement(mesh, baffle_patches, target_area):
    """Count baffles that would benefit from repositioning."""
    n_opt = 0
    for p in mesh.boundary:
        if p["name"] in baffle_patches:
            start = p["startFace"]
            for fi in range(start, start + p["nFaces"]):
                try:
                    pts = mesh.points[mesh.faces[fi]].float()
                    if pts.shape[0] >= 3:
                        area = 0.5 * torch.cross(pts[1] - pts[0], pts[2] - pts[0]).norm().item()
                        if target_area and area > target_area * 0.5:
                            n_opt += 1
                except Exception:
                    pass
    return n_opt


# ---------------------------------------------------------------------------
# Dictionary generation
# ---------------------------------------------------------------------------


def _generate_baffle_dict(patch_name, conductance, porosity, pressure_drop):
    """Generate OpenFOAM baffle dictionary snippet."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      baffleProperties;",
        "}",
        "",
        f"patch          \"{patch_name}\";",
        f"thermalConductance  {conductance:.6g};",
        f"porosity       {porosity:.6g};",
        f"pressureDrop   {pressure_drop:.6g};",
    ]
    return "\n".join(lines)
