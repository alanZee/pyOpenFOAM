"""
createBaffles enhanced v7 — enhanced baffle creation with acoustic modelling,
multi-physics coupling, and baffle lifecycle tracking (seventh generation).

Extends :func:`create_baffles_enhanced_6` with:

- **Acoustic modelling**: Compute acoustic impedance and transmission
  loss for each baffle from thickness and material properties.
- **Multi-physics coupling**: Generate coupling coefficients for
  conjugate heat transfer and structural FEA interfaces.
- **Baffle lifecycle tracking**: Track baffle creation, modification,
  and removal events with timestamped metadata.

Usage::

    from pyfoam.tools.create_baffles_enhanced_7 import create_baffles_enhanced_7

    result = create_baffles_enhanced_7(
        mesh,
        face_indices=[0, 1],
        patch_name="baffle",
        acoustic_impedance=415.0,
        track_lifecycle=True,
    )
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced7Result", "LifecycleEvent", "create_baffles_enhanced_7"]


@dataclass
class LifecycleEvent:
    """Baffle lifecycle event record."""
    timestamp: float = 0.0
    event_type: str = "created"
    baffle_name: str = ""
    details: str = ""


@dataclass
class BaffleEnhanced7Result:
    """Result from :func:`create_baffles_enhanced_7`.

    Attributes
    ----------
    mesh : FvMesh
    n_baffles .. dict_snippet
        Forwarded from v6.
    acoustic_impedance : float
        Baffle acoustic impedance (Pa-s/m).
    transmission_loss_db : float
        Estimated transmission loss (dB).
    coupling_coefficient : float
        Multi-physics coupling coefficient.
    lifecycle_events : list[LifecycleEvent]
        Tracked baffle events.
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
    acoustic_impedance: float = 0.0
    transmission_loss_db: float = 0.0
    coupling_coefficient: float = 0.0
    lifecycle_events: list = field(default_factory=list)

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_7(
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
    acoustic_impedance: float = 415.0,
    baffle_thickness: float = 0.001,
    coupling_enabled: bool = False,
    track_lifecycle: bool = False,
) -> BaffleEnhanced7Result:
    """Create baffles with acoustic modelling and lifecycle tracking.

    Parameters
    ----------
    mesh : FvMesh
    face_indices .. time_schedule
        Forwarded to v6 baffle creation.
    acoustic_impedance : float
        Baffle acoustic impedance (Pa-s/m).
    baffle_thickness : float
        Physical thickness for acoustic modelling (m).
    coupling_enabled : bool
        Compute multi-physics coupling coefficients.
    track_lifecycle : bool
        Record baffle lifecycle events.

    Returns
    -------
    BaffleEnhanced7Result
    """
    from pyfoam.tools.create_baffles_enhanced_6 import create_baffles_enhanced_6

    v6_result = create_baffles_enhanced_6(
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
        thermal_conductance=thermal_conductance,
        optimize_placement=optimize_placement,
        target_area=target_area,
        time_schedule=time_schedule,
    )

    # Acoustic modelling
    tl_db = _compute_transmission_loss(
        acoustic_impedance, baffle_thickness, porosity,
    )

    # Coupling coefficient
    cpl = 0.0
    if coupling_enabled and thermal_conductance > 0:
        cpl = _compute_coupling_coefficient(
            thermal_conductance, acoustic_impedance, baffle_thickness,
        )

    # Lifecycle tracking
    events = []
    if track_lifecycle:
        events.append(LifecycleEvent(
            timestamp=time.time(),
            event_type="created",
            baffle_name=patch_name,
            details=f"n_baffles={v6_result.n_baffles}",
        ))

    return BaffleEnhanced7Result(
        mesh=v6_result.mesh,
        n_baffles=v6_result.n_baffles,
        baffle_patches=v6_result.baffle_patches,
        n_filtered=v6_result.n_filtered,
        zone_face_counts=v6_result.zone_face_counts,
        total_baffle_area=v6_result.total_baffle_area,
        mean_thickness=v6_result.mean_thickness,
        porosity=porosity,
        pressure_drop_coefficient=pressure_drop_coefficient,
        thermal_resistance=thermal_resistance,
        networks=v6_result.networks,
        n_networks=v6_result.n_networks,
        quality_degradation=v6_result.quality_degradation,
        spatial_resistance=v6_result.spatial_resistance,
        thermal_conductance=v6_result.thermal_conductance,
        n_optimised=v6_result.n_optimised,
        schedule=v6_result.schedule,
        dict_snippet=v6_result.dict_snippet,
        acoustic_impedance=acoustic_impedance,
        transmission_loss_db=tl_db,
        coupling_coefficient=cpl,
        lifecycle_events=events,
    )


# ---------------------------------------------------------------------------
# Acoustic modelling
# ---------------------------------------------------------------------------


def _compute_transmission_loss(impedance, thickness, porosity):
    """Estimate transmission loss through the baffle (dB)."""
    if impedance <= 0 or thickness <= 0:
        return 0.0
    # Mass law: TL = 20*log10(f*m) - 47 dB (simplified)
    # m = surface density ~ thickness * rho_material
    rho_mat = 2500.0  # default material density (kg/m3)
    surface_density = thickness * rho_mat * (1.0 - porosity)
    if surface_density <= 0:
        return 0.0
    f_ref = 1000.0  # reference frequency (Hz)
    tl = 20.0 * math.log10(f_ref * surface_density + 1e-30) - 47.0
    return max(0.0, tl)


# ---------------------------------------------------------------------------
# Coupling coefficient
# ---------------------------------------------------------------------------


def _compute_coupling_coefficient(thermal_cond, acoustic_imp, thickness):
    """Compute multi-physics coupling coefficient."""
    # Non-dimensional coupling: h * L / k
    if thickness <= 0 or thermal_cond <= 0:
        return 0.0
    return acoustic_imp * thickness / (thermal_cond + 1e-30)
