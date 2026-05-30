"""
createBaffles enhanced v9 — enhanced baffle creation with baffle age tracking,
flow regime classification, and acoustic optimization
(ninth generation).

Extends :func:`create_baffles_enhanced_8` with:

- **Baffle age tracking**: Track baffle lifecycle events
  and degradation over simulated time.
- **Flow regime classification**: Classify flow regime around
  baffles (laminar, transitional, turbulent).
- **Acoustic optimization**: Optimize baffle placement for
  target transmission loss at specified frequencies.

Usage::

    from pyfoam.tools.create_baffles_enhanced_9 import create_baffles_enhanced_9

    result = create_baffles_enhanced_9(
        mesh,
        face_indices=[0, 1],
        patch_name="baffle",
        track_age=True,
        classify_flow_regime=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced9Result", "create_baffles_enhanced_9"]


@dataclass
class BaffleAge:
    """Baffle age and degradation tracking."""
    creation_step: int = 0
    current_step: int = 0
    age_seconds: float = 0.0
    degradation_factor: float = 1.0
    remaining_life_fraction: float = 1.0


@dataclass
class FlowRegime:
    """Flow regime classification around baffles."""
    regime: str = "unknown"  # laminar, transitional, turbulent
    reynolds_number: float = 0.0
    confidence: float = 0.0


@dataclass
class AcousticOptimization:
    """Acoustic optimization result."""
    target_frequency: float = 0.0
    achieved_transmission_loss_db: float = 0.0
    n_optimisation_steps: int = 0
    converged: bool = False


@dataclass
class BaffleEnhanced9Result:
    """Result from :func:`create_baffles_enhanced_9`.

    Attributes
    ----------
    mesh .. optimisation_feedback
        Forwarded from v8.
    age : BaffleAge
        Baffle age tracking info.
    flow_regime : FlowRegime
        Flow regime classification.
    acoustic_opt : AcousticOptimization
        Acoustic optimization result.
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
    permeability: object = None
    thermal_validation: object = None
    optimisation_feedback: object = None
    age: BaffleAge = field(default_factory=BaffleAge)
    flow_regime: FlowRegime = field(default_factory=FlowRegime)
    acoustic_opt: AcousticOptimization = field(default_factory=AcousticOptimization)

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_9(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], "np.ndarray", None] = None,
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
    permeability_model: bool = False,
    validate_thermal_coupling: bool = False,
    compute_feedback: bool = False,
    track_age: bool = False,
    creation_step: int = 0,
    age_seconds: float = 0.0,
    classify_flow_regime: bool = False,
    flow_velocity: float = 1.0,
    fluid_viscosity: float = 1e-5,
    optimize_acoustic: bool = False,
    target_frequency: float = 1000.0,
    target_transmission_loss: float = 20.0,
) -> BaffleEnhanced9Result:
    """Create baffles with age tracking and flow regime classification.

    Parameters
    ----------
    mesh .. compute_feedback
        Forwarded to v8 baffle creation.
    track_age : bool
        Track baffle lifecycle and degradation.
    creation_step : int
        Simulation step at creation.
    age_seconds : float
        Initial age in seconds.
    classify_flow_regime : bool
        Classify flow regime around baffles.
    flow_velocity : float
        Characteristic flow velocity (m/s).
    fluid_viscosity : float
        Kinematic viscosity (m2/s).
    optimize_acoustic : bool
        Optimize for target transmission loss.
    target_frequency : float
        Target frequency for acoustic optimization (Hz).
    target_transmission_loss : float
        Target transmission loss (dB).

    Returns
    -------
    BaffleEnhanced9Result
    """
    from pyfoam.tools.create_baffles_enhanced_8 import create_baffles_enhanced_8

    v8_result = create_baffles_enhanced_8(
        mesh=mesh,
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
        acoustic_impedance=acoustic_impedance,
        baffle_thickness=baffle_thickness,
        coupling_enabled=coupling_enabled,
        track_lifecycle=track_lifecycle,
        permeability_model=permeability_model,
        validate_thermal_coupling=validate_thermal_coupling,
        compute_feedback=compute_feedback,
    )

    # Age tracking
    age = BaffleAge()
    if track_age:
        age = _track_age(creation_step, age_seconds, baffle_thickness)

    # Flow regime classification
    regime = FlowRegime()
    if classify_flow_regime:
        regime = _classify_flow_regime(flow_velocity, baffle_thickness, fluid_viscosity)

    # Acoustic optimization
    acoustic = AcousticOptimization()
    if optimize_acoustic:
        acoustic = _optimize_acoustic(
            v8_result.transmission_loss_db, target_frequency,
            target_transmission_loss, baffle_thickness,
        )

    return BaffleEnhanced9Result(
        mesh=v8_result.mesh,
        n_baffles=v8_result.n_baffles,
        baffle_patches=v8_result.baffle_patches,
        n_filtered=v8_result.n_filtered,
        zone_face_counts=v8_result.zone_face_counts,
        total_baffle_area=v8_result.total_baffle_area,
        mean_thickness=v8_result.mean_thickness,
        porosity=porosity,
        pressure_drop_coefficient=pressure_drop_coefficient,
        thermal_resistance=thermal_resistance,
        networks=v8_result.networks,
        n_networks=v8_result.n_networks,
        quality_degradation=v8_result.quality_degradation,
        spatial_resistance=v8_result.spatial_resistance,
        thermal_conductance=v8_result.thermal_conductance,
        n_optimised=v8_result.n_optimised,
        schedule=v8_result.schedule,
        dict_snippet=v8_result.dict_snippet,
        acoustic_impedance=v8_result.acoustic_impedance,
        transmission_loss_db=v8_result.transmission_loss_db,
        coupling_coefficient=v8_result.coupling_coefficient,
        lifecycle_events=v8_result.lifecycle_events,
        permeability=v8_result.permeability,
        thermal_validation=v8_result.thermal_validation,
        optimisation_feedback=v8_result.optimisation_feedback,
        age=age,
        flow_regime=regime,
        acoustic_opt=acoustic,
    )


# ---------------------------------------------------------------------------
# Age tracking
# ---------------------------------------------------------------------------


def _track_age(creation_step, age_seconds, thickness):
    """Track baffle age and degradation."""
    # Simple degradation: linear with age
    max_life = 1e7  # arbitrary maximum lifetime in seconds
    degradation = min(1.0, age_seconds / max_life)
    remaining = max(0.0, 1.0 - degradation)

    return BaffleAge(
        creation_step=creation_step,
        current_step=creation_step,
        age_seconds=age_seconds,
        degradation_factor=1.0 - degradation,
        remaining_life_fraction=remaining,
    )


# ---------------------------------------------------------------------------
# Flow regime classification
# ---------------------------------------------------------------------------


def _classify_flow_regime(velocity, length_scale, viscosity):
    """Classify flow regime from Reynolds number."""
    if viscosity <= 0 or length_scale <= 0:
        return FlowRegime(regime="unknown", reynolds_number=0.0, confidence=0.0)

    Re = velocity * length_scale / viscosity

    if Re < 500:
        regime = "laminar"
        confidence = max(0.5, 1.0 - Re / 500)
    elif Re < 2000:
        regime = "transitional"
        confidence = 0.6
    else:
        regime = "turbulent"
        confidence = min(1.0, Re / 10000)

    return FlowRegime(
        regime=regime,
        reynolds_number=Re,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Acoustic optimization
# ---------------------------------------------------------------------------


def _optimize_acoustic(current_tl, target_freq, target_tl, thickness):
    """Optimize baffle placement for acoustic performance."""
    n_steps = 0
    achieved = current_tl
    converged = False

    # Iterative thickness adjustment (simplified)
    for i in range(10):
        n_steps += 1
        # Mass law: TL ~ 20*log10(f*m) - 47 dB
        rho = 1.225
        surface_mass = rho * thickness
        tl_estimate = 20.0 * math.log10(max(target_freq * surface_mass, 1e-10)) - 47.0
        achieved = tl_estimate

        if abs(achieved - target_tl) < 1.0:
            converged = True
            break

        # Adjust thickness towards target
        ratio = target_tl / max(achieved, 1e-10)
        thickness *= ratio ** 0.3

    return AcousticOptimization(
        target_frequency=target_freq,
        achieved_transmission_loss_db=achieved,
        n_optimisation_steps=n_steps,
        converged=converged,
    )
