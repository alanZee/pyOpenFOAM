"""
createBaffles enhanced v8 — enhanced baffle creation with thermal coupling
validation, permeability modelling, and baffle optimisation feedback
(eighth generation).

Extends :func:`create_baffles_enhanced_7` with:

- **Thermal coupling validation**: Verify thermal coupling consistency
  across conjugate heat transfer interfaces.
- **Permeability modelling**: Compute Darcy-Forchheimer permeability
  coefficients for porous baffles.
- **Optimisation feedback**: Generate feedback metrics for iterative
  baffle placement optimisation.

Usage::

    from pyfoam.tools.create_baffles_enhanced_8 import create_baffles_enhanced_8

    result = create_baffles_enhanced_8(
        mesh,
        face_indices=[0, 1],
        patch_name="baffle",
        permeability_model=True,
        validate_thermal_coupling=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced8Result", "create_baffles_enhanced_8"]


@dataclass
class PermeabilityCoefficients:
    """Darcy-Forchheimer permeability model coefficients."""
    darcy_coefficient: float = 0.0
    forchheimer_coefficient: float = 0.0
    porosity: float = 0.0
    permeability: float = 0.0


@dataclass
class ThermalCouplingValidation:
    """Validation result for thermal coupling."""
    is_consistent: bool = True
    max_temperature_jump: float = 0.0
    n_inconsistent_interfaces: int = 0
    warnings: list = field(default_factory=list)


@dataclass
class OptimisationFeedback:
    """Feedback metrics for baffle placement optimisation."""
    mean_flow_resistance: float = 0.0
    area_weighted_resistance: float = 0.0
    effectiveness_score: float = 0.0
    suggestions: list = field(default_factory=list)


@dataclass
class BaffleEnhanced8Result:
    """Result from :func:`create_baffles_enhanced_8`.

    Attributes
    ----------
    mesh .. lifecycle_events
        Forwarded from v7.
    permeability : PermeabilityCoefficients
        Darcy-Forchheimer model coefficients.
    thermal_validation : ThermalCouplingValidation
        Thermal coupling validation result.
    optimisation_feedback : OptimisationFeedback
        Placement optimisation metrics.
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
    permeability: PermeabilityCoefficients = field(default_factory=PermeabilityCoefficients)
    thermal_validation: ThermalCouplingValidation = field(default_factory=ThermalCouplingValidation)
    optimisation_feedback: OptimisationFeedback = field(default_factory=OptimisationFeedback)

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_8(
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
) -> BaffleEnhanced8Result:
    """Create baffles with permeability modelling and thermal validation.

    Parameters
    ----------
    mesh .. track_lifecycle
        Forwarded to v7 baffle creation.
    permeability_model : bool
        Compute Darcy-Forchheimer permeability coefficients.
    validate_thermal_coupling : bool
        Verify thermal coupling consistency.
    compute_feedback : bool
        Generate optimisation feedback metrics.

    Returns
    -------
    BaffleEnhanced8Result
    """
    from pyfoam.tools.create_baffles_enhanced_7 import create_baffles_enhanced_7

    v7_result = create_baffles_enhanced_7(
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
        acoustic_impedance=acoustic_impedance,
        baffle_thickness=baffle_thickness,
        coupling_enabled=coupling_enabled,
        track_lifecycle=track_lifecycle,
    )

    # Permeability modelling
    perm = PermeabilityCoefficients()
    if permeability_model:
        perm = _compute_permeability(
            porosity, pressure_drop_coefficient, baffle_thickness,
        )

    # Thermal coupling validation
    thermal_val = ThermalCouplingValidation()
    if validate_thermal_coupling:
        thermal_val = _validate_thermal_coupling(
            v7_result.thermal_resistance, v7_result.thermal_conductance,
            baffle_thickness,
        )

    # Optimisation feedback
    feedback = OptimisationFeedback()
    if compute_feedback:
        feedback = _compute_optimisation_feedback(
            v7_result.n_baffles, v7_result.total_baffle_area,
            v7_result.pressure_drop_coefficient, v7_result.quality_degradation,
        )

    return BaffleEnhanced8Result(
        mesh=v7_result.mesh,
        n_baffles=v7_result.n_baffles,
        baffle_patches=v7_result.baffle_patches,
        n_filtered=v7_result.n_filtered,
        zone_face_counts=v7_result.zone_face_counts,
        total_baffle_area=v7_result.total_baffle_area,
        mean_thickness=v7_result.mean_thickness,
        porosity=porosity,
        pressure_drop_coefficient=pressure_drop_coefficient,
        thermal_resistance=thermal_resistance,
        networks=v7_result.networks,
        n_networks=v7_result.n_networks,
        quality_degradation=v7_result.quality_degradation,
        spatial_resistance=v7_result.spatial_resistance,
        thermal_conductance=v7_result.thermal_conductance,
        n_optimised=v7_result.n_optimised,
        schedule=v7_result.schedule,
        dict_snippet=v7_result.dict_snippet,
        acoustic_impedance=v7_result.acoustic_impedance,
        transmission_loss_db=v7_result.transmission_loss_db,
        coupling_coefficient=v7_result.coupling_coefficient,
        lifecycle_events=v7_result.lifecycle_events,
        permeability=perm,
        thermal_validation=thermal_val,
        optimisation_feedback=feedback,
    )


# ---------------------------------------------------------------------------
# Permeability modelling
# ---------------------------------------------------------------------------


def _compute_permeability(porosity, dp_coeff, thickness):
    """Compute Darcy-Forchheimer permeability coefficients."""
    if porosity <= 0 or porosity >= 1.0:
        return PermeabilityCoefficients()

    # Ergun equation approximation
    darcy = 150.0 * (1.0 - porosity) ** 2 / (porosity ** 3 + 1e-30)
    forchheimer = 1.75 * (1.0 - porosity) / (porosity ** 3 + 1e-30)
    permeability = porosity ** 3 / (150.0 * (1.0 - porosity) ** 2 + 1e-30)

    return PermeabilityCoefficients(
        darcy_coefficient=darcy,
        forchheimer_coefficient=forchheimer,
        porosity=porosity,
        permeability=permeability,
    )


# ---------------------------------------------------------------------------
# Thermal coupling validation
# ---------------------------------------------------------------------------


def _validate_thermal_coupling(thermal_res, thermal_cond, thickness):
    """Validate thermal coupling consistency."""
    warnings = []
    is_consistent = True
    max_t_jump = 0.0
    n_inconsistent = 0

    if thermal_res > 0 and thermal_cond > 0:
        # Expected temperature jump: q * R_thermal
        expected_jump = thermal_res * 1000.0  # assume 1 kW/m2 heat flux
        max_t_jump = expected_jump
        if expected_jump > 100.0:
            warnings.append("Large temperature jump across baffle")
            n_inconsistent = 1
            is_consistent = False

    if thickness <= 0:
        warnings.append("Zero or negative thickness")
        is_consistent = False

    return ThermalCouplingValidation(
        is_consistent=is_consistent,
        max_temperature_jump=max_t_jump,
        n_inconsistent_interfaces=n_inconsistent,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Optimisation feedback
# ---------------------------------------------------------------------------


def _compute_optimisation_feedback(n_baffles, area, dp_coeff, quality_deg):
    """Generate optimisation feedback metrics."""
    suggestions = []
    mean_resist = dp_coeff / max(n_baffles, 1)
    area_weighted = dp_coeff * area

    # Effectiveness score
    effectiveness = max(0.0, 1.0 - quality_deg) * min(1.0, area / 10.0)

    if quality_deg > 0.5:
        suggestions.append("Consider reducing baffle count to improve mesh quality")
    if n_baffles > 0 and area < 0.01:
        suggestions.append("Baffle area is very small; consider consolidating baffles")
    if effectiveness < 0.3:
        suggestions.append("Low effectiveness; review baffle placement strategy")

    return OptimisationFeedback(
        mean_flow_resistance=mean_resist,
        area_weighted_resistance=area_weighted,
        effectiveness_score=effectiveness,
        suggestions=suggestions,
    )
