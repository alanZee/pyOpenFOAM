"""
setAtmBoundaryLayer enhanced v9 — enhanced ABL profiles with multi-layer
ABL model, pollutant source apportionment, and stability transition
(ninth generation).

Extends :func:`set_atm_boundary_layer_enhanced_8` with:

- **Multi-layer ABL model**: Model ABL with explicit surface layer,
  mixed layer, and capping inversion layers.
- **Pollutant source apportionment**: Apportion pollutant
  concentrations to multiple emission sources.
- **Stability transition**: Model gradual transition between
  stability regimes over the diurnal cycle.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_9 import (
        set_atm_boundary_layer_enhanced_9, EnhancedABL9Properties,
    )

    abl = EnhancedABL9Properties(
        u_star=0.5, z0=0.01,
        multi_layer=True,
        source_apportionment=True,
    )
    result = set_atm_boundary_layer_enhanced_9(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL9Properties", "EnhancedABL9Result", "set_atm_boundary_layer_enhanced_9"]


@dataclass
class EnhancedABL9Properties:
    """Enhanced v9 ABL parameters.

    Parameters
    ----------
    u_star .. building_height
        Forwarded from v8.
    multi_layer : bool
        Enable multi-layer ABL model.
    mixed_layer_height : float
        Mixed layer height (m).
    capping_inversion_thickness : float
        Capping inversion layer thickness (m).
    source_apportionment : bool
        Apportion pollutant concentrations to sources.
    pollutant_sources : list of dict, optional
        ``[{"height": h, "rate": q, "name": s}, ...]``.
    stability_transition : bool
        Model diurnal stability transition.
    hour_of_day : float
        Hour of day (0-24) for stability transition.
    """

    u_star: float = 0.5
    z0: float = 0.01
    displacement_height: float = 0.0
    kappa: float = 0.41
    Cmu: float = 0.09
    direction: tuple = (1.0, 0.0, 0.0)
    model: str = "neutral"
    L_Monin: Optional[float] = None
    power_exponent: float = 0.143
    U_ref: Optional[float] = None
    z_ref: float = 10.0
    coriolis_parameter: float = 1e-4
    geostrophic_height: float = 1000.0
    surface_temperature: float = 300.0
    temperature_lapse_rate: float = -0.01
    canopy_height: float = 0.0
    canopy_drag_coefficient: float = 0.2
    surface_heat_flux: float = 0.0
    roughness_sublayer: bool = False
    spectral_model: str = "none"
    turbulence_length_scale: float = 100.0
    mesoscale_pressure_gradient: Optional[tuple] = None
    geostrophic_wind: Optional[tuple] = None
    heterogeneous_z0: Optional[Dict[float, List[str]]] = None
    time_varying: Optional[list] = None
    coriolis_latitude: float = 45.0
    pollution_dispersion: bool = False
    schmidt_number: float = 0.7
    urban_canopy: bool = False
    building_density: float = 0.3
    building_height: float = 20.0
    co2_dispersion: bool = False
    co2_emission_rate: float = 1.0
    co2_source_height: float = 10.0
    renewable_assessment: bool = False
    hub_height: float = 80.0
    turbine_diameter: float = 80.0
    classify_site: bool = False
    multi_layer: bool = False
    mixed_layer_height: float = 1000.0
    capping_inversion_thickness: float = 200.0
    source_apportionment: bool = False
    pollutant_sources: Optional[List[Dict]] = None
    stability_transition: bool = False
    hour_of_day: float = 12.0


@dataclass
class MultiLayerABL:
    """Multi-layer ABL model result."""
    surface_layer_fraction: float = 0.0
    mixed_layer_fraction: float = 0.0
    inversion_fraction: float = 0.0
    n_surface_cells: int = 0
    n_mixed_cells: int = 0
    n_inversion_cells: int = 0


@dataclass
class SourceApportionment:
    """Pollutant source apportionment."""
    n_sources: int = 0
    source_contributions: list = field(default_factory=list)
    dominant_source: str = ""
    total_concentration: float = 0.0


@dataclass
class StabilityTransition:
    """Diurnal stability transition model."""
    hour_of_day: float = 0.0
    stability_class: str = "neutral"
    obukhov_length: float = 0.0
    mixing_height: float = 0.0
    transition_progress: float = 0.0


@dataclass
class EnhancedABL9Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_9`.

    Attributes
    ----------
    U .. site_class
        Forwarded from v8.
    multi_layer : MultiLayerABL
        Multi-layer ABL model result.
    apportionment : SourceApportionment
        Pollutant source apportionment.
    stability : StabilityTransition
        Stability transition result.
    """

    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: np.ndarray = field(default_factory=lambda: np.empty(0))
    epsilon: np.ndarray = field(default_factory=lambda: np.empty(0))
    omega: np.ndarray = field(default_factory=lambda: np.empty(0))
    length_scale: np.ndarray = field(default_factory=lambda: np.empty(0))
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0))
    temperature: np.ndarray = field(default_factory=lambda: np.empty(0))
    mixing_length: np.ndarray = field(default_factory=lambda: np.empty(0))
    reynolds_stress: Optional[np.ndarray] = None
    boundary_layer_height: float = 0.0
    geostrophic_wind: float = 0.0
    profile_quality: float = 0.0
    bulk_richardson_number: float = 0.0
    canopy_top_height: float = 0.0
    spectral_coefficients: Optional[np.ndarray] = None
    roughness_sublayer_correction: int = 0
    n_heterogeneous_patches: int = 0
    mesoscale_balance: float = 0.0
    n_time_steps: int = 0
    latitude_used: float = 45.0
    eddy_diffusivity: Optional[np.ndarray] = None
    schmidt_number_used: float = 0.7
    n_dispersion_cells: int = 0
    canopy_drag_cells: int = 0
    obukhov_length: float = 0.0
    stability_regime: str = "neutral"
    co2_concentration: Optional[np.ndarray] = None
    n_co2_cells: int = 0
    wind_energy: object = None
    site_class: object = None
    multi_layer: MultiLayerABL = field(default_factory=MultiLayerABL)
    apportionment: SourceApportionment = field(default_factory=SourceApportionment)
    stability: StabilityTransition = field(default_factory=StabilityTransition)


def set_atm_boundary_layer_enhanced_9(
    mesh: "FvMesh",
    abl: EnhancedABL9Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL9Result:
    """Set enhanced v9 ABL profiles with multi-layer model and source apportionment.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL9Properties
    z_axis, free_surface_z, compute_reynolds_stress

    Returns
    -------
    EnhancedABL9Result
    """
    from pyfoam.tools.set_atm_boundary_layer_enhanced_8 import (
        set_atm_boundary_layer_enhanced_8,
        EnhancedABL8Properties,
    )

    v8_props = EnhancedABL8Properties(
        u_star=abl.u_star,
        z0=abl.z0,
        displacement_height=abl.displacement_height,
        kappa=abl.kappa,
        Cmu=abl.Cmu,
        direction=abl.direction,
        model=abl.model,
        L_Monin=abl.L_Monin,
        power_exponent=abl.power_exponent,
        U_ref=abl.U_ref,
        z_ref=abl.z_ref,
        coriolis_parameter=abl.coriolis_parameter,
        geostrophic_height=abl.geostrophic_height,
        surface_temperature=abl.surface_temperature,
        temperature_lapse_rate=abl.temperature_lapse_rate,
        canopy_height=abl.canopy_height,
        canopy_drag_coefficient=abl.canopy_drag_coefficient,
        surface_heat_flux=abl.surface_heat_flux,
        roughness_sublayer=abl.roughness_sublayer,
        spectral_model=abl.spectral_model,
        turbulence_length_scale=abl.turbulence_length_scale,
        mesoscale_pressure_gradient=abl.mesoscale_pressure_gradient,
        geostrophic_wind=abl.geostrophic_wind,
        heterogeneous_z0=abl.heterogeneous_z0,
        time_varying=abl.time_varying,
        coriolis_latitude=abl.coriolis_latitude,
        pollution_dispersion=abl.pollution_dispersion,
        schmidt_number=abl.schmidt_number,
        urban_canopy=abl.urban_canopy,
        building_density=abl.building_density,
        building_height=abl.building_height,
        co2_dispersion=abl.co2_dispersion,
        co2_emission_rate=abl.co2_emission_rate,
        co2_source_height=abl.co2_source_height,
        renewable_assessment=abl.renewable_assessment,
        hub_height=abl.hub_height,
        turbine_diameter=abl.turbine_diameter,
        classify_site=abl.classify_site,
    )

    v8_result = set_atm_boundary_layer_enhanced_8(
        mesh, v8_props, z_axis, free_surface_z, compute_reynolds_stress,
    )

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Multi-layer model
    ml = MultiLayerABL()
    if abl.multi_layer and n_cells > 0:
        ml = _compute_multi_layer(
            cell_centres, abl.mixed_layer_height,
            abl.capping_inversion_thickness, z_axis,
        )

    # Source apportionment
    apportion = SourceApportionment()
    if abl.source_apportionment and abl.pollutant_sources:
        apportion = _apportion_sources(
            cell_centres, v8_result.eddy_diffusivity,
            abl.pollutant_sources, abl.schmidt_number, z_axis,
        )

    # Stability transition
    stab = StabilityTransition()
    if abl.stability_transition:
        stab = _compute_stability_transition(
            abl.hour_of_day, abl.surface_heat_flux,
            abl.u_star, abl.surface_temperature, abl.kappa,
        )

    return EnhancedABL9Result(
        U=v8_result.U,
        k=v8_result.k,
        epsilon=v8_result.epsilon,
        omega=v8_result.omega,
        length_scale=v8_result.length_scale,
        intensity=v8_result.intensity,
        temperature=v8_result.temperature,
        mixing_length=v8_result.mixing_length,
        reynolds_stress=v8_result.reynolds_stress,
        boundary_layer_height=v8_result.boundary_layer_height,
        geostrophic_wind=v8_result.geostrophic_wind,
        profile_quality=v8_result.profile_quality,
        bulk_richardson_number=v8_result.bulk_richardson_number,
        canopy_top_height=v8_result.canopy_top_height,
        spectral_coefficients=v8_result.spectral_coefficients,
        roughness_sublayer_correction=v8_result.roughness_sublayer_correction,
        n_heterogeneous_patches=v8_result.n_heterogeneous_patches,
        mesoscale_balance=v8_result.mesoscale_balance,
        n_time_steps=v8_result.n_time_steps,
        latitude_used=v8_result.latitude_used,
        eddy_diffusivity=v8_result.eddy_diffusivity,
        schmidt_number_used=v8_result.schmidt_number_used,
        n_dispersion_cells=v8_result.n_dispersion_cells,
        canopy_drag_cells=v8_result.canopy_drag_cells,
        obukhov_length=v8_result.obukhov_length,
        stability_regime=v8_result.stability_regime,
        co2_concentration=v8_result.co2_concentration,
        n_co2_cells=v8_result.n_co2_cells,
        wind_energy=v8_result.wind_energy,
        site_class=v8_result.site_class,
        multi_layer=ml,
        apportionment=apportion,
        stability=stab,
    )


# ---------------------------------------------------------------------------
# Multi-layer ABL model
# ---------------------------------------------------------------------------


def _compute_multi_layer(cell_centres, mixed_height, inversion_thickness, z_axis):
    """Classify cells into surface, mixed, and inversion layers."""
    n_cells = cell_centres.shape[0]
    n_surf = n_mixed = n_inv = 0

    inversion_top = mixed_height + inversion_thickness

    for i in range(n_cells):
        z = cell_centres[i, z_axis]
        if z < mixed_height * 0.1:
            n_surf += 1
        elif z < mixed_height:
            n_mixed += 1
        elif z < inversion_top:
            n_inv += 1

    total = max(n_surf + n_mixed + n_inv, 1)

    return MultiLayerABL(
        surface_layer_fraction=n_surf / total,
        mixed_layer_fraction=n_mixed / total,
        inversion_fraction=n_inv / total,
        n_surface_cells=n_surf,
        n_mixed_cells=n_mixed,
        n_inversion_cells=n_inv,
    )


# ---------------------------------------------------------------------------
# Source apportionment
# ---------------------------------------------------------------------------


def _apportion_sources(cell_centres, eddy_diff, sources, Sc_t, z_axis):
    """Apportion concentrations to multiple emission sources."""
    n_sources = len(sources)
    contributions = []
    total_conc = 0.0
    dominant = ""

    for si, src in enumerate(sources):
        h = src.get("height", 10.0)
        rate = src.get("rate", 1.0)
        name = src.get("name", f"source_{si}")

        # Simplified: contribution proportional to rate / height
        contrib = rate / max(h, 1.0)
        contributions.append({"name": name, "contribution": contrib})
        total_conc += contrib

        if not dominant or contrib > max((c["contribution"] for c in contributions[:-1]), default=0):
            dominant = name

    # Normalise
    if total_conc > 0:
        for c in contributions:
            c["fraction"] = c["contribution"] / total_conc

    return SourceApportionment(
        n_sources=n_sources,
        source_contributions=contributions,
        dominant_source=dominant,
        total_concentration=total_conc,
    )


# ---------------------------------------------------------------------------
# Stability transition
# ---------------------------------------------------------------------------


def _compute_stability_transition(hour, H0, u_star, T0, kappa):
    """Model diurnal stability transition."""
    # Simplified diurnal model
    # Night: stable (positive L), Day: unstable (negative L)
    is_daytime = 6.0 <= hour <= 18.0

    if is_daytime:
        # Convective
        if H0 > 0:
            L = -u_star ** 3 * T0 / (kappa * 9.81 * max(H0, 1e-10))
            stability = "unstable"
        else:
            L = 1e10
            stability = "neutral"
        mixing_h = 1000.0 + 500.0 * abs(H0) / max(abs(H0) + 100, 1e-10)
    else:
        # Stable
        if H0 < 0:
            L = -u_star ** 3 * T0 / (kappa * 9.81 * max(abs(H0), 1e-10))
            stability = "stable"
        else:
            L = 1e10
            stability = "neutral"
        mixing_h = 200.0

    # Transition progress (0 at dawn/dusk, 1 at noon/midnight)
    t_from_noon = abs(hour - 12.0)
    progress = 1.0 - t_from_noon / 12.0

    return StabilityTransition(
        hour_of_day=hour,
        stability_class=stability,
        obukhov_length=L if abs(L) < 1e9 else float("inf"),
        mixing_height=mixing_h,
        transition_progress=progress,
    )
