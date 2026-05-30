"""
setAtmBoundaryLayer enhanced v8 — enhanced ABL profiles with CO2 dispersion
modelling, renewable energy assessment, and site classification
(eighth generation).

Extends :func:`set_atm_boundary_layer_enhanced_7` with:

- **CO2 dispersion modelling**: Compute CO2 concentration fields
  from source emissions with atmospheric dispersion.
- **Renewable energy assessment**: Estimate wind power density
  and capacity factor at hub height.
- **Site classification**: Classify the terrain according to
  Eurocode/WMO standards based on roughness and exposure.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_8 import (
        set_atm_boundary_layer_enhanced_8, EnhancedABL8Properties,
    )

    abl = EnhancedABL8Properties(
        u_star=0.5, z0=0.01,
        co2_dispersion=True,
        renewable_assessment=True,
    )
    result = set_atm_boundary_layer_enhanced_8(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL8Properties", "EnhancedABL8Result", "set_atm_boundary_layer_enhanced_8"]


@dataclass
class EnhancedABL8Properties:
    """Enhanced v8 ABL parameters.

    Parameters
    ----------
    u_star .. building_height
        Forwarded from v7.
    co2_dispersion : bool
        Enable CO2 dispersion modelling.
    co2_emission_rate : float
        CO2 emission rate (kg/s).
    co2_source_height : float
        Source emission height (m).
    renewable_assessment : bool
        Enable wind energy assessment.
    hub_height : float
        Wind turbine hub height (m).
    turbine_diameter : float
        Rotor diameter (m).
    classify_site : bool
        Classify terrain per standards.
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


@dataclass
class WindEnergyMetrics:
    """Wind energy assessment metrics."""
    wind_power_density: float = 0.0
    capacity_factor: float = 0.0
    mean_wind_speed_hub: float = 0.0
    weibull_k: float = 2.0
    weibull_a: float = 0.0


@dataclass
class SiteClassification:
    """Terrain site classification."""
    eurocode_category: str = ""
    wmo_exposure: str = ""
    roughness_length_used: float = 0.0
    description: str = ""


@dataclass
class EnhancedABL8Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_8`.

    Attributes
    ----------
    U .. stability_regime
        Forwarded from v7.
    co2_concentration : np.ndarray, optional
        CO2 concentration field (ppm).
    n_co2_cells : int
        Cells with CO2 concentration computed.
    wind_energy : WindEnergyMetrics
        Wind energy assessment metrics.
    site_class : SiteClassification
        Terrain site classification.
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
    wind_energy: WindEnergyMetrics = field(default_factory=WindEnergyMetrics)
    site_class: SiteClassification = field(default_factory=SiteClassification)


def set_atm_boundary_layer_enhanced_8(
    mesh: "FvMesh",
    abl: EnhancedABL8Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL8Result:
    """Set enhanced v8 ABL profiles with CO2 dispersion and wind energy.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL8Properties
    z_axis, free_surface_z, compute_reynolds_stress

    Returns
    -------
    EnhancedABL8Result
    """
    from pyfoam.tools.set_atm_boundary_layer_enhanced_7 import (
        set_atm_boundary_layer_enhanced_7,
        EnhancedABL7Properties,
    )

    v7_props = EnhancedABL7Properties(
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
    )

    v7_result = set_atm_boundary_layer_enhanced_7(
        mesh, v7_props, z_axis, free_surface_z, compute_reynolds_stress,
    )

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # CO2 dispersion
    co2_conc = None
    n_co2 = 0
    if abl.co2_dispersion and n_cells > 0:
        co2_conc, n_co2 = _compute_co2_concentration(
            cell_centres, v7_result.U, v7_result.eddy_diffusivity,
            abl.co2_emission_rate, abl.co2_source_height,
            abl.schmidt_number, z_axis,
        )

    # Wind energy assessment
    wind_energy = WindEnergyMetrics()
    if abl.renewable_assessment and n_cells > 0:
        wind_energy = _assess_wind_energy(
            cell_centres, v7_result.U, abl.hub_height,
            abl.u_star, abl.kappa, abl.z0,
            abl.displacement_height, z_axis,
        )

    # Site classification
    site_class = SiteClassification()
    if abl.classify_site:
        site_class = _classify_site(abl.z0)

    return EnhancedABL8Result(
        U=v7_result.U,
        k=v7_result.k,
        epsilon=v7_result.epsilon,
        omega=v7_result.omega,
        length_scale=v7_result.length_scale,
        intensity=v7_result.intensity,
        temperature=v7_result.temperature,
        mixing_length=v7_result.mixing_length,
        reynolds_stress=v7_result.reynolds_stress,
        boundary_layer_height=v7_result.boundary_layer_height,
        geostrophic_wind=v7_result.geostrophic_wind,
        profile_quality=v7_result.profile_quality,
        bulk_richardson_number=v7_result.bulk_richardson_number,
        canopy_top_height=v7_result.canopy_top_height,
        spectral_coefficients=v7_result.spectral_coefficients,
        roughness_sublayer_correction=v7_result.roughness_sublayer_correction,
        n_heterogeneous_patches=v7_result.n_heterogeneous_patches,
        mesoscale_balance=v7_result.mesoscale_balance,
        n_time_steps=v7_result.n_time_steps,
        latitude_used=v7_result.latitude_used,
        eddy_diffusivity=v7_result.eddy_diffusivity,
        schmidt_number_used=v7_result.schmidt_number_used,
        n_dispersion_cells=v7_result.n_dispersion_cells,
        canopy_drag_cells=v7_result.canopy_drag_cells,
        obukhov_length=v7_result.obukhov_length,
        stability_regime=v7_result.stability_regime,
        co2_concentration=co2_conc,
        n_co2_cells=n_co2,
        wind_energy=wind_energy,
        site_class=site_class,
    )


# ---------------------------------------------------------------------------
# CO2 dispersion
# ---------------------------------------------------------------------------


def _compute_co2_concentration(cell_centres, velocity, eddy_diff,
                                emission_rate, source_height, Sc_t, z_axis):
    """Compute CO2 concentration from Gaussian plume model (simplified)."""
    n_cells = cell_centres.shape[0]
    co2 = np.zeros(n_cells, dtype=np.float64)
    n_mod = 0
    rho_air = 1.225  # kg/m3

    for ci in range(n_cells):
        z = cell_centres[ci, z_axis]
        dz = abs(z - source_height)
        speed = np.linalg.norm(velocity[ci]) if ci < velocity.shape[0] else 0.0

        # Simplified Gaussian: C = Q / (2*pi*sigma_y*sigma_z*u)
        sigma_z = max(1.0, dz * 0.1 + 1.0)
        u_eff = max(speed, 0.1)
        conc = emission_rate / (2.0 * math.pi * sigma_z * sigma_z * u_eff + 1e-30)
        co2[ci] = conc * 1e6 / rho_air  # convert to ppm
        n_mod += 1

    return co2, n_mod


# ---------------------------------------------------------------------------
# Wind energy assessment
# ---------------------------------------------------------------------------


def _assess_wind_energy(cell_centres, velocity, hub_height,
                         u_star, kappa, z0, d, z_axis):
    """Assess wind power density at hub height."""
    # Mean wind speed at hub height (log law)
    z_eff = hub_height - d
    if z_eff <= z0 or z0 <= 0:
        return WindEnergyMetrics()

    u_hub = (u_star / kappa) * math.log(z_eff / z0 + 1e-30)

    # Wind power density: 0.5 * rho * U^3
    rho = 1.225
    wpd = 0.5 * rho * u_hub ** 3

    # Capacity factor (simplified: assume cut-in 3, rated 12, cut-out 25 m/s)
    if u_hub < 3.0:
        cf = 0.0
    elif u_hub > 25.0:
        cf = 0.0
    elif u_hub > 12.0:
        cf = 1.0
    else:
        cf = (u_hub - 3.0) / 9.0

    return WindEnergyMetrics(
        wind_power_density=wpd,
        capacity_factor=cf,
        mean_wind_speed_hub=u_hub,
        weibull_k=2.0,
        weibull_a=u_hub * 2.0 / math.sqrt(math.pi),
    )


# ---------------------------------------------------------------------------
# Site classification
# ---------------------------------------------------------------------------


def _classify_site(z0):
    """Classify terrain per Eurocode/WMO standards."""
    if z0 >= 1.0:
        cat = "IV"
        exposure = "rough"
        desc = "Urban/suburban areas with dense buildings"
    elif z0 >= 0.3:
        cat = "III"
        exposure = "rough"
        desc = "Suburban areas, industrial zones, forests"
    elif z0 >= 0.05:
        cat = "II"
        exposure = "open"
        desc = "Agricultural land with low vegetation"
    elif z0 >= 0.003:
        cat = "I"
        exposure = "open"
        desc = "Flat terrain with few obstacles"
    else:
        cat = "0"
        exposure = "sea"
        desc = "Sea, coastal areas"

    return SiteClassification(
        eurocode_category=f"Category {cat}",
        wmo_exposure=exposure,
        roughness_length_used=z0,
        description=desc,
    )
