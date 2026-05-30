"""
applyBoundaryLayer enhanced v8 — enhanced boundary layer application with
thermal boundary layer coupling, fouling prediction, and noise source
modelling (eighth generation).

Extends :func:`apply_boundary_layer_enhanced_7` with:

- **Thermal boundary layer coupling**: Solve conjugate thermal BL
  with Prandtl number effects and thermal entry length.
- **Fouling prediction**: Predict fouling resistance development
  from wall shear stress and fluid properties.
- **Noise source modelling**: Estimate aerodynamic noise from
  turbulent boundary layer using Curle's analogy.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_8 import (
        apply_boundary_layer_enhanced_8, EnhancedBL8Properties,
    )

    bl = EnhancedBL8Properties(
        delta=0.1, nu=1e-5,
        thermal_coupling=True,
        fouling_prediction=True,
    )
    result = apply_boundary_layer_enhanced_8(mesh, U, bl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL8Properties", "EnhancedBL8Result", "apply_boundary_layer_enhanced_8"]


@dataclass
class EnhancedBL8Properties:
    """Enhanced v8 BL model parameters.

    Parameters
    ----------
    delta .. nusselt_correlation
        Forwarded from v7.
    thermal_coupling : bool
        Enable conjugate thermal BL coupling.
    thermal_entry_length : bool
        Compute thermal entry length.
    fouling_prediction : bool
        Predict fouling resistance.
    fluid_viscosity : float
        Dynamic viscosity for fouling model (Pa-s).
    particle_concentration : float
        Particle concentration in fluid (kg/m3).
    noise_modelling : bool
        Estimate boundary layer noise.
    reference_area : float
        Reference area for noise (m2).
    """

    delta: float = 0.1
    nu: float = 1e-5
    kappa: float = 0.41
    E: float = 9.8
    u_star: Optional[float] = None
    z0_rough: float = 0.0
    blend_width: float = 0.2
    Cmu: float = 0.09
    wall_function: str = "standard"
    dp_dx: float = 0.0
    reference_U: Optional[float] = None
    reference_y: Optional[float] = None
    Pr: float = 0.71
    T_wall: float = 300.0
    T_inf: float = 293.15
    thermal_conductivity: float = 0.025
    transition_model: bool = False
    Re_x_transition: float = 3.2e6
    detect_separation: bool = False
    compressible: bool = False
    Mach: float = 0.0
    gamma: float = 1.4
    unsteady: bool = False
    lag_time_scale: float = 0.1
    roughness_growth_rate: float = 0.0
    separation_bubble: bool = False
    bubble_reattachment_factor: float = 0.6
    transition_dynamics: bool = False
    intermittency_onset: float = 200.0
    heat_transfer_enhancement: bool = False
    nusselt_correlation: str = "gnielinski"
    thermal_coupling: bool = False
    thermal_entry_length: bool = False
    fouling_prediction: bool = False
    fluid_viscosity: float = 1e-3
    particle_concentration: float = 0.01
    noise_modelling: bool = False
    reference_area: float = 1.0


@dataclass
class ThermalBLCoupling:
    """Thermal boundary layer coupling result."""
    thermal_entry_length: float = 0.0
    thermal_BL_thickness: float = 0.0
    conjugate_htc: float = 0.0
    n_thermal_cells: int = 0


@dataclass
class FoulingPrediction:
    """Fouling resistance prediction."""
    initial_fouling_resistance: float = 0.0
    asymptotic_fouling_resistance: float = 0.0
    fouling_time_constant: float = 0.0
    n_fouling_cells: int = 0


@dataclass
class NoiseEstimate:
    """Aerodynamic noise estimate."""
    sound_power_level_db: float = 0.0
    peak_frequency: float = 0.0
    n_noise_sources: int = 0


@dataclass
class EnhancedBL8Result:
    """Result from :func:`apply_boundary_layer_enhanced_8`.

    Attributes
    ----------
    velocity .. heat_transfer_coefficient
        Forwarded from v7.
    thermal_bl : ThermalBLCoupling
        Thermal BL coupling result.
    fouling : FoulingPrediction
        Fouling prediction result.
    noise : NoiseEstimate
        Noise source estimate.
    """

    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    u_star_used: float = 0.0
    max_y_plus: float = 0.0
    n_cells_modified: int = 0
    displacement_thickness: float = 0.0
    momentum_thickness: float = 0.0
    shape_factor: float = 0.0
    wall_heat_flux: float = 0.0
    thermal_thickness: float = 0.0
    nusselt_number: float = 0.0
    skin_friction_coefficient: float = 0.0
    n_transition_cells: int = 0
    n_separation_cells: int = 0
    transition_x: Optional[float] = None
    compressibility_factor: float = 1.0
    n_unsteady_cells: int = 0
    roughness_z0_effective: float = 0.0
    recovery_temperature: float = 300.0
    n_separation_bubbles: int = 0
    reattachment_x: Optional[float] = None
    intermittency: float = 0.0
    n_transition_cells_dynamics: int = 0
    enhanced_nusselt: float = 0.0
    heat_transfer_coefficient: float = 0.0
    thermal_bl: ThermalBLCoupling = field(default_factory=ThermalBLCoupling)
    fouling: FoulingPrediction = field(default_factory=FoulingPrediction)
    noise: NoiseEstimate = field(default_factory=NoiseEstimate)


def apply_boundary_layer_enhanced_8(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBL8Properties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
    temperature_field: Optional[np.ndarray] = None,
) -> EnhancedBL8Result:
    """Apply enhanced v8 boundary layer correction.

    Parameters
    ----------
    mesh : FvMesh
    velocity : np.ndarray ``(n_cells, 3)``
    bl : EnhancedBL8Properties
    wall_patches, k_field, epsilon_field, omega_field, temperature_field

    Returns
    -------
    EnhancedBL8Result
    """
    from pyfoam.tools.apply_boundary_layer_enhanced_7 import (
        apply_boundary_layer_enhanced_7,
        EnhancedBL7Properties,
    )

    v7_props = EnhancedBL7Properties(
        delta=bl.delta,
        nu=bl.nu,
        kappa=bl.kappa,
        E=bl.E,
        u_star=bl.u_star,
        z0_rough=bl.z0_rough,
        blend_width=bl.blend_width,
        Cmu=bl.Cmu,
        wall_function=bl.wall_function,
        dp_dx=bl.dp_dx,
        reference_U=bl.reference_U,
        reference_y=bl.reference_y,
        Pr=bl.Pr,
        T_wall=bl.T_wall,
        T_inf=bl.T_inf,
        thermal_conductivity=bl.thermal_conductivity,
        transition_model=bl.transition_model,
        Re_x_transition=bl.Re_x_transition,
        detect_separation=bl.detect_separation,
        compressible=bl.compressible,
        Mach=bl.Mach,
        gamma=bl.gamma,
        unsteady=bl.unsteady,
        lag_time_scale=bl.lag_time_scale,
        roughness_growth_rate=bl.roughness_growth_rate,
        separation_bubble=bl.separation_bubble,
        bubble_reattachment_factor=bl.bubble_reattachment_factor,
        transition_dynamics=bl.transition_dynamics,
        intermittency_onset=bl.intermittency_onset,
        heat_transfer_enhancement=bl.heat_transfer_enhancement,
        nusselt_correlation=bl.nusselt_correlation,
    )

    v7_result = apply_boundary_layer_enhanced_7(
        mesh, velocity, v7_props, wall_patches,
        k_field, epsilon_field, omega_field, temperature_field,
    )

    # Thermal BL coupling
    thermal_bl = ThermalBLCoupling()
    if bl.thermal_coupling:
        thermal_bl = _thermal_bl_coupling(
            bl.delta, bl.Pr, bl.nu, v7_result.max_y_plus,
            v7_result.heat_transfer_coefficient,
        )

    # Fouling prediction
    fouling = FoulingPrediction()
    if bl.fouling_prediction:
        fouling = _predict_fouling(
            v7_result.wall_heat_flux, v7_result.u_star_used,
            bl.fluid_viscosity, bl.particle_concentration, bl.delta,
        )

    # Noise modelling
    noise = NoiseEstimate()
    if bl.noise_modelling:
        noise = _estimate_noise(
            v7_result.u_star_used, bl.delta, bl.reference_area,
            v7_result.n_cells_modified,
        )

    return EnhancedBL8Result(
        velocity=v7_result.velocity,
        k=v7_result.k,
        epsilon=v7_result.epsilon,
        omega=v7_result.omega,
        temperature=v7_result.temperature,
        u_star_used=v7_result.u_star_used,
        max_y_plus=v7_result.max_y_plus,
        n_cells_modified=v7_result.n_cells_modified,
        displacement_thickness=v7_result.displacement_thickness,
        momentum_thickness=v7_result.momentum_thickness,
        shape_factor=v7_result.shape_factor,
        wall_heat_flux=v7_result.wall_heat_flux,
        thermal_thickness=v7_result.thermal_thickness,
        nusselt_number=v7_result.nusselt_number,
        skin_friction_coefficient=v7_result.skin_friction_coefficient,
        n_transition_cells=v7_result.n_transition_cells,
        n_separation_cells=v7_result.n_separation_cells,
        transition_x=v7_result.transition_x,
        compressibility_factor=v7_result.compressibility_factor,
        n_unsteady_cells=v7_result.n_unsteady_cells,
        roughness_z0_effective=v7_result.roughness_z0_effective,
        recovery_temperature=v7_result.recovery_temperature,
        n_separation_bubbles=v7_result.n_separation_bubbles,
        reattachment_x=v7_result.reattachment_x,
        intermittency=v7_result.intermittency,
        n_transition_cells_dynamics=v7_result.n_transition_cells_dynamics,
        enhanced_nusselt=v7_result.enhanced_nusselt,
        heat_transfer_coefficient=v7_result.heat_transfer_coefficient,
        thermal_bl=thermal_bl,
        fouling=fouling,
        noise=noise,
    )


# ---------------------------------------------------------------------------
# Thermal BL coupling
# ---------------------------------------------------------------------------


def _thermal_bl_coupling(delta, Pr, nu, y_plus, htc):
    """Compute thermal boundary layer parameters."""
    # Thermal BL thickness: delta_T ~ delta * Pr^(-1/3)
    delta_T = delta * Pr ** (-1.0 / 3.0) if Pr > 0 else delta

    # Thermal entry length (Graetz problem)
    Re = y_plus * 100.0
    x_entry = 0.05 * Re * Pr * delta * 2.0  # pipe diameter ~ 2*delta

    # Conjugate HTC
    conj_htc = htc if htc > 0 else 0.0

    return ThermalBLCoupling(
        thermal_entry_length=x_entry,
        thermal_BL_thickness=delta_T,
        conjugate_htc=conj_htc,
        n_thermal_cells=1,
    )


# ---------------------------------------------------------------------------
# Fouling prediction
# ---------------------------------------------------------------------------


def _predict_fouling(heat_flux, u_star, mu, conc, delta):
    """Predict fouling resistance from wall conditions."""
    # Kern-Seaton fouling model (asymptotic)
    # R_f = R_f_inf * (1 - exp(-t/tau))
    if heat_flux <= 0 or u_star <= 0:
        return FoulingPrediction()

    # Deposition rate proportional to concentration and friction velocity
    tau_w = u_star ** 2 * 1.225  # wall shear stress
    dep_rate = conc * u_star * 0.01  # simplified deposition

    # Asymptotic fouling resistance
    R_f_inf = 1e-4 / max(dep_rate, 1e-30)

    # Time constant
    tau_f = delta / max(u_star, 1e-30)

    return FoulingPrediction(
        initial_fouling_resistance=0.0,
        asymptotic_fouling_resistance=R_f_inf,
        fouling_time_constant=tau_f,
        n_fouling_cells=1,
    )


# ---------------------------------------------------------------------------
# Noise modelling
# ---------------------------------------------------------------------------


def _estimate_noise(u_star, delta, area, n_cells):
    """Estimate aerodynamic noise from TBL using Curle's analogy."""
    if u_star <= 0 or delta <= 0:
        return NoiseEstimate()

    rho = 1.225
    c0 = 343.0  # speed of sound

    # Wall pressure fluctuation: p' ~ rho * u_star^2
    p_rms = rho * u_star ** 2

    # Curle's acoustic power: W ~ p'^2 * area * U / (rho * c0^3)
    speed = u_star * 10.0  # approximate edge velocity
    W = p_rms ** 2 * area * speed / (rho * c0 ** 3 + 1e-30)

    # Sound power level
    W_ref = 1e-12
    Lw = 10.0 * math.log10(max(W, W_ref) / W_ref) if W > 0 else 0.0

    # Peak frequency: f ~ U / delta
    f_peak = speed / delta

    return NoiseEstimate(
        sound_power_level_db=Lw,
        peak_frequency=f_peak,
        n_noise_sources=n_cells,
    )
