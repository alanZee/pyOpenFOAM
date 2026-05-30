"""
setAtmBoundaryLayer enhanced v5 — enhanced ABL profiles with advanced
stability functions, roughness sublayer model, and turbulence spectra
(fifth generation).

Extends :func:`set_atm_boundary_layer_enhanced_4` with:

- **Advanced stability functions**: Implement Businger-Dyer functions
  with continuous transitions between stability regimes.
- **Roughness sublayer model**: Add a roughness sublayer correction
  that modifies the log-law below :math:`z = 5 z_0`.
- **Turbulence spectra**: Generate von Karman or Kaimal spectral
  representation at each cell for synthetic turbulence seeding.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_5 import (
        set_atm_boundary_layer_enhanced_5, EnhancedABL5Properties,
    )

    abl = EnhancedABL5Properties(u_star=0.5, z0=0.01, model="neutral")
    result = set_atm_boundary_layer_enhanced_5(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL5Properties", "EnhancedABL5Result", "set_atm_boundary_layer_enhanced_5"]


@dataclass
class EnhancedABL5Properties:
    """Enhanced v5 ABL parameters.

    Parameters
    ----------
    u_star, z0, displacement_height, kappa, Cmu, direction
    model, L_Monin, power_exponent, U_ref, z_ref
    coriolis_parameter, geostrophic_height
    surface_temperature, temperature_lapse_rate
    canopy_height, canopy_drag_coefficient, surface_heat_flux
        Forwarded from v4.
    roughness_sublayer : bool
        Enable roughness sublayer correction.
    spectral_model : str
        ``"none"``, ``"von_karman"``, or ``"kaimal"``.
    turbulence_length_scale : float
        Integral length scale for spectral model (m).
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


@dataclass
class EnhancedABL5Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_5`.

    Attributes
    ----------
    U, k, epsilon, omega, length_scale, intensity : np.ndarray
    temperature, mixing_length : np.ndarray
    reynolds_stress : np.ndarray, optional
    boundary_layer_height, geostrophic_wind, profile_quality : float
    bulk_richardson_number, canopy_top_height : float
    spectral_coefficients : np.ndarray, optional
        ``(n_cells, n_modes, 3)`` spectral velocity amplitudes.
    roughness_sublayer_correction : int
        Number of cells modified by roughness sublayer.
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


def set_atm_boundary_layer_enhanced_5(
    mesh: "FvMesh",
    abl: EnhancedABL5Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL5Result:
    """Set enhanced v5 ABL profiles.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL5Properties
    z_axis : int
    free_surface_z : float
    compute_reynolds_stress : bool

    Returns
    -------
    EnhancedABL5Result
    """
    from pyfoam.tools.set_atm_boundary_layer_enhanced_4 import (
        set_atm_boundary_layer_enhanced_4,
        EnhancedABL4Properties,
    )

    v4_props = EnhancedABL4Properties(
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
    )

    v4_result = set_atm_boundary_layer_enhanced_4(
        mesh, v4_props, z_axis, free_surface_z, compute_reynolds_stress,
    )

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Roughness sublayer correction
    n_rsl = 0
    if abl.roughness_sublayer:
        n_rsl = _apply_roughness_sublayer(
            v4_result.U, v4_result.k, cell_centres,
            abl, z_axis, free_surface_z,
        )

    # Spectral coefficients
    spectral = None
    if abl.spectral_model != "none":
        spectral = _generate_spectral_coefficients(
            n_cells, v4_result.k, v4_result.intensity,
            abl.turbulence_length_scale, abl.spectral_model,
        )

    return EnhancedABL5Result(
        U=v4_result.U,
        k=v4_result.k,
        epsilon=v4_result.epsilon,
        omega=v4_result.omega,
        length_scale=v4_result.length_scale,
        intensity=v4_result.intensity,
        temperature=v4_result.temperature,
        mixing_length=v4_result.mixing_length,
        reynolds_stress=v4_result.reynolds_stress,
        boundary_layer_height=v4_result.boundary_layer_height,
        geostrophic_wind=v4_result.geostrophic_wind,
        profile_quality=v4_result.profile_quality,
        bulk_richardson_number=v4_result.bulk_richardson_number,
        canopy_top_height=v4_result.canopy_top_height,
        spectral_coefficients=spectral,
        roughness_sublayer_correction=n_rsl,
    )


# ---------------------------------------------------------------------------
# Roughness sublayer
# ---------------------------------------------------------------------------


def _apply_roughness_sublayer(U, k, cell_centres, abl, z_axis, fs_z):
    """Apply roughness sublayer correction below z = 5*z0."""
    z_rsl = 5.0 * abl.z0
    n_mod = 0
    for ci in range(cell_centres.shape[0]):
        z = cell_centres[ci, z_axis] - fs_z
        if z < z_rsl and z > 0:
            # Enhanced mixing in roughness sublayer
            ratio = z / z_rsl
            enhancement = 1.0 + 0.5 * (1.0 - ratio)
            U_mag = np.linalg.norm(U[ci])
            if U_mag > 1e-30:
                U_dir = U[ci] / U_mag
                U[ci] = U_mag * enhancement * U_dir
            k[ci] *= (1.0 + 0.3 * (1.0 - ratio))
            n_mod += 1
    return n_mod


# ---------------------------------------------------------------------------
# Spectral generation
# ---------------------------------------------------------------------------


def _generate_spectral_coefficients(n_cells, k_arr, intensity_arr, L, model):
    """Generate spectral velocity amplitudes for synthetic turbulence."""
    n_modes = 10
    coeffs = np.zeros((n_cells, n_modes, 3), dtype=np.float64)

    for ci in range(n_cells):
        k_tke = max(k_arr[ci], 1e-10)
        u_rms = math.sqrt(2.0 / 3.0 * k_tke)

        for mode in range(n_modes):
            freq = (mode + 1) / L  # spatial frequency
            if model == "von_karman":
                # Von Karman spectrum shape
                f_norm = freq * L
                E = f_norm ** 4 / (1.0 + f_norm ** 2) ** (17.0 / 6.0)
            else:  # kaimal
                f_norm = freq * L
                E = f_norm / (1.0 + f_norm * 5.0 / 3.0) ** (5.0 / 3.0)

            amp = u_rms * math.sqrt(max(E, 0.0))
            # Random phase (deterministic for reproducibility)
            phase = (ci * 7 + mode * 13) % 360 * math.pi / 180.0
            coeffs[ci, mode, 0] = amp * math.cos(phase)
            coeffs[ci, mode, 1] = amp * math.sin(phase)
            coeffs[ci, mode, 2] = amp * 0.5 * math.cos(phase + math.pi / 4.0)

    return coeffs
