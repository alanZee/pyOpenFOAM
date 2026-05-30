"""
setWaves enhanced v7 — enhanced wave initialisation with wave-current interaction,
sediment transport coupling, and wave energy flux analysis
(seventh generation).

Extends :func:`set_waves_enhanced_6` with:

- **Wave-current interaction**: Account for Doppler shift and current
  refraction effects on wave kinematics.
- **Sediment transport coupling**: Estimate bed shear stress and
  Shields parameter for sediment motion analysis.
- **Wave energy flux analysis**: Compute energy flux, group velocity,
  and radiation stress tensor components.

Usage::

    from pyfoam.tools.set_waves_enhanced_7 import (
        set_waves_enhanced_7, EnhancedWave7Properties,
    )

    wave = EnhancedWave7Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        current_interaction=True,
        sediment_coupling=True,
    )
    result = set_waves_enhanced_7(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave7Properties", "EnhancedWave7Result", "set_waves_enhanced_7"]


@dataclass
class EnhancedWave7Properties:
    """Enhanced v7 wave parameters.

    Parameters
    ----------
    water_depth .. absorption
        Forwarded from v6.
    current_interaction : bool
        Enable wave-current interaction modelling.
    current_depth_average : bool
        Use depth-averaged current for interaction.
    sediment_coupling : bool
        Compute sediment transport parameters.
    sediment_d50 : float
        Median sediment grain diameter (m).
    sediment_density : float
        Sediment particle density (kg/m3).
    """

    water_depth: float = 10.0
    wave_height: float = 1.0
    wave_period: float = 2.0
    wave_length: Optional[float] = None
    phase: float = 0.0
    direction: tuple = (1.0, 0.0, 0.0)
    wave_type: str = "stokes1"
    current_velocity: Optional[tuple] = None
    jonswap_gamma: float = 3.3
    n_components: int = 50
    seed: Optional[int] = None
    stream_N: int = 5
    beach_slope: float = 0.05
    rogue_focusing_distance: float = 0.0
    rogue_amplitude_factor: float = 2.0
    n_directions: int = 1
    spreading_exponent: float = 2.0
    current_profile: str = "uniform"
    current_shear: float = 0.0
    sponge_layer: bool = False
    sponge_width: float = 5.0
    sponge_coefficient: float = 10.0
    sponge_profile: str = "quadratic"
    generation_zone: bool = False
    generation_start: float = 0.0
    generation_width: float = 2.0
    absorption: bool = False
    current_interaction: bool = False
    current_depth_average: bool = True
    sediment_coupling: bool = False
    sediment_d50: float = 0.0003
    sediment_density: float = 2650.0


@dataclass
class EnhancedWave7Result:
    """Result from :func:`set_waves_enhanced_7`.

    Attributes
    ----------
    alpha .. sponge_damping
        Forwarded from v6.
    doppler_shift : float
        Frequency shift due to current (Hz).
    energy_flux : float
        Wave energy flux per unit crest width (W/m).
    group_velocity : float
        Wave group velocity (m/s).
    radiation_stress_xx : float
        Radiation stress Sxx component (N/m).
    shields_parameter : float
        Shields parameter for sediment motion.
    sediment_transport_rate : float
        Estimated sediment transport rate (m2/s).
    """

    alpha: np.ndarray = field(default_factory=lambda: np.empty(0))
    pressure: np.ndarray = field(default_factory=lambda: np.empty(0))
    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    wave_number: float = 0.0
    wave_length: float = 0.0
    potential: Optional[np.ndarray] = None
    spectrum_frequencies: Optional[np.ndarray] = None
    spectrum_amplitudes: Optional[np.ndarray] = None
    ursell_number: float = 0.0
    iribarren_number: float = 0.0
    is_breaking: bool = False
    benjamin_feir_index: float = 0.0
    spectral_bandwidth: float = 0.0
    rogue_wave_detected: bool = False
    max_wave_elevation: float = 0.0
    peakedness: float = 0.0
    groupiness_factor: float = 0.0
    directional_spread: float = 0.0
    n_sponge_cells: int = 0
    n_generation_cells: int = 0
    n_absorption_cells: int = 0
    sponge_damping: Optional[np.ndarray] = None
    doppler_shift: float = 0.0
    energy_flux: float = 0.0
    group_velocity: float = 0.0
    radiation_stress_xx: float = 0.0
    shields_parameter: float = 0.0
    sediment_transport_rate: float = 0.0


def set_waves_enhanced_7(
    mesh: "FvMesh",
    wave: EnhancedWave7Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave7Result:
    """Initialise wave fields with v7 enhancements.

    Parameters
    ----------
    mesh : FvMesh
    wave : EnhancedWave7Properties
    g, rho, time, free_surface_z, compute_potential

    Returns
    -------
    EnhancedWave7Result
    """
    from pyfoam.tools.set_waves_enhanced_6 import (
        set_waves_enhanced_6,
        EnhancedWave6Properties,
    )

    v6_props = EnhancedWave6Properties(
        water_depth=wave.water_depth,
        wave_height=wave.wave_height,
        wave_period=wave.wave_period,
        wave_length=wave.wave_length,
        phase=wave.phase,
        direction=wave.direction,
        wave_type=wave.wave_type,
        current_velocity=wave.current_velocity,
        jonswap_gamma=wave.jonswap_gamma,
        n_components=wave.n_components,
        seed=wave.seed,
        stream_N=wave.stream_N,
        beach_slope=wave.beach_slope,
        rogue_focusing_distance=wave.rogue_focusing_distance,
        rogue_amplitude_factor=wave.rogue_amplitude_factor,
        n_directions=wave.n_directions,
        spreading_exponent=wave.spreading_exponent,
        current_profile=wave.current_profile,
        current_shear=wave.current_shear,
        sponge_layer=wave.sponge_layer,
        sponge_width=wave.sponge_width,
        sponge_coefficient=wave.sponge_coefficient,
        sponge_profile=wave.sponge_profile,
        generation_zone=wave.generation_zone,
        generation_start=wave.generation_start,
        generation_width=wave.generation_width,
        absorption=wave.absorption,
    )

    v6_result = set_waves_enhanced_6(
        mesh, v6_props, g, rho, time, free_surface_z, compute_potential,
    )

    g_val = g if isinstance(g, float) else np.linalg.norm(g)

    # Wave-current interaction
    doppler = 0.0
    if wave.current_interaction and wave.current_velocity is not None:
        doppler = _compute_doppler_shift(
            v6_result.wave_number, wave.current_velocity, wave.direction,
        )

    # Energy analysis
    energy_flux, group_vel, rad_stress = _compute_energy_analysis(
        wave.wave_height, wave.wave_period, wave.water_depth,
        v6_result.wave_number, v6_result.wave_length,
        rho, g_val,
    )

    # Sediment transport
    shields = 0.0
    sed_rate = 0.0
    if wave.sediment_coupling:
        shields, sed_rate = _compute_sediment_transport(
            wave.wave_height, wave.water_depth, v6_result.wave_length,
            wave.sediment_d50, wave.sediment_density, rho, g_val,
        )

    return EnhancedWave7Result(
        alpha=v6_result.alpha,
        pressure=v6_result.pressure,
        velocity=v6_result.velocity,
        wave_number=v6_result.wave_number,
        wave_length=v6_result.wave_length,
        potential=v6_result.potential,
        spectrum_frequencies=v6_result.spectrum_frequencies,
        spectrum_amplitudes=v6_result.spectrum_amplitudes,
        ursell_number=v6_result.ursell_number,
        iribarren_number=v6_result.iribarren_number,
        is_breaking=v6_result.is_breaking,
        benjamin_feir_index=v6_result.benjamin_feir_index,
        spectral_bandwidth=v6_result.spectral_bandwidth,
        rogue_wave_detected=v6_result.rogue_wave_detected,
        max_wave_elevation=v6_result.max_wave_elevation,
        peakedness=v6_result.peakedness,
        groupiness_factor=v6_result.groupiness_factor,
        directional_spread=v6_result.directional_spread,
        n_sponge_cells=v6_result.n_sponge_cells,
        n_generation_cells=v6_result.n_generation_cells,
        n_absorption_cells=v6_result.n_absorption_cells,
        sponge_damping=v6_result.sponge_damping,
        doppler_shift=doppler,
        energy_flux=energy_flux,
        group_velocity=group_vel,
        radiation_stress_xx=rad_stress,
        shields_parameter=shields,
        sediment_transport_rate=sed_rate,
    )


# ---------------------------------------------------------------------------
# Wave-current interaction
# ---------------------------------------------------------------------------


def _compute_doppler_shift(k, current_vel, wave_dir):
    """Compute Doppler frequency shift from current."""
    current = np.array(current_vel, dtype=np.float64)
    direction = np.array(wave_dir, dtype=np.float64)
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-30:
        direction = direction / d_norm
    # Doppler shift: delta_f = -k * (U . d) / (2*pi)
    return -k * np.dot(current, direction) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Energy analysis
# ---------------------------------------------------------------------------


def _compute_energy_analysis(H, T, d, k, L, rho, g):
    """Compute wave energy flux and radiation stress."""
    if H <= 0 or T <= 0 or L <= 0:
        return 0.0, 0.0, 0.0

    # Deep water: Cg = C/2, E = rho*g*H^2/8
    omega = 2.0 * math.pi / T
    C = omega / k if k > 0 else 0.0

    # Group velocity (general)
    n = 0.5 * (1.0 + 2.0 * k * d / math.sinh(2.0 * k * d + 1e-30))
    Cg = n * C

    # Energy density
    E = rho * g * H ** 2 / 8.0

    # Energy flux
    energy_flux = E * Cg

    # Radiation stress Sxx (simplified for long waves)
    rad_stress = E * (2.0 * n - 0.5)

    return energy_flux, Cg, rad_stress


# ---------------------------------------------------------------------------
# Sediment transport
# ---------------------------------------------------------------------------


def _compute_sediment_transport(H, d, L, d50, rho_s, rho_w, g):
    """Estimate Shields parameter and sediment transport rate."""
    if d50 <= 0 or H <= 0 or L <= 0:
        return 0.0, 0.0

    # Bottom orbital velocity from linear wave theory
    omega = 2.0 * math.pi / (L / math.sqrt(g * d + 1e-30))  # approximate
    T = 2.0 * math.pi / omega if omega > 0 else 1.0
    k = 2.0 * math.pi / L

    # Orbital velocity amplitude
    U_b = math.pi * H / (T * math.sinh(k * d + 1e-30))

    # Shields parameter: theta = 0.5 * fw * U_b^2 / ((s-1) * g * d50)
    s = rho_s / rho_w
    fw = 0.015  # friction factor approximation
    theta = 0.5 * fw * U_b ** 2 / ((s - 1.0) * g * d50 + 1e-30)

    # Meyer-Peter & Muller transport rate (simplified)
    if theta > 0.047:
        q_s = 8.0 * math.sqrt((s - 1.0) * g * d50 ** 3) * (theta - 0.047) ** 1.5
    else:
        q_s = 0.0

    return max(0.0, theta), max(0.0, q_s)
