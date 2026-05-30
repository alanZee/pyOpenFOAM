"""
setWaves enhanced v8 — enhanced wave initialisation with wave-structure
interaction, coastal morphodynamics coupling, and wave load estimation
(eighth generation).

Extends :func:`set_waves_enhanced_7` with:

- **Wave-structure interaction**: Compute wave forces (Morison equation)
  on cylindrical and rectangular structures.
- **Coastal morphodynamics**: Estimate bed level change from sediment
  transport gradients (Exner equation).
- **Wave load estimation**: Estimate wave overtopping rates and
  impact pressures for coastal structure design.

Usage::

    from pyfoam.tools.set_waves_enhanced_8 import (
        set_waves_enhanced_8, EnhancedWave8Properties,
    )

    wave = EnhancedWave8Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        structure_interaction=True,
        morphodynamics=True,
    )
    result = set_waves_enhanced_8(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave8Properties", "EnhancedWave8Result", "set_waves_enhanced_8"]


@dataclass
class EnhancedWave8Properties:
    """Enhanced v8 wave parameters.

    Parameters
    ----------
    water_depth .. sediment_density
        Forwarded from v7.
    structure_interaction : bool
        Compute wave forces on structures.
    structure_diameter : float
        Structure diameter for Morison equation (m).
    structure_draft : float
        Structure draft depth (m).
    morphodynamics : bool
        Enable coastal morphodynamics coupling.
    morpho_dt : float
        Morphodynamic time step (s).
    morpho_porosity : float
        Bed porosity (0-1).
    estimate_overtopping : bool
        Estimate wave overtopping rate.
    crest_height : float
        Structure crest height above SWL (m).
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
    structure_interaction: bool = False
    structure_diameter: float = 1.0
    structure_draft: float = 5.0
    morphodynamics: bool = False
    morpho_dt: float = 3600.0
    morpho_porosity: float = 0.4
    estimate_overtopping: bool = False
    crest_height: float = 5.0


@dataclass
class EnhancedWave8Result:
    """Result from :func:`set_waves_enhanced_8`.

    Attributes
    ----------
    alpha .. sediment_transport_rate
        Forwarded from v7.
    morison_force : float
        Morison wave force per unit length (N/m).
    morison_inertia : float
        Inertia force component (N/m).
    morison_drag : float
        Drag force component (N/m).
    bed_level_change : float
        Estimated bed level change (m).
    overtopping_rate : float
        Wave overtopping rate (m3/s/m).
    impact_pressure : float
        Wave impact pressure (Pa).
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
    morison_force: float = 0.0
    morison_inertia: float = 0.0
    morison_drag: float = 0.0
    bed_level_change: float = 0.0
    overtopping_rate: float = 0.0
    impact_pressure: float = 0.0


def set_waves_enhanced_8(
    mesh: "FvMesh",
    wave: EnhancedWave8Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave8Result:
    """Initialise wave fields with v8 enhancements.

    Parameters
    ----------
    mesh : FvMesh
    wave : EnhancedWave8Properties
    g, rho, time, free_surface_z, compute_potential

    Returns
    -------
    EnhancedWave8Result
    """
    from pyfoam.tools.set_waves_enhanced_7 import (
        set_waves_enhanced_7,
        EnhancedWave7Properties,
    )

    v7_props = EnhancedWave7Properties(
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
        current_interaction=wave.current_interaction,
        current_depth_average=wave.current_depth_average,
        sediment_coupling=wave.sediment_coupling,
        sediment_d50=wave.sediment_d50,
        sediment_density=wave.sediment_density,
    )

    v7_result = set_waves_enhanced_7(
        mesh, v7_props, g, rho, time, free_surface_z, compute_potential,
    )

    g_val = g if isinstance(g, float) else np.linalg.norm(g)

    # Wave-structure interaction (Morison equation)
    morison_force = 0.0
    morison_inertia = 0.0
    morison_drag = 0.0
    if wave.structure_interaction:
        morison_force, morison_inertia, morison_drag = _compute_morison_force(
            wave.wave_height, wave.wave_period, wave.water_depth,
            wave.structure_diameter, rho, g_val,
        )

    # Coastal morphodynamics (Exner equation)
    dbed = 0.0
    if wave.morphodynamics:
        dbed = _compute_morphodynamics(
            v7_result.sediment_transport_rate, wave.morpho_dt,
            wave.morpho_porosity, wave.beach_slope,
        )

    # Wave overtopping
    overtop = 0.0
    impact_p = 0.0
    if wave.estimate_overtopping:
        overtop, impact_p = _estimate_overtopping(
            wave.wave_height, wave.wave_period, wave.crest_height,
            wave.water_depth, rho, g_val,
        )

    return EnhancedWave8Result(
        alpha=v7_result.alpha,
        pressure=v7_result.pressure,
        velocity=v7_result.velocity,
        wave_number=v7_result.wave_number,
        wave_length=v7_result.wave_length,
        potential=v7_result.potential,
        spectrum_frequencies=v7_result.spectrum_frequencies,
        spectrum_amplitudes=v7_result.spectrum_amplitudes,
        ursell_number=v7_result.ursell_number,
        iribarren_number=v7_result.iribarren_number,
        is_breaking=v7_result.is_breaking,
        benjamin_feir_index=v7_result.benjamin_feir_index,
        spectral_bandwidth=v7_result.spectral_bandwidth,
        rogue_wave_detected=v7_result.rogue_wave_detected,
        max_wave_elevation=v7_result.max_wave_elevation,
        peakedness=v7_result.peakedness,
        groupiness_factor=v7_result.groupiness_factor,
        directional_spread=v7_result.directional_spread,
        n_sponge_cells=v7_result.n_sponge_cells,
        n_generation_cells=v7_result.n_generation_cells,
        n_absorption_cells=v7_result.n_absorption_cells,
        sponge_damping=v7_result.sponge_damping,
        doppler_shift=v7_result.doppler_shift,
        energy_flux=v7_result.energy_flux,
        group_velocity=v7_result.group_velocity,
        radiation_stress_xx=v7_result.radiation_stress_xx,
        shields_parameter=v7_result.shields_parameter,
        sediment_transport_rate=v7_result.sediment_transport_rate,
        morison_force=morison_force,
        morison_inertia=morison_inertia,
        morison_drag=morison_drag,
        bed_level_change=dbed,
        overtopping_rate=overtop,
        impact_pressure=impact_p,
    )


# ---------------------------------------------------------------------------
# Morison equation
# ---------------------------------------------------------------------------


def _compute_morison_force(H, T, d, D, rho, g):
    """Compute Morison wave force on a vertical cylinder."""
    if H <= 0 or T <= 0 or D <= 0:
        return 0.0, 0.0, 0.0

    omega = 2.0 * math.pi / T
    k = omega ** 2 / (g + 1e-30)  # deep water approximation

    # Orbital velocity at structure
    U_m = math.pi * H / (T * math.sinh(k * d + 1e-30) + 1e-30)
    U_dot = 2.0 * math.pi ** 2 * H / (T ** 2 * math.sinh(k * d + 1e-30) + 1e-30)

    # Inertia coefficient (Cm = 2.0 for circular cylinder)
    Cm = 2.0
    Cd = 1.2  # drag coefficient

    # Froude-Krylov + diffraction inertia
    F_I = Cm * rho * math.pi * D ** 2 / 4.0 * U_dot

    # Drag force
    F_D = 0.5 * Cd * rho * D * U_m ** 2

    F_total = math.sqrt(F_I ** 2 + F_D ** 2)

    return F_total, F_I, F_D


# ---------------------------------------------------------------------------
# Coastal morphodynamics
# ---------------------------------------------------------------------------


def _compute_morphodynamics(qs, dt, porosity, slope):
    """Compute bed level change from sediment transport (Exner equation)."""
    # db/dt = -1/(1-p) * dq/dx
    # Simplified: assume gradient = qs * slope
    if qs <= 0:
        return 0.0
    dq_dx = qs * slope
    dbdt = -dq_dx / (1.0 - porosity + 1e-30)
    return dbdt * dt


# ---------------------------------------------------------------------------
# Wave overtopping
# ---------------------------------------------------------------------------


def _estimate_overtopping(H, T, Rc, d, rho, g):
    """Estimate wave overtopping rate (EurOtop formula)."""
    if H <= 0 or Rc <= 0:
        return 0.0, 0.0

    # Relative crest freeboard
    Rc_H = Rc / H

    # EurOtop (2018) empirical formula (simplified)
    if Rc_H > 5.0:
        return 0.0, 0.0

    q = math.sqrt(g * H ** 3) * 0.023 * math.exp(-2.3 * Rc_H)

    # Impact pressure (simplified Goda formula)
    p_impact = 0.5 * rho * g * H * (1.0 + math.sinh(2.0 * math.pi * d / (H + 1e-30)))

    return q, p_impact
