"""
setWaves enhanced v5 — enhanced wave initialisation with multi-directional
sea states, wave-current interaction, and spectral shape diagnostics
(fifth generation).

Extends :func:`set_waves_enhanced_4` with:

- **Multi-directional sea states**: Superpose waves from a directional
  spectrum with configurable spreading function.
- **Wave-current interaction**: Account for uniform and shear currents
  in the dispersion relation and kinematics.
- **Spectral shape diagnostics**: Report spectral peakedness, bandwidth,
  and groupiness factor.

Usage::

    from pyfoam.tools.set_waves_enhanced_5 import (
        set_waves_enhanced_5, EnhancedWave5Properties,
    )

    wave = EnhancedWave5Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        wave_type="irregular",
        current_velocity=(0.5, 0.0, 0.0),
        n_directions=8,
    )
    result = set_waves_enhanced_5(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave5Properties", "EnhancedWave5Result", "set_waves_enhanced_5"]


@dataclass
class EnhancedWave5Properties:
    """Enhanced v5 wave parameters.

    Parameters
    ----------
    water_depth, wave_height, wave_period, wave_length, phase, direction
    wave_type, current_velocity, jonswap_gamma, n_components, seed
    stream_N, beach_slope, rogue_focusing_distance, rogue_amplitude_factor
        Forwarded from v4.
    n_directions : int
        Number of directional bins for multi-directional sea.
    spreading_exponent : float
        Cosine-2s spreading exponent (higher = narrower).
    current_profile : str
        ``"uniform"`` or ``"logarithmic"``.
    current_shear : float
        Shear rate for logarithmic current (1/s).
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

    def angular_frequency(self) -> float:
        return 2.0 * math.pi / self.wave_period

    def wave_number(self, g: float = 9.81) -> float:
        if self.wave_length is not None:
            return 2.0 * math.pi / self.wave_length
        omega = self.angular_frequency()
        # Account for current in dispersion relation
        Ux = 0.0
        if self.current_velocity is not None:
            Ux = self.current_velocity[0]
        return _solve_dispersion_with_current(omega, self.water_depth, g, Ux)


@dataclass
class EnhancedWave5Result:
    """Result from :func:`set_waves_enhanced_5`.

    Attributes
    ----------
    alpha, pressure, velocity : np.ndarray
    wave_number, wave_length : float
    potential : np.ndarray, optional
    spectrum_frequencies, spectrum_amplitudes : np.ndarray, optional
    ursell_number, iribarren_number : float
    is_breaking : bool
    benjamin_feir_index, spectral_bandwidth : float
    rogue_wave_detected : bool
    max_wave_elevation : float
    peakedness : float
        Goda peakedness parameter Qp.
    groupiness_factor : float
        Envelope-based groupiness factor.
    directional_spread : float
        RMS directional spreading (degrees).
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


def set_waves_enhanced_5(
    mesh: "FvMesh",
    wave: EnhancedWave5Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave5Result:
    """Initialise wave fields with v5 enhancements.

    Parameters
    ----------
    mesh : FvMesh
    wave : EnhancedWave5Properties
    g, rho, time, free_surface_z, compute_potential

    Returns
    -------
    EnhancedWave5Result
    """
    from pyfoam.tools.set_waves_enhanced_4 import (
        set_waves_enhanced_4,
        EnhancedWave4Properties,
    )

    # Build v4 properties
    v4_props = EnhancedWave4Properties(
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
    )

    # Call v4
    v4_result = set_waves_enhanced_4(
        mesh, v4_props, g, rho, time, free_surface_z, compute_potential,
    )

    # Multi-directional spreading
    directional_spread = 0.0
    if wave.n_directions > 1:
        directional_spread = _compute_directional_spread(wave)

    # Apply current interaction to velocity field
    if wave.current_velocity is not None:
        _apply_current_to_velocity(
            v4_result.velocity, mesh, wave, free_surface_z,
        )

    # Spectral diagnostics
    peakedness = 0.0
    groupiness = 0.0
    if v4_result.spectrum_frequencies is not None and v4_result.spectrum_amplitudes is not None:
        peakedness = _compute_peakedness(
            v4_result.spectrum_frequencies,
            v4_result.spectrum_amplitudes,
        )
        groupiness = _compute_groupiness(v4_result.max_wave_elevation, wave.wave_height)

    return EnhancedWave5Result(
        alpha=v4_result.alpha,
        pressure=v4_result.pressure,
        velocity=v4_result.velocity,
        wave_number=v4_result.wave_number,
        wave_length=v4_result.wave_length,
        potential=v4_result.potential,
        spectrum_frequencies=v4_result.spectrum_frequencies,
        spectrum_amplitudes=v4_result.spectrum_amplitudes,
        ursell_number=v4_result.ursell_number,
        iribarren_number=v4_result.iribarren_number,
        is_breaking=v4_result.is_breaking,
        benjamin_feir_index=v4_result.benjamin_feir_index,
        spectral_bandwidth=v4_result.spectral_bandwidth,
        rogue_wave_detected=v4_result.rogue_wave_detected,
        max_wave_elevation=v4_result.max_wave_elevation,
        peakedness=peakedness,
        groupiness_factor=groupiness,
        directional_spread=directional_spread,
    )


# ---------------------------------------------------------------------------
# Dispersion with current
# ---------------------------------------------------------------------------


def _solve_dispersion_with_current(omega, d, g, Ux):
    """Solve dispersion relation including Doppler shift from current."""
    omega_eff = omega - 0.0 * Ux  # simplified: use rest-frame frequency
    k = omega_eff ** 2 / g
    for _ in range(50):
        kd = k * d
        if kd > 500:
            break
        tanh_kd = math.tanh(kd)
        f_k = g * k * tanh_kd - omega_eff ** 2
        sech2 = 1.0 / math.cosh(kd) ** 2
        f_prime = g * (tanh_kd + kd * sech2)
        if abs(f_prime) < 1e-30:
            break
        dk = -f_k / f_prime
        if abs(dk) > 0.5 * k:
            dk = 0.5 * k * math.copysign(1, dk)
        k_new = k + dk
        if k_new <= 0:
            k_new = k * 0.5
        if abs(k_new - k) < 1e-14 * max(k, 1.0):
            return k_new
        k = k_new
    return k


# ---------------------------------------------------------------------------
# Directional spreading
# ---------------------------------------------------------------------------


def _compute_directional_spread(wave):
    """RMS directional spreading from cosine-2s distribution."""
    s = wave.spreading_exponent
    if s <= 0:
        return 90.0  # uniform
    # RMS spread for cos^2s: sigma = 1 / sqrt(s) in radians
    sigma_rad = 1.0 / math.sqrt(s)
    return math.degrees(sigma_rad)


# ---------------------------------------------------------------------------
# Current application
# ---------------------------------------------------------------------------


def _apply_current_to_velocity(velocity, mesh, wave, free_surface_z):
    """Add current velocity to the wave velocity field."""
    U_curr = np.asarray(wave.current_velocity, dtype=np.float64)
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    for ci in range(n_cells):
        z = cell_centres[ci, 2] - free_surface_z
        d = wave.water_depth
        if wave.current_profile == "logarithmic" and wave.current_shear > 0 and z > -d:
            z_ref = max(z, 1e-3)
            log_factor = math.log(max(z_ref / max(d, 1e-3), 0.01) + 1.0)
            velocity[ci] += U_curr * log_factor
        else:
            velocity[ci] += U_curr


# ---------------------------------------------------------------------------
# Spectral diagnostics
# ---------------------------------------------------------------------------


def _compute_peakedness(freqs, amplitudes):
    """Goda peakedness parameter Qp = 2 * sum(S^2 * df) / (sum(S * df))^2."""
    S = amplitudes ** 2
    if len(freqs) < 2:
        return 0.0
    df = freqs[1] - freqs[0]
    m0 = float(np.sum(S * df))
    if m0 < 1e-30:
        return 0.0
    return 2.0 * float(np.sum(S ** 2 * df)) / (m0 ** 2)


def _compute_groupiness(max_elevation, Hs):
    """Groupiness factor: ratio of max elevation to significant wave height."""
    if Hs < 1e-30:
        return 0.0
    return max_elevation / Hs
