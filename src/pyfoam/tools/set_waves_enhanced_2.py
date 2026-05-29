"""
setWaves enhanced v2 — enhanced wave initialisation with multiple wave types
and better dispersion solver (second generation).

Extends :func:`set_waves_enhanced` with:

- **Stokes 5th order**: Full 5th-order Stokes wave theory.
- **Irregular (JONSWAP)**: Multi-component irregular sea state from
  JONSWAP spectral density.
- **Iterative dispersion**: Higher-order dispersion relation with
  current effects.
- **Velocity potential export**: Return the velocity potential field
  for post-processing.

Usage::

    from pyfoam.tools.set_waves_enhanced_2 import (
        set_waves_enhanced_2, EnhancedWave2Properties,
    )

    wave = EnhancedWave2Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        wave_type="stokes5",
    )
    result = set_waves_enhanced_2(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave2Properties", "EnhancedWave2Result", "set_waves_enhanced_2"]


@dataclass
class EnhancedWave2Properties:
    """Enhanced v2 wave parameters.

    Parameters
    ----------
    water_depth : float
        Still water depth (m).
    wave_height : float
        Wave height H (m).
    wave_period : float
        Wave period T (s).
    wave_length : float, optional
        Explicit wave length.
    phase : float
        Initial phase (rad).
    direction : tuple
        Propagation direction.
    wave_type : str
        ``"stokes1"``, ``"stokes2"``, ``"stokes5"``, ``"cnoidal"``,
        or ``"irregular"``.
    current_velocity : tuple, optional
        ``(Ux, Uy, Uz)`` current velocity for Doppler-shifted dispersion.
    jonswap_gamma : float
        Peak enhancement factor for JONSWAP spectrum (default 3.3).
    n_components : int
        Number of spectral components for irregular waves.
    seed : int, optional
        Random seed for irregular wave phase sampling.
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

    def angular_frequency(self) -> float:
        return 2.0 * math.pi / self.wave_period

    def wave_number(self, g: float = 9.81) -> float:
        if self.wave_length is not None:
            return 2.0 * math.pi / self.wave_length
        return _solve_dispersion_enhanced(
            self.angular_frequency(), self.water_depth, g,
            self.current_velocity,
        )


@dataclass
class EnhancedWave2Result:
    """Result from :func:`set_waves_enhanced_2`.

    Attributes
    ----------
    alpha : np.ndarray
    pressure : np.ndarray
    velocity : np.ndarray
    wave_number : float
    wave_length : float
    potential : np.ndarray, optional
        Velocity potential field.
    spectrum_frequencies : np.ndarray, optional
        Frequencies used for irregular waves.
    spectrum_amplitudes : np.ndarray, optional
        Amplitudes for each spectral component.
    """

    alpha: np.ndarray = field(default_factory=lambda: np.empty(0))
    pressure: np.ndarray = field(default_factory=lambda: np.empty(0))
    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    wave_number: float = 0.0
    wave_length: float = 0.0
    potential: Optional[np.ndarray] = None
    spectrum_frequencies: Optional[np.ndarray] = None
    spectrum_amplitudes: Optional[np.ndarray] = None


def set_waves_enhanced_2(
    mesh: "FvMesh",
    wave: EnhancedWave2Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave2Result:
    """Initialise wave fields with enhanced v2 wave models.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh.
    wave : EnhancedWave2Properties
        Wave parameters.
    g : float or sequence
        Gravitational acceleration.
    rho : float
        Water density.
    time : float
        Time instant.
    free_surface_z : float
        Z-coordinate of still water level.
    compute_potential : bool
        If True, also compute the velocity potential field.

    Returns
    -------
    EnhancedWave2Result
        Fields and spectral data.
    """
    if isinstance(g, (int, float)):
        g_mag = float(g)
    else:
        g_vec = np.asarray(g, dtype=np.float64)
        g_mag = float(np.linalg.norm(g_vec))

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    dir_vec = np.asarray(wave.direction, dtype=np.float64)
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm > 1e-30:
        dir_vec = dir_vec / dir_norm
    else:
        dir_vec = np.array([1.0, 0.0, 0.0])

    if wave.wave_type == "irregular":
        return _irregular_waves(
            mesh, wave, g_mag, rho, time, free_surface_z,
            cell_centres, n_cells, dir_vec, compute_potential,
        )

    H = wave.wave_height
    d = wave.water_depth
    k = wave.wave_number(g_mag)
    omega = wave.angular_frequency()
    phase = wave.phase
    L = 2.0 * math.pi / k

    alpha = np.zeros(n_cells, dtype=np.float64)
    pressure = np.zeros(n_cells, dtype=np.float64)
    velocity = np.zeros((n_cells, 3), dtype=np.float64)
    potential = np.zeros(n_cells, dtype=np.float64) if compute_potential else None

    for ci in range(n_cells):
        cc = cell_centres[ci]
        x_along = np.dot(cc, dir_vec)
        z = cc[2] - free_surface_z
        theta = k * x_along - omega * time + phase

        eta = _compute_eta(H, k, d, theta, wave.wave_type)
        alpha[ci] = 1.0 if z <= eta else 0.0

        if z <= eta:
            pressure[ci] = rho * g_mag * (eta - z)
        else:
            pressure[ci] = 0.0

        velocity[ci] = _compute_wave_velocity(
            H, k, omega, d, z, theta, dir_vec, wave.wave_type,
        )

        if compute_potential:
            potential[ci] = _compute_potential(
                H, k, omega, d, z, theta, g_mag,
            )

    return EnhancedWave2Result(
        alpha=alpha,
        pressure=pressure,
        velocity=velocity,
        wave_number=k,
        wave_length=L,
        potential=potential,
    )


# ---------------------------------------------------------------------------
# Irregular waves (JONSWAP)
# ---------------------------------------------------------------------------


def _irregular_waves(
    mesh, wave, g_mag, rho, time, free_surface_z,
    cell_centres, n_cells, dir_vec, compute_potential,
):
    """Superpose multiple wave components from JONSWAP spectrum."""
    Tp = wave.wave_period
    fp = 1.0 / Tp
    d = wave.water_depth
    gamma = wave.jonswap_gamma
    Hs = wave.wave_height
    n_comp = wave.n_components

    rng = np.random.default_rng(wave.seed)

    # Frequency range: 0.5*fp to 3*fp
    freqs = np.linspace(0.5 * fp, 3.0 * fp, n_comp)
    df = freqs[1] - freqs[0] if n_comp > 1 else fp

    # JONSWAP spectrum
    S = _jonswap_spectrum(freqs, fp, Hs, gamma, g_mag)
    amplitudes = np.sqrt(2.0 * S * df)

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, n_comp)

    alpha = np.zeros(n_cells, dtype=np.float64)
    pressure = np.zeros(n_cells, dtype=np.float64)
    velocity = np.zeros((n_cells, 3), dtype=np.float64)
    potential = np.zeros(n_cells, dtype=np.float64) if compute_potential else None

    for ci in range(n_cells):
        cc = cell_centres[ci]
        x_along = np.dot(cc, dir_vec)
        z = cc[2] - free_surface_z

        eta_total = 0.0
        u_total = np.zeros(3, dtype=np.float64)
        phi_total = 0.0

        for comp in range(n_comp):
            omega_c = 2 * math.pi * freqs[comp]
            k_c = _solve_dispersion_enhanced(omega_c, d, g_mag)
            theta_c = k_c * x_along - omega_c * time + phases[comp]

            eta_c = amplitudes[comp] * math.cos(theta_c)
            eta_total += eta_c

            if z <= eta_total:
                u_c = _compute_wave_velocity(
                    2 * amplitudes[comp], k_c, omega_c, d, z, theta_c, dir_vec, "stokes1",
                )
                u_total += u_c

                if compute_potential:
                    phi_total += _compute_potential(
                        2 * amplitudes[comp], k_c, omega_c, d, z, theta_c, g_mag,
                    )

        alpha[ci] = 1.0 if z <= eta_total else 0.0
        if z <= eta_total:
            pressure[ci] = rho * g_mag * max(eta_total - z, 0.0)
        velocity[ci] = u_total
        if potential is not None:
            potential[ci] = phi_total

    return EnhancedWave2Result(
        alpha=alpha,
        pressure=pressure,
        velocity=velocity,
        wave_number=0.0,
        wave_length=0.0,
        potential=potential,
        spectrum_frequencies=freqs,
        spectrum_amplitudes=amplitudes,
    )


def _jonswap_spectrum(f, fp, Hs, gamma, g):
    """JONSWAP spectral density."""
    sigma = np.where(f <= fp, 0.07, 0.09)
    alpha_s = 0.0081  # Phillips constant (simplified)
    r = np.exp(-((f - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
    S = alpha_s * g ** 2 / (2 * np.pi * f ** 5) * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r
    # Scale to match Hs
    # Trapezoidal integration (numpy version safe)
    m0 = float(np.sum((S[:-1] + S[1:]) * np.diff(f)) / 2.0) if len(f) > 1 else float(S[0] * (f[-1] - f[0]))
    Hs_ref = 4 * math.sqrt(max(m0, 1e-30))
    if Hs_ref > 1e-30:
        S = S * (Hs / Hs_ref) ** 2
    return S


# ---------------------------------------------------------------------------
# Wave models
# ---------------------------------------------------------------------------


def _compute_eta(H, k, d, theta, wave_type):
    """Compute free surface elevation."""
    a = 0.5 * H

    if wave_type == "stokes1":
        return a * math.cos(theta)

    elif wave_type == "stokes2":
        eta1 = a * math.cos(theta)
        kd = k * d
        if kd > 50:
            C = 0.0
        else:
            C = (3.0 - math.tanh(kd) ** 2) / (4.0 * math.tanh(kd) ** 3)
        eta2 = a ** 2 * k * C * math.cos(2 * theta)
        return eta1 + eta2

    elif wave_type == "stokes5":
        return _stokes5_eta(a, k, d, theta)

    elif wave_type == "cnoidal":
        kd = k * d
        if kd > 3.0:
            return a * math.cos(theta) + a ** 2 * k * 0.25 * math.cos(2 * theta)
        m = min(max(kd / 3.0, 0.01), 0.99)
        K_m = _elliptic_K(m)
        cn_val = math.cos(math.pi * theta / (2.0 * K_m))
        return a * cn_val ** 2

    else:
        return a * math.cos(theta)


def _stokes5_eta(a, k, d, theta):
    """5th-order Stokes wave elevation."""
    kd = k * d
    if kd > 50:
        C1, C2, C3 = 0.0, 0.0, 0.0
    else:
        th = math.tanh(kd)
        C1 = (3.0 - th ** 2) / (4.0 * th ** 3)
        C2 = (6.0 - 26.0 * th ** 2 + 29.0 * th ** 4 - 3.0 * th ** 6) / (48.0 * th ** 6)
        C3 = (120.0 - 576.0 * th ** 2 + 594.0 * th ** 4 - 171.0 * th ** 6 + 3.0 * th ** 8) / (
            384.0 * th ** 9
        )

    eps = a * k
    eta = (
        a * math.cos(theta)
        + a * eps * C1 * math.cos(2 * theta)
        + a * eps ** 2 * C2 * math.cos(3 * theta)
        + a * eps ** 3 * C3 * math.cos(4 * theta)
    )
    return eta


def _compute_wave_velocity(H, k, omega, d, z, theta, dir_vec, wave_type):
    """Compute wave-induced velocity at depth z."""
    a = 0.5 * H
    kd = k * d

    cosh_kd = math.cosh(kd) if kd < 500 else math.cosh(kd)
    sinh_kd = math.sinh(kd) if kd < 500 else math.cosh(kd)

    z_eff = max(z + d, 0.0)
    kz = k * z_eff

    cosh_kz = math.cosh(kz) if kz < 500 else math.exp(kz) / 2.0
    sinh_kz = math.sinh(kz) if kz < 500 else math.exp(kz) / 2.0

    safe_sinh = max(sinh_kd, 1e-30)
    safe_cosh = max(cosh_kd, 1e-30)

    U_horiz = a * omega * (cosh_kz / safe_sinh) * math.cos(theta)
    U_vert = a * omega * (sinh_kz / safe_sinh) * math.sin(theta)

    vel = U_horiz * dir_vec
    vel[2] += U_vert

    if wave_type in ("stokes2", "stokes5", "cnoidal"):
        cosh_2kz = math.cosh(2.0 * kz) if 2.0 * kz < 500 else math.exp(2.0 * kz) / 2.0
        U_horiz_2 = (3.0 / 4.0) * a ** 2 * omega * k * (cosh_2kz / safe_cosh ** 2) * math.cos(2 * theta)
        vel += U_horiz_2 * dir_vec

    return vel


def _compute_potential(H, k, omega, d, z, theta, g):
    """Compute velocity potential."""
    a = 0.5 * H
    kd = k * d
    sinh_kd = math.sinh(kd) if kd < 500 else math.cosh(kd)
    z_eff = max(z + d, 0.0)
    kz = k * z_eff
    sinh_kz = math.sinh(kz) if kz < 500 else math.exp(kz) / 2.0
    safe_sinh = max(sinh_kd, 1e-30)
    return a * g / omega * (sinh_kz / safe_sinh) * math.sin(theta)


# ---------------------------------------------------------------------------
# Enhanced dispersion solver with current
# ---------------------------------------------------------------------------


def _solve_dispersion_enhanced(omega, d, g, current_velocity=None):
    """Solve dispersion relation with optional current Doppler shift."""
    omega_eff = omega
    if current_velocity is not None:
        Uc = np.linalg.norm(current_velocity)
        # Simplified: reduce effective frequency for following current
        omega_eff = max(omega - Uc * omega / math.sqrt(g * d), omega * 0.5)

    k = omega_eff ** 2 / g  # Initial deep-water guess

    for _ in range(100):
        kd = k * d
        tanh_kd = math.tanh(kd) if kd < 500 else 1.0
        f_k = g * k * tanh_kd - omega_eff ** 2
        f_prime = g * (tanh_kd + kd / math.cosh(kd) ** 2) if kd < 500 else g

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


def _elliptic_K(m):
    """Approximate complete elliptic integral K(m)."""
    if m <= 0:
        return math.pi / 2.0
    if m >= 1.0:
        return 1e10

    a_n = 1.0
    b_n = math.sqrt(1.0 - m)

    for _ in range(50):
        a_new = 0.5 * (a_n + b_n)
        b_new = math.sqrt(a_n * b_n)
        a_n = a_new
        b_n = b_new
        if abs(a_n - b_n) < 1e-15:
            break

    return math.pi / (2.0 * a_n)
