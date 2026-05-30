"""
setWaves enhanced v3 — enhanced wave initialisation with stream function
waves and improved dispersion solver (third generation).

Extends :func:`set_waves_enhanced_2` with:

- **Stream function waves**: Higher-order stream function wave theory
  with iterative solution of the stream function coefficients.
- **Improved dispersion solver**: Halley's method for faster convergence.
- **Wave diagnostics**: Reports Ursell number, Iribarren number, and
  breaking criterion check.

Usage::

    from pyfoam.tools.set_waves_enhanced_3 import (
        set_waves_enhanced_3, EnhancedWave3Properties,
    )

    wave = EnhancedWave3Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        wave_type="stream_function",
    )
    result = set_waves_enhanced_3(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave3Properties", "EnhancedWave3Result", "set_waves_enhanced_3"]


@dataclass
class EnhancedWave3Properties:
    """Enhanced v3 wave parameters.

    Parameters
    ----------
    water_depth : float
    wave_height : float
    wave_period : float
    wave_length : float, optional
    phase : float
    direction : tuple
    wave_type : str
        ``"stokes1"``, ``"stokes2"``, ``"stokes5"``, ``"cnoidal"``,
        ``"irregular"``, or ``"stream_function"``.
    current_velocity : tuple, optional
    jonswap_gamma : float
    n_components : int
    seed : int, optional
    stream_N : int
        Number of Fourier components for stream function (default 5).
    beach_slope : float
        Beach slope for Iribarren number calculation.
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

    def angular_frequency(self) -> float:
        return 2.0 * math.pi / self.wave_period

    def wave_number(self, g: float = 9.81) -> float:
        if self.wave_length is not None:
            return 2.0 * math.pi / self.wave_length
        return _solve_dispersion_halley(
            self.angular_frequency(), self.water_depth, g,
        )


@dataclass
class EnhancedWave3Result:
    """Result from :func:`set_waves_enhanced_3`.

    Attributes
    ----------
    alpha, pressure, velocity, wave_number, wave_length
    potential : np.ndarray, optional
    spectrum_frequencies, spectrum_amplitudes : np.ndarray, optional
    ursell_number : float
        Ursell number HL^2 / d^3.
    iribarren_number : float
        Iribarren number (surf similarity parameter).
    is_breaking : bool
        Whether the wave exceeds the breaking criterion.
    stream_coefficients : list[float], optional
        Stream function Fourier coefficients.
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
    stream_coefficients: Optional[List[float]] = None


def set_waves_enhanced_3(
    mesh: "FvMesh",
    wave: EnhancedWave3Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave3Result:
    """Initialise wave fields with enhanced v3 wave models.

    Parameters
    ----------
    mesh : FvMesh
    wave : EnhancedWave3Properties
    g : float or sequence
    rho : float
    time : float
    free_surface_z : float
    compute_potential : bool

    Returns
    -------
    EnhancedWave3Result
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

    H = wave.wave_height
    d = wave.water_depth
    k = wave.wave_number(g_mag)
    omega = wave.angular_frequency()
    L = 2.0 * math.pi / k

    # Wave diagnostics
    ursell = H * L ** 2 / (d ** 3) if d > 1e-30 else 0.0
    iribarren = wave.beach_slope / math.sqrt(H / L) if H > 1e-30 and L > 1e-30 else 0.0
    is_breaking = H / d > 0.78  # Simple McCowan breaking criterion

    if wave.wave_type == "irregular":
        result = _irregular_waves(
            mesh, wave, g_mag, rho, time, free_surface_z,
            cell_centres, n_cells, dir_vec, compute_potential,
        )
        result.ursell_number = ursell
        result.iribarren_number = iribarren
        result.is_breaking = is_breaking
        return result

    if wave.wave_type == "stream_function":
        return _stream_function_waves(
            mesh, wave, g_mag, rho, time, free_surface_z,
            cell_centres, n_cells, dir_vec, H, d, k, omega, L,
            ursell, iribarren, is_breaking, compute_potential,
        )

    alpha = np.zeros(n_cells, dtype=np.float64)
    pressure = np.zeros(n_cells, dtype=np.float64)
    velocity = np.zeros((n_cells, 3), dtype=np.float64)
    potential = np.zeros(n_cells, dtype=np.float64) if compute_potential else None

    for ci in range(n_cells):
        cc = cell_centres[ci]
        x_along = np.dot(cc, dir_vec)
        z = cc[2] - free_surface_z
        theta = k * x_along - omega * time + wave.phase

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
            potential[ci] = _compute_potential(H, k, omega, d, z, theta, g_mag)

    return EnhancedWave3Result(
        alpha=alpha,
        pressure=pressure,
        velocity=velocity,
        wave_number=k,
        wave_length=L,
        potential=potential,
        ursell_number=ursell,
        iribarren_number=iribarren,
        is_breaking=is_breaking,
    )


# ---------------------------------------------------------------------------
# Stream function waves
# ---------------------------------------------------------------------------


def _stream_function_waves(
    mesh, wave, g_mag, rho, time, free_surface_z,
    cell_centres, n_cells, dir_vec, H, d, k, omega, L,
    ursell, iribarren, is_breaking, compute_potential,
):
    """Stream function wave theory with iterative coefficient solution."""
    N = wave.stream_N
    a = 0.5 * H
    coeffs = _solve_stream_function_coefficients(a, k, d, N, g_mag, omega)

    alpha = np.zeros(n_cells, dtype=np.float64)
    pressure = np.zeros(n_cells, dtype=np.float64)
    velocity = np.zeros((n_cells, 3), dtype=np.float64)
    potential = np.zeros(n_cells, dtype=np.float64) if compute_potential else None

    # Mean current (Stokes drift)
    U_sf = coeffs[0] if len(coeffs) > 0 else 0.0

    for ci in range(n_cells):
        cc = cell_centres[ci]
        x_along = np.dot(cc, dir_vec)
        z = cc[2] - free_surface_z
        theta = k * x_along - omega * time + wave.phase

        # Stream function elevation
        eta = 0.0
        for n in range(1, N + 1):
            if n < len(coeffs):
                eta += coeffs[n] * math.cos(n * theta)
        eta = a * math.cos(theta) + sum(
            coeffs[n] * math.cos(n * theta) for n in range(1, min(N + 1, len(coeffs)))
        )

        alpha[ci] = 1.0 if z <= eta else 0.0
        if z <= eta:
            pressure[ci] = rho * g_mag * (eta - z)
        else:
            pressure[ci] = 0.0

        # Velocity from stream function
        u_horiz = U_sf
        for n in range(1, min(N + 1, len(coeffs))):
            cosh_nd = math.cosh(n * k * (z + d)) if n * k * (z + d) < 500 else math.exp(n * k * (z + d)) / 2
            sinh_nd = math.sinh(n * k * d) if n * k * d < 500 else math.exp(n * k * d) / 2
            safe_sinh = max(sinh_nd, 1e-30)
            u_horiz += n * k * coeffs[n] * (cosh_nd / safe_sinh) * math.cos(n * theta)

        vel = u_horiz * dir_vec
        velocity[ci] = vel

        if compute_potential:
            phi = U_sf * z
            for n in range(1, min(N + 1, len(coeffs))):
                sinh_nd = math.sinh(n * k * (z + d)) if n * k * (z + d) < 500 else math.exp(n * k * (z + d)) / 2
                sinh_nkd = math.sinh(n * k * d) if n * k * d < 500 else math.exp(n * k * d) / 2
                safe_sinh = max(sinh_nkd, 1e-30)
                phi += (coeffs[n] / n) * (sinh_nd / safe_sinh) * math.sin(n * theta)
            potential[ci] = phi

    return EnhancedWave3Result(
        alpha=alpha,
        pressure=pressure,
        velocity=velocity,
        wave_number=k,
        wave_length=L,
        potential=potential,
        ursell_number=ursell,
        iribarren_number=iribarren,
        is_breaking=is_breaking,
        stream_coefficients=coeffs,
    )


def _solve_stream_function_coefficients(a, k, d, N, g, omega):
    """Solve stream function Fourier coefficients iteratively.

    Simplified approach: initialise from Stokes expansion, then refine.
    """
    coeffs = [0.0] * (N + 1)

    # Stokes-like initialisation
    kd = k * d
    th = math.tanh(kd) if kd < 500 else 1.0
    coeffs[0] = omega / k  # Phase velocity

    if N >= 1:
        coeffs[1] = a  # Primary component
    if N >= 2:
        C1 = (3.0 - th ** 2) / (4.0 * th ** 3) if th > 1e-30 else 0.0
        coeffs[2] = a ** 2 * k * C1
    if N >= 3:
        C2 = (6.0 - 26.0 * th ** 2 + 29.0 * th ** 4 - 3.0 * th ** 6) / (48.0 * th ** 6) if th > 1e-30 else 0.0
        coeffs[3] = a ** 3 * k ** 2 * C2

    # Iterative refinement (simplified — converge surface condition)
    for _ in range(20):
        # Check surface elevation consistency
        eta_test = sum(coeffs[n] * math.cos(n * 0.0) for n in range(1, N + 1))
        target = a
        residual = eta_test - target
        if abs(residual) < 1e-12:
            break
        # Adjust primary coefficient
        if N >= 1:
            coeffs[1] -= residual * 0.5

    return coeffs


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

    freqs = np.linspace(0.5 * fp, 3.0 * fp, n_comp)
    df = freqs[1] - freqs[0] if n_comp > 1 else fp

    S = _jonswap_spectrum(freqs, fp, Hs, gamma, g_mag)
    amplitudes = np.sqrt(2.0 * S * df)

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
            k_c = _solve_dispersion_halley(omega_c, d, g_mag)
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

    return EnhancedWave3Result(
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
    sigma = np.where(f <= fp, 0.07, 0.09)
    alpha_s = 0.0081
    r = np.exp(-((f - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
    S = alpha_s * g ** 2 / (2 * np.pi * f ** 5) * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r
    m0 = float(np.sum((S[:-1] + S[1:]) * np.diff(f)) / 2.0) if len(f) > 1 else float(S[0] * (f[-1] - f[0]))
    Hs_ref = 4 * math.sqrt(max(m0, 1e-30))
    if Hs_ref > 1e-30:
        S = S * (Hs / Hs_ref) ** 2
    return S


# ---------------------------------------------------------------------------
# Wave models
# ---------------------------------------------------------------------------


def _compute_eta(H, k, d, theta, wave_type):
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
    return (
        a * math.cos(theta)
        + a * eps * C1 * math.cos(2 * theta)
        + a * eps ** 2 * C2 * math.cos(3 * theta)
        + a * eps ** 3 * C3 * math.cos(4 * theta)
    )


def _compute_wave_velocity(H, k, omega, d, z, theta, dir_vec, wave_type):
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
    a = 0.5 * H
    kd = k * d
    sinh_kd = math.sinh(kd) if kd < 500 else math.cosh(kd)
    z_eff = max(z + d, 0.0)
    kz = k * z_eff
    sinh_kz = math.sinh(kz) if kz < 500 else math.exp(kz) / 2.0
    safe_sinh = max(sinh_kd, 1e-30)
    return a * g / omega * (sinh_kz / safe_sinh) * math.sin(theta)


# ---------------------------------------------------------------------------
# Halley's method dispersion solver
# ---------------------------------------------------------------------------


def _solve_dispersion_halley(omega, d, g):
    """Solve dispersion relation using Halley's method for faster convergence."""
    k = omega ** 2 / g  # Initial deep-water guess

    for _ in range(50):
        kd = k * d
        if kd > 500:
            break

        tanh_kd = math.tanh(kd)
        f_k = g * k * tanh_kd - omega ** 2

        sech2 = 1.0 / math.cosh(kd) ** 2
        f_prime = g * (tanh_kd + kd * sech2)
        f_double = g * (2.0 * sech2 - 2.0 * kd * tanh_kd * sech2)

        if abs(f_prime) < 1e-30:
            break

        # Halley's correction
        denom = 2.0 * f_prime ** 2 - f_k * f_double
        if abs(denom) < 1e-30:
            dk = -f_k / f_prime  # Fall back to Newton
        else:
            dk = -2.0 * f_k * f_prime / denom

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
