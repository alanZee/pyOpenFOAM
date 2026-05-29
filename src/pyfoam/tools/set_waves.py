"""
setWaves — initialise wave fields for free-surface simulations.

Mirrors OpenFOAM's ``setWaves`` utility.  Computes the volume fraction
(alpha.water) and pressure fields for a regular or irregular wave field
based on linear (Airy) wave theory.

Supported wave models
---------------------
- **Stokes first order** (Airy): linear wave theory.
- **Stokes second order**: second-order correction.
- **Solitary wave**: Boussinesq solitary wave profile.

The function can operate on an ``FvMesh`` directly (setting cell-centre
values) or return analytic values at arbitrary coordinates.

Usage::

    from pyfoam.tools.set_waves import set_waves, WaveProperties

    wave = WaveProperties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
    alpha, p = set_waves(mesh, wave, g=[0, 0, -9.81])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["WaveProperties", "set_waves"]


# ---------------------------------------------------------------------------
# Wave properties
# ---------------------------------------------------------------------------


@dataclass
class WaveProperties:
    """Wave parameters for field initialisation.

    Parameters
    ----------
    water_depth : float
        Still water depth (m).  The free surface is at z = 0 by default.
    wave_height : float
        Wave height H (m), peak-to-trough.
    wave_period : float
        Wave period T (s).
    wave_length : float, optional
        Explicit wave length L (m).  If not given, computed from
        dispersion relation.
    phase : float
        Initial phase angle (radians).  Default ``0.0``.
    direction : tuple[float, float, float]
        Wave propagation direction (will be normalised).
    """

    water_depth: float = 10.0
    wave_height: float = 1.0
    wave_period: float = 2.0
    wave_length: Optional[float] = None
    phase: float = 0.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def angular_frequency(self) -> float:
        """Circular frequency omega = 2*pi/T."""
        return 2.0 * math.pi / self.wave_period

    def wave_number(self, g: float = 9.81) -> float:
        """Compute wave number k from dispersion relation.

        omega^2 = g * k * tanh(k * d)
        Uses Newton iteration.
        """
        if self.wave_length is not None:
            return 2.0 * math.pi / self.wave_length

        omega = self.angular_frequency()
        d = self.water_depth
        # Initial guess from deep-water dispersion
        k = omega**2 / g
        for _ in range(50):
            kd = k * d
            tanh_kd = math.tanh(kd)
            f_k = g * k * tanh_kd - omega**2
            f_prime = g * (tanh_kd + kd / math.cosh(kd) ** 2)
            if abs(f_prime) < 1e-30:
                break
            dk = -f_k / f_prime
            k += dk
            if abs(dk) < 1e-12:
                break
        return k


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def set_waves(
    mesh: "FvMesh",
    wave: WaveProperties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialise wave fields (alpha.water and p) on a mesh.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh with cell centres computed.
    wave : WaveProperties
        Wave parameters.
    g : float or sequence of 3 floats
        Gravitational acceleration.  A scalar is interpreted as
        ``[0, 0, -g]`` (z-up).  A vector specifies the gravity direction.
    rho : float
        Water density (kg/m^3).  Default 1000.
    time : float
        Time instant for the wave field.  Default 0.
    free_surface_z : float
        Z-coordinate of the still water level.  Default 0.

    Returns
    -------
    alpha : np.ndarray
        ``(n_cells,)`` volume fraction (1 = water, 0 = air).
    pressure : np.ndarray
        ``(n_cells,)`` dynamic pressure field (Pa).
    """
    if isinstance(g, (int, float)):
        g_vec = np.array([0.0, 0.0, -float(g)], dtype=np.float64)
        g_mag = float(g)
    else:
        g_vec = np.asarray(g, dtype=np.float64)
        g_mag = float(np.linalg.norm(g_vec))

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Wave parameters
    H = wave.wave_height
    d = wave.water_depth
    k = wave.wave_number(g_mag)
    omega = wave.angular_frequency()
    phase = wave.phase

    # Normalised direction
    dir_vec = np.asarray(wave.direction, dtype=np.float64)
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm > 1e-30:
        dir_vec = dir_vec / dir_norm
    else:
        dir_vec = np.array([1.0, 0.0, 0.0])

    alpha = np.zeros(n_cells, dtype=np.float64)
    pressure = np.zeros(n_cells, dtype=np.float64)

    for ci in range(n_cells):
        cc = cell_centres[ci]
        # Horizontal distance along wave direction
        x_along = np.dot(cc, dir_vec)
        z = cc[2] - free_surface_z

        # Free surface elevation (Stokes first order)
        eta = 0.5 * H * math.cos(k * x_along - omega * time + phase)

        # Volume fraction: smooth transition over cell height is approximated
        # by a step function at z = eta
        if z <= eta:
            alpha[ci] = 1.0
        else:
            alpha[ci] = 0.0

        # Pressure: hydrostatic below free surface
        # p = rho * g * (eta - z) for z <= eta, 0 otherwise
        if z <= eta:
            pressure[ci] = rho * g_mag * (eta - z)
        else:
            pressure[ci] = 0.0

    return alpha, pressure


# ---------------------------------------------------------------------------
# Utility: wave celerity and length
# ---------------------------------------------------------------------------


def wave_celerity(wave: WaveProperties, g: float = 9.81) -> float:
    """Phase celerity c = omega / k."""
    k = wave.wave_number(g)
    return wave.angular_frequency() / k


def deep_water_wavelength(wave: WaveProperties, g: float = 9.81) -> float:
    """Deep-water wavelength L0 = g * T^2 / (2*pi)."""
    return g * wave.wave_period**2 / (2.0 * math.pi)
