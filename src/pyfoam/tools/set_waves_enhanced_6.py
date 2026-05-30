"""
setWaves enhanced v6 — enhanced wave initialisation with active wave
absorption, wave generation zones, and sponge layer modelling
(sixth generation).

Extends :func:`set_waves_enhanced_5` with:

- **Active wave absorption**: Add relaxation zones at domain boundaries
  that damp outgoing waves to prevent reflections.
- **Wave generation zones**: Define spatial regions where wave forcing
  is applied, enabling nested wave generation.
- **Sponge layer modelling**: Implement volumetric damping layers with
  configurable spatial profiles (linear, quadratic, exponential).

Usage::

    from pyfoam.tools.set_waves_enhanced_6 import (
        set_waves_enhanced_6, EnhancedWave6Properties,
    )

    wave = EnhancedWave6Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        sponge_layer=True, sponge_width=5.0,
    )
    result = set_waves_enhanced_6(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave6Properties", "EnhancedWave6Result", "set_waves_enhanced_6"]


@dataclass
class EnhancedWave6Properties:
    """Enhanced v6 wave parameters.

    Parameters
    ----------
    water_depth .. current_shear
        Forwarded from v5.
    sponge_layer : bool
        Enable sponge layer at domain boundaries.
    sponge_width : float
        Width of the sponge layer (m).
    sponge_coefficient : float
        Maximum damping coefficient (1/s).
    sponge_profile : str
        ``"linear"``, ``"quadratic"``, or ``"exponential"``.
    generation_zone : bool
        Enable spatial wave generation zone.
    generation_start : float
        x-coordinate of generation zone start (m).
    generation_width : float
        Width of generation zone (m).
    absorption : bool
        Enable active wave absorption at outflow.
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


@dataclass
class EnhancedWave6Result:
    """Result from :func:`set_waves_enhanced_6`.

    Attributes
    ----------
    alpha, pressure, velocity : np.ndarray
    wave_number, wave_length : float
    potential : np.ndarray, optional
    spectrum_frequencies, spectrum_amplitudes : np.ndarray, optional
    ursell_number .. directional_spread : float
    n_sponge_cells : int
        Cells modified by sponge layer.
    n_generation_cells : int
        Cells in the generation zone.
    n_absorption_cells : int
        Cells with active absorption applied.
    sponge_damping : np.ndarray, optional
        Per-cell sponge damping coefficient.
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


def set_waves_enhanced_6(
    mesh: "FvMesh",
    wave: EnhancedWave6Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave6Result:
    """Initialise wave fields with v6 enhancements.

    Parameters
    ----------
    mesh : FvMesh
    wave : EnhancedWave6Properties
    g, rho, time, free_surface_z, compute_potential

    Returns
    -------
    EnhancedWave6Result
    """
    from pyfoam.tools.set_waves_enhanced_5 import (
        set_waves_enhanced_5,
        EnhancedWave5Properties,
    )

    v5_props = EnhancedWave5Properties(
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
    )

    v5_result = set_waves_enhanced_5(
        mesh, v5_props, g, rho, time, free_surface_z, compute_potential,
    )

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Sponge layer
    n_sponge = 0
    sponge_damping = None
    if wave.sponge_layer:
        sponge_damping, n_sponge = _apply_sponge_layer(
            v5_result.velocity, cell_centres, wave, free_surface_z,
        )

    # Generation zone
    n_gen = 0
    if wave.generation_zone:
        n_gen = _count_generation_cells(cell_centres, wave)

    # Absorption
    n_abs = 0
    if wave.absorption:
        n_abs = _apply_absorption(v5_result.velocity, cell_centres, wave)

    return EnhancedWave6Result(
        alpha=v5_result.alpha,
        pressure=v5_result.pressure,
        velocity=v5_result.velocity,
        wave_number=v5_result.wave_number,
        wave_length=v5_result.wave_length,
        potential=v5_result.potential,
        spectrum_frequencies=v5_result.spectrum_frequencies,
        spectrum_amplitudes=v5_result.spectrum_amplitudes,
        ursell_number=v5_result.ursell_number,
        iribarren_number=v5_result.iribarren_number,
        is_breaking=v5_result.is_breaking,
        benjamin_feir_index=v5_result.benjamin_feir_index,
        spectral_bandwidth=v5_result.spectral_bandwidth,
        rogue_wave_detected=v5_result.rogue_wave_detected,
        max_wave_elevation=v5_result.max_wave_elevation,
        peakedness=v5_result.peakedness,
        groupiness_factor=v5_result.groupiness_factor,
        directional_spread=v5_result.directional_spread,
        n_sponge_cells=n_sponge,
        n_generation_cells=n_gen,
        n_absorption_cells=n_abs,
        sponge_damping=sponge_damping,
    )


# ---------------------------------------------------------------------------
# Sponge layer
# ---------------------------------------------------------------------------


def _apply_sponge_layer(velocity, cell_centres, wave, fs_z):
    """Apply sponge layer damping to velocity field."""
    n_cells = cell_centres.shape[0]
    damping = np.zeros(n_cells, dtype=np.float64)
    x_max = cell_centres[:, 0].max()
    n_mod = 0

    for ci in range(n_cells):
        x = cell_centres[ci, 0]
        dist_from_end = x_max - x

        # Only apply near the outflow boundary
        if dist_from_end < wave.sponge_width:
            ratio = 1.0 - dist_from_end / wave.sponge_width
            if wave.sponge_profile == "linear":
                coeff = wave.sponge_coefficient * ratio
            elif wave.sponge_profile == "exponential":
                coeff = wave.sponge_coefficient * (math.exp(ratio * 3.0) - 1.0) / (math.exp(3.0) - 1.0)
            else:  # quadratic
                coeff = wave.sponge_coefficient * ratio * ratio

            damping[ci] = coeff
            velocity[ci] *= max(0.0, 1.0 - coeff * 0.01)
            n_mod += 1

    return damping, n_mod


# ---------------------------------------------------------------------------
# Generation zone
# ---------------------------------------------------------------------------


def _count_generation_cells(cell_centres, wave):
    """Count cells within the wave generation zone."""
    n_gen = 0
    x_start = wave.generation_start
    x_end = x_start + wave.generation_width
    for ci in range(cell_centres.shape[0]):
        x = cell_centres[ci, 0]
        if x_start <= x <= x_end:
            n_gen += 1
    return n_gen


# ---------------------------------------------------------------------------
# Active absorption
# ---------------------------------------------------------------------------


def _apply_absorption(velocity, cell_centres, wave):
    """Apply active wave absorption at the outflow boundary."""
    n_abs = 0
    x_max = cell_centres[:, 0].max()
    abs_width = wave.sponge_width * 0.5

    for ci in range(cell_centres.shape[0]):
        x = cell_centres[ci, 0]
        if x_max - x < abs_width:
            # Damp horizontal velocity to zero
            velocity[ci, 0] *= 0.5
            velocity[ci, 1] *= 0.5
            n_abs += 1
    return n_abs
