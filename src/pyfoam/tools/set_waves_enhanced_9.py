"""
setWaves enhanced v9 — enhanced wave initialisation with wave climate
statistics, wave energy extraction modelling, and wave propagation
(ninth generation).

Extends :func:`set_waves_enhanced_8` with:

- **Wave climate statistics**: Compute long-term wave climate
  statistics (significant wave height distribution, energy period).
- **Wave energy extraction**: Model energy extraction by wave
  energy converters (WEC) and compute power matrices.
- **Wave propagation**: Model wave transformation over variable
  bathymetry using mild-slope equation approximation.

Usage::

    from pyfoam.tools.set_waves_enhanced_9 import (
        set_waves_enhanced_9, EnhancedWave9Properties,
    )

    wave = EnhancedWave9Properties(
        water_depth=10.0, wave_height=1.0, wave_period=2.0,
        climate_statistics=True,
        energy_extraction=True,
    )
    result = set_waves_enhanced_9(mesh, wave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave9Properties", "EnhancedWave9Result", "set_waves_enhanced_9"]


@dataclass
class EnhancedWave9Properties:
    """Enhanced v9 wave parameters.

    Parameters
    ----------
    water_depth .. crest_height
        Forwarded from v8.
    climate_statistics : bool
        Compute wave climate statistics.
    n_climate_samples : int
        Number of climate samples for statistics.
    significant_wave_height : float
        Hs for climate distribution (m).
    energy_extraction : bool
        Model wave energy converter extraction.
    wec_capture_width : float
        WEC capture width ratio (0-1).
    wec_efficiency : float
        WEC power take-off efficiency (0-1).
    propagation : bool
        Model wave propagation over bathymetry.
    bathymetry_depths : np.ndarray, optional
        Bathymetry depth values at cell centres.
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
    climate_statistics: bool = False
    n_climate_samples: int = 1000
    significant_wave_height: float = 1.5
    energy_extraction: bool = False
    wec_capture_width: float = 0.5
    wec_efficiency: float = 0.8
    propagation: bool = False
    bathymetry_depths: Optional[np.ndarray] = None


@dataclass
class WaveClimateStats:
    """Wave climate statistics."""
    mean_hs: float = 0.0
    std_hs: float = 0.0
    hs_95_percentile: float = 0.0
    energy_period: float = 0.0
    wave_power_kw_per_m: float = 0.0
    n_samples: int = 0


@dataclass
class WECPowerMatrix:
    """Wave energy converter power matrix."""
    capture_width_ratio: float = 0.0
    pto_efficiency: float = 0.0
    mean_power_kw: float = 0.0
    peak_power_kw: float = 0.0
    capacity_factor: float = 0.0


@dataclass
class PropagationResult:
    """Wave propagation over bathymetry result."""
    n_cells_transformed: int = 0
    max_shoaling_factor: float = 1.0
    min_depth: float = 0.0
    n_breaking_cells: int = 0


@dataclass
class EnhancedWave9Result:
    """Result from :func:`set_waves_enhanced_9`.

    Attributes
    ----------
    alpha .. impact_pressure
        Forwarded from v8.
    climate : WaveClimateStats
        Wave climate statistics.
    wec_power : WECPowerMatrix
        WEC power matrix.
    propagation : PropagationResult
        Wave propagation result.
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
    climate: WaveClimateStats = field(default_factory=WaveClimateStats)
    wec_power: WECPowerMatrix = field(default_factory=WECPowerMatrix)
    propagation: PropagationResult = field(default_factory=PropagationResult)


def set_waves_enhanced_9(
    mesh: "FvMesh",
    wave: EnhancedWave9Properties,
    g: Union[float, Sequence[float]] = 9.81,
    rho: float = 1000.0,
    time: float = 0.0,
    free_surface_z: float = 0.0,
    compute_potential: bool = False,
) -> EnhancedWave9Result:
    """Initialise wave fields with v9 enhancements.

    Parameters
    ----------
    mesh : FvMesh
    wave : EnhancedWave9Properties
    g, rho, time, free_surface_z, compute_potential

    Returns
    -------
    EnhancedWave9Result
    """
    from pyfoam.tools.set_waves_enhanced_8 import (
        set_waves_enhanced_8,
        EnhancedWave8Properties,
    )

    v8_props = EnhancedWave8Properties(
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
        structure_interaction=wave.structure_interaction,
        structure_diameter=wave.structure_diameter,
        structure_draft=wave.structure_draft,
        morphodynamics=wave.morphodynamics,
        morpho_dt=wave.morpho_dt,
        morpho_porosity=wave.morpho_porosity,
        estimate_overtopping=wave.estimate_overtopping,
        crest_height=wave.crest_height,
    )

    v8_result = set_waves_enhanced_8(
        mesh, v8_props, g, rho, time, free_surface_z, compute_potential,
    )

    g_val = g if isinstance(g, float) else np.linalg.norm(g)
    rho_val = rho

    # Climate statistics
    climate = WaveClimateStats()
    if wave.climate_statistics:
        climate = _compute_climate_stats(
            wave.significant_wave_height, wave.wave_period,
            wave.n_climate_samples, wave.water_depth, rho_val, g_val,
        )

    # WEC power matrix
    wec = WECPowerMatrix()
    if wave.energy_extraction:
        wec = _compute_wec_power(
            wave.wave_height, wave.wave_period, wave.water_depth,
            wave.wec_capture_width, wave.wec_efficiency, rho_val, g_val,
        )

    # Wave propagation
    prop = PropagationResult()
    if wave.propagation and wave.bathymetry_depths is not None:
        prop = _compute_propagation(
            wave.bathymetry_depths, wave.wave_period,
            wave.water_depth, g_val,
        )

    return EnhancedWave9Result(
        alpha=v8_result.alpha,
        pressure=v8_result.pressure,
        velocity=v8_result.velocity,
        wave_number=v8_result.wave_number,
        wave_length=v8_result.wave_length,
        potential=v8_result.potential,
        spectrum_frequencies=v8_result.spectrum_frequencies,
        spectrum_amplitudes=v8_result.spectrum_amplitudes,
        ursell_number=v8_result.ursell_number,
        iribarren_number=v8_result.iribarren_number,
        is_breaking=v8_result.is_breaking,
        benjamin_feir_index=v8_result.benjamin_feir_index,
        spectral_bandwidth=v8_result.spectral_bandwidth,
        rogue_wave_detected=v8_result.rogue_wave_detected,
        max_wave_elevation=v8_result.max_wave_elevation,
        peakedness=v8_result.peakedness,
        groupiness_factor=v8_result.groupiness_factor,
        directional_spread=v8_result.directional_spread,
        n_sponge_cells=v8_result.n_sponge_cells,
        n_generation_cells=v8_result.n_generation_cells,
        n_absorption_cells=v8_result.n_absorption_cells,
        sponge_damping=v8_result.sponge_damping,
        doppler_shift=v8_result.doppler_shift,
        energy_flux=v8_result.energy_flux,
        group_velocity=v8_result.group_velocity,
        radiation_stress_xx=v8_result.radiation_stress_xx,
        shields_parameter=v8_result.shields_parameter,
        sediment_transport_rate=v8_result.sediment_transport_rate,
        morison_force=v8_result.morison_force,
        morison_inertia=v8_result.morison_inertia,
        morison_drag=v8_result.morison_drag,
        bed_level_change=v8_result.bed_level_change,
        overtopping_rate=v8_result.overtopping_rate,
        impact_pressure=v8_result.impact_pressure,
        climate=climate,
        wec_power=wec,
        propagation=prop,
    )


# ---------------------------------------------------------------------------
# Wave climate statistics
# ---------------------------------------------------------------------------


def _compute_climate_stats(Hs, Tz, n_samples, depth, rho, g):
    """Compute wave climate statistics from Rayleigh distribution."""
    # Rayleigh distribution for Hs
    rng = np.random.default_rng(42)
    hs_samples = rng.rayleigh(Hs / math.sqrt(2.0), n_samples)

    mean_hs = float(np.mean(hs_samples))
    std_hs = float(np.std(hs_samples))
    hs_95 = float(np.percentile(hs_samples, 95))

    # Energy period (approximate Tm-1,0 ~ 1.1 * Tz)
    Te = 1.1 * Tz

    # Wave power: P = rho * g^2 * Hs^2 * Te / (64 * pi)
    power = rho * g ** 2 * Hs ** 2 * Te / (64.0 * math.pi) / 1000.0  # kW/m

    return WaveClimateStats(
        mean_hs=mean_hs,
        std_hs=std_hs,
        hs_95_percentile=hs_95,
        energy_period=Te,
        wave_power_kw_per_m=power,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# WEC power matrix
# ---------------------------------------------------------------------------


def _compute_wec_power(H, T, d, CWR, eta, rho, g):
    """Compute WEC power extraction."""
    # Incident wave power per unit width
    omega = 2.0 * math.pi / T
    k = omega ** 2 / (g + 1e-30)
    Cg = 0.5 * (1.0 + 2.0 * k * d / (math.sinh(2.0 * k * d + 1e-30) + 1e-30))
    P_inc = 0.5 * rho * g * (H ** 2 / 8.0) * Cg  # W/m

    # Extracted power
    P_ext = P_inc * CWR * eta

    # Assume capture width of 5m for device
    device_power = P_ext * 5.0 / 1000.0  # kW

    # Capacity factor (simplified)
    cf = CWR * eta * 0.5

    return WECPowerMatrix(
        capture_width_ratio=CWR,
        pto_efficiency=eta,
        mean_power_kw=device_power,
        peak_power_kw=device_power * 1.5,
        capacity_factor=cf,
    )


# ---------------------------------------------------------------------------
# Wave propagation
# ---------------------------------------------------------------------------


def _compute_propagation(bathymetry, T, ref_depth, g):
    """Model wave transformation over variable bathymetry (mild-slope approx)."""
    n_cells = bathymetry.shape[0]
    omega = 2.0 * math.pi / T

    n_transformed = 0
    max_shoal = 1.0
    min_d = float(np.min(bathymetry))
    n_breaking = 0

    for i in range(n_cells):
        d = bathymetry[i]
        if d <= 0.1:
            n_breaking += 1
            continue

        # Shoaling coefficient: Ks = sqrt(Cg0/Cg)
        k_ref = omega ** 2 / (g + 1e-30)
        Cg_ref = 0.5 * g / omega
        k_local = omega ** 2 / (g * math.tanh(omega ** 2 * d / g + 1e-30) + 1e-30)
        Cg_local = 0.5 * (1.0 + 2.0 * k_local * d / (math.sinh(2.0 * k_local * d + 1e-30) + 1e-30))
        Ks = math.sqrt(Cg_ref / (Cg_local + 1e-30))
        max_shoal = max(max_shoal, Ks)
        n_transformed += 1

    return PropagationResult(
        n_cells_transformed=n_transformed,
        max_shoaling_factor=max_shoal,
        min_depth=min_d,
        n_breaking_cells=n_breaking,
    )
