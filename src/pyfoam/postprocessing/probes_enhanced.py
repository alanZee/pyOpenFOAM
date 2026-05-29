"""
ProbesEnhanced — enhanced probes with time interpolation and spectral analysis.

Extends :class:`~pyfoam.postprocessing.sampling.Probes` with:

- Time interpolation at probe locations (linear and cubic)
- Spectral analysis (FFT power spectrum) at probe locations
- Windowed Fourier analysis for non-stationary signals

Usage::

    probes = ProbesEnhanced("probes1", {
        "fields": ["p"],
        "probeLocations": [(0.5, 0.5, 0.5)],
        "interpolationScheme": "cellCentre",
    })
    probes.initialise(mesh, fields)
    probes.execute(0.001)
    probes.execute(0.002)
    # ...
    spectrum = probes.compute_spectrum("p", probe_idx=0, dt=0.001)

References
----------
- OpenFOAM ``probes`` function object source
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.sampling import Probes

__all__ = ["ProbesEnhanced", "SpectrumResult"]

logger = logging.getLogger(__name__)


@dataclass
class SpectrumResult:
    """Result of spectral analysis at a probe.

    Attributes:
        frequencies: Frequency values, shape ``(n_freq,)``.
        power: Power spectral density, shape ``(n_freq,)``.
        dt: Time step used.
        n_samples: Number of samples in the analysis.
        probe_idx: Probe index.
        field_name: Field name analysed.
    """

    frequencies: torch.Tensor
    power: torch.Tensor
    dt: float
    n_samples: int
    probe_idx: int
    field_name: str


class ProbesEnhanced(Probes):
    """Enhanced probes with time interpolation and spectral analysis.

    Configuration keys (in addition to :class:`Probes`):

    - ``computeSpectrum``: enable spectral analysis (default: False)
    - ``windowFunction``: FFT window function (``"hanning"``, ``"hamming"``, ``"none"``)
    """

    def __init__(
        self,
        name: str = "probes",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_spectrum: bool = self.config.get("computeSpectrum", False)
        self._window_function: str = self.config.get("windowFunction", "hanning")

        # Cached spectra
        self._spectra: Dict[Tuple[str, int], SpectrumResult] = {}

    # ------------------------------------------------------------------
    # Time interpolation
    # ------------------------------------------------------------------

    def interpolate_at_time(
        self,
        field_name: str,
        probe_idx: int,
        query_time: float,
        method: str = "linear",
    ) -> Optional[float]:
        """Interpolate probe data at a specific time.

        Args:
            field_name: Field name.
            probe_idx: Probe index.
            query_time: Time to interpolate at.
            method: ``"linear"`` or ``"cubic"`` interpolation.

        Returns:
            Interpolated scalar value, or None if data is unavailable.
        """
        if field_name not in self._results:
            return None
        if probe_idx not in self._results[field_name]:
            return None

        times = self._times
        values = self._results[field_name][probe_idx]

        if len(times) < 2:
            return values[0] if values else None

        # Clamp to range
        if query_time <= times[0]:
            return values[0]
        if query_time >= times[-1]:
            return values[-1]

        # Find bracketing indices
        idx = 0
        for i in range(len(times) - 1):
            if times[i] <= query_time <= times[i + 1]:
                idx = i
                break

        if method == "cubic" and len(times) >= 4:
            return self._cubic_interpolate(times, values, query_time, idx)
        else:
            return self._linear_interpolate(times, values, query_time, idx)

    def _linear_interpolate(
        self,
        times: List[float],
        values: List[float],
        t: float,
        idx: int,
    ) -> float:
        """Linear interpolation between two time points."""
        if idx + 1 >= len(times):
            return values[idx]

        t0, t1 = times[idx], times[idx + 1]
        v0, v1 = values[idx], values[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return v0 + alpha * (v1 - v0)

    def _cubic_interpolate(
        self,
        times: List[float],
        values: List[float],
        t: float,
        idx: int,
    ) -> float:
        """Cubic (Catmull-Rom) interpolation."""
        n = len(times)
        # Get 4 points: idx-1, idx, idx+1, idx+2
        i0 = max(0, idx - 1)
        i1 = idx
        i2 = min(n - 1, idx + 1)
        i3 = min(n - 1, idx + 2)

        t0, t1, t2, t3 = times[i0], times[i1], times[i2], times[i3]
        v0, v1, v2, v3 = values[i0], values[i1], values[i2], values[i3]

        # Normalised parameter
        tau = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
        tau2 = tau * tau
        tau3 = tau2 * tau

        # Catmull-Rom spline
        result = (
            0.5 * (
                (2 * v1)
                + (-v0 + v2) * tau
                + (2 * v0 - 5 * v1 + 4 * v2 - v3) * tau2
                + (-v0 + 3 * v1 - 3 * v2 + v3) * tau3
            )
        )
        return result

    def interpolate_at_times(
        self,
        field_name: str,
        probe_idx: int,
        query_times: List[float],
        method: str = "linear",
    ) -> List[Optional[float]]:
        """Interpolate probe data at multiple times.

        Args:
            field_name: Field name.
            probe_idx: Probe index.
            query_times: List of query times.
            method: Interpolation method.

        Returns:
            List of interpolated values (None for out-of-range).
        """
        return [
            self.interpolate_at_time(field_name, probe_idx, t, method)
            for t in query_times
        ]

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def compute_spectrum(
        self,
        field_name: str,
        probe_idx: int,
        dt: Optional[float] = None,
        window: Optional[str] = None,
    ) -> Optional[SpectrumResult]:
        """Compute power spectrum of a probe signal.

        Uses FFT to compute the power spectral density.  Applies
        an optional window function before the FFT.

        Args:
            field_name: Field name.
            probe_idx: Probe index.
            dt: Time step. If None, computed from recorded times.
            window: Window function name. If None, uses config setting.

        Returns:
            :class:`SpectrumResult` or None if data is insufficient.
        """
        cache_key = (field_name, probe_idx)
        if cache_key in self._spectra:
            return self._spectra[cache_key]

        if field_name not in self._results:
            return None
        if probe_idx not in self._results[field_name]:
            return None

        values = self._results[field_name][probe_idx]
        n = len(values)

        if n < 4:
            logger.warning("Not enough samples (%d) for spectral analysis", n)
            return None

        # Determine dt
        if dt is None:
            if len(self._times) >= 2:
                dt = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
            else:
                logger.warning("Cannot determine dt from recorded times")
                return None

        # Apply window function
        win_name = window or self._window_function
        signal = torch.tensor(values, dtype=torch.float64)
        signal = self._apply_window(signal, win_name)

        # FFT
        fft_result = torch.fft.rfft(signal)
        power = (fft_result.real ** 2 + fft_result.imag ** 2) / n

        # Frequency axis
        n_freq = power.shape[0]
        freqs = torch.arange(n_freq, dtype=torch.float64) / (n * dt)

        result = SpectrumResult(
            frequencies=freqs,
            power=power,
            dt=dt,
            n_samples=n,
            probe_idx=probe_idx,
            field_name=field_name,
        )

        self._spectra[cache_key] = result
        return result

    def compute_all_spectra(
        self,
        dt: Optional[float] = None,
    ) -> Dict[Tuple[str, int], SpectrumResult]:
        """Compute spectra for all fields and probes.

        Args:
            dt: Time step.

        Returns:
            Dict mapping ``(field_name, probe_idx)`` -> :class:`SpectrumResult`.
        """
        for fname in self._field_names:
            for probe_idx in range(len(self._locations)):
                self.compute_spectrum(fname, probe_idx, dt=dt)
        return self._spectra

    def get_peak_frequency(
        self,
        field_name: str,
        probe_idx: int,
    ) -> Optional[float]:
        """Get the dominant frequency from the spectrum.

        Args:
            field_name: Field name.
            probe_idx: Probe index.

        Returns:
            Peak frequency in Hz, or None if spectrum not computed.
        """
        spec = self.compute_spectrum(field_name, probe_idx)
        if spec is None or spec.frequencies.numel() < 2:
            return None

        # Skip DC component (index 0)
        peak_idx = spec.power[1:].argmax().item() + 1
        return spec.frequencies[peak_idx].item()

    def _apply_window(self, signal: torch.Tensor, window_name: str) -> torch.Tensor:
        """Apply a window function to the signal.

        Args:
            signal: Input signal tensor.
            window_name: ``"hanning"``, ``"hamming"``, or ``"none"``.

        Returns:
            Windowed signal.
        """
        n = signal.shape[0]

        if window_name == "hanning":
            k = torch.arange(n, dtype=torch.float64)
            window = 0.5 * (1 - torch.cos(2 * math.pi * k / (n - 1)))
        elif window_name == "hamming":
            k = torch.arange(n, dtype=torch.float64)
            window = 0.54 - 0.46 * torch.cos(2 * math.pi * k / (n - 1))
        else:
            return signal

        return signal * window

    # ------------------------------------------------------------------
    # Write enhanced data
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write probe data and spectra."""
        super().write()

        if self._output_path is None:
            return

        # Write spectra
        for (fname, probe_idx), spec in self._spectra.items():
            spec_file = self._output_path / f"{fname}_probe{probe_idx}_spectrum.dat"
            with open(spec_file, "w") as f:
                f.write(f"# Power spectrum: field={fname}, probe={probe_idx}\n")
                f.write(f"# dt={spec.dt:.6e}, n_samples={spec.n_samples}\n")
                f.write("# frequency  power\n")
                for i in range(spec.frequencies.numel()):
                    f.write(
                        f"{spec.frequencies[i].item():.6e}"
                        f"  {spec.power[i].item():.6e}\n"
                    )
            logger.info("Wrote spectrum to %s", spec_file)
