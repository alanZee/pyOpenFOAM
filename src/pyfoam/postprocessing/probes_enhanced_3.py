"""
ProbesEnhanced3 — Enhanced probes v3 with per-probe spectral analysis.

在 Enhanced v2 基础上增加：

- **每探针频谱分析**：自动对每个探针信号计算功率谱密度
- **相干性矩阵**：多探针间的相干性矩阵
- **信号质量评估**：SNR、谱斜率、主频检测

Usage::

    probes = ProbesEnhanced3("probes3", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "computeSpectrumPerProbe": True,
        "nFFTPoints": 1024,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced_2 import (
    ProbesEnhanced2,
    CrossSpectrumResult,
    ProbeGroup,
)

__all__ = ["ProbesEnhanced3", "ProbeSpectrumResult", "CoherenceMatrix"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class ProbeSpectrumResult:
    """Spectral analysis result for a single probe.

    Attributes:
        frequencies: Frequency values ``(n_freq,)``.
        psd: Power spectral density ``(n_freq,)``.
        peak_frequency: Dominant frequency (Hz).
        spectral_slope: Log-log spectral slope.
        snr_estimate: Signal-to-noise ratio estimate.
        probe_index: Probe index.
        field_name: Field name.
        dt: Time step used.
        n_samples: Number of samples.
    """

    frequencies: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    psd: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    peak_frequency: float = 0.0
    spectral_slope: float = 0.0
    snr_estimate: float = 0.0
    probe_index: int = 0
    field_name: str = ""
    dt: float = 0.0
    n_samples: int = 0


@dataclass
class CoherenceMatrix:
    """Pairwise coherence matrix between probes.

    Attributes:
        matrix: Coherence matrix ``(n_probes, n_probes, n_freq)``.
        frequencies: Frequency axis ``(n_freq,)``.
        probe_indices: List of probe indices included.
        field_name: Field name.
    """

    matrix: Optional[torch.Tensor] = None
    frequencies: Optional[torch.Tensor] = None
    probe_indices: List[int] = field(default_factory=list)
    field_name: str = ""


class ProbesEnhanced3(ProbesEnhanced2):
    """Enhanced probes v3 with per-probe spectral analysis.

    在 ProbesEnhanced2 基础上增加的配置键：

    - ``computeSpectrumPerProbe``: auto-compute PSD for each probe (default: True)
    - ``nFFTPoints``: FFT size for spectral analysis (default: 1024)
    - ``computeCoherenceMatrix``: compute full pairwise coherence (default: False)
    """

    def __init__(
        self,
        name: str = "probesEnhanced3",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_spectrum_per_probe: bool = self.config.get(
            "computeSpectrumPerProbe", True,
        )
        self._n_fft_points: int = max(4, int(self.config.get("nFFTPoints", 1024)))
        self._compute_coherence_matrix: bool = self.config.get(
            "computeCoherenceMatrix", False,
        )

        # Storage
        self._probe_spectra: Dict[Tuple[str, int], ProbeSpectrumResult] = {}
        self._coherence_matrices: Dict[str, CoherenceMatrix] = {}

    # ------------------------------------------------------------------
    # Per-probe spectral analysis
    # ------------------------------------------------------------------

    def compute_probe_spectrum(
        self,
        field_name: str,
        probe_idx: int,
        dt: Optional[float] = None,
    ) -> Optional[ProbeSpectrumResult]:
        """Compute power spectral density for a single probe.

        Parameters
        ----------
        field_name : str
            Field name.
        probe_idx : int
            Probe index.
        dt : float, optional
            Time step. If None, computed from recorded times.

        Returns
        -------
        ProbeSpectrumResult or None
        """
        cache_key = (field_name, probe_idx)
        if cache_key in self._probe_spectra:
            return self._probe_spectra[cache_key]

        if field_name not in self._results:
            return None

        results_field = self._results[field_name]
        if probe_idx not in results_field:
            return None

        signal = results_field[probe_idx]
        n = len(signal)

        if n < 4:
            return None

        # Determine dt
        if dt is None:
            if len(self._times) >= 2:
                dt = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
            else:
                return None

        # Convert to tensor
        s = torch.tensor(signal[:min(n, self._n_fft_points)], dtype=torch.float64)
        n_use = s.numel()

        # Apply window
        s = self._apply_window(s, self._window_function)

        # FFT
        fft_s = torch.fft.rfft(s)
        psd = (fft_s.real ** 2 + fft_s.imag ** 2) / n_use
        psd = psd.clamp(min=_EPS)

        # Frequency axis
        n_freq = psd.shape[0]
        freqs = torch.arange(n_freq, dtype=torch.float64) / (n_use * dt)

        # Peak frequency
        peak_idx = int(psd.argmax().item())
        peak_freq = float(freqs[peak_idx].item())

        # Spectral slope (log-log linear fit)
        # Use frequencies > 0 to avoid log(0)
        mask = (freqs > _EPS) & (psd > _EPS)
        if mask.sum() > 2:
            log_f = torch.log(freqs[mask])
            log_psd = torch.log(psd[mask])
            # Simple linear regression: slope = cov(x,y)/var(x)
            x_mean = log_f.mean()
            y_mean = log_psd.mean()
            cov = ((log_f - x_mean) * (log_psd - y_mean)).mean()
            var = (log_f - x_mean).pow(2).mean()
            slope = float((cov / var.clamp(min=_EPS)).item())
        else:
            slope = 0.0

        # SNR estimate: ratio of peak to median
        psd_median = float(psd.median().item())
        snr = float(psd.max().item()) / max(psd_median, _EPS)

        result = ProbeSpectrumResult(
            frequencies=freqs,
            psd=psd,
            peak_frequency=peak_freq,
            spectral_slope=slope,
            snr_estimate=snr,
            probe_index=probe_idx,
            field_name=field_name,
            dt=dt,
            n_samples=n_use,
        )

        self._probe_spectra[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Coherence matrix
    # ------------------------------------------------------------------

    def compute_coherence_matrix(
        self,
        field_name: str,
    ) -> Optional[CoherenceMatrix]:
        """Compute pairwise coherence matrix between all probes.

        Parameters
        ----------
        field_name : str
            Field name.

        Returns
        -------
        CoherenceMatrix or None
        """
        if field_name in self._coherence_matrices:
            return self._coherence_matrices[field_name]

        if field_name not in self._results:
            return None

        probe_indices = sorted(self._results[field_name].keys())
        n_probes = len(probe_indices)

        if n_probes < 2:
            return None

        # Compute spectra first
        spectra = []
        for idx in probe_indices:
            s = self.compute_probe_spectrum(field_name, idx)
            if s is None:
                return None
            spectra.append(s)

        n_freq = spectra[0].frequencies.numel()
        matrix = torch.ones(n_probes, n_probes, n_freq, dtype=torch.float64)

        for i in range(n_probes):
            for j in range(i + 1, n_probes):
                cs = self.compute_cross_spectrum(
                    field_name, probe_indices[i], probe_indices[j],
                )
                if cs is not None:
                    n = min(cs.coherence.numel(), n_freq)
                    matrix[i, j, :n] = cs.coherence[:n]
                    matrix[j, i, :n] = cs.coherence[:n]

        result = CoherenceMatrix(
            matrix=matrix,
            frequencies=spectra[0].frequencies,
            probe_indices=probe_indices,
            field_name=field_name,
        )

        self._coherence_matrices[field_name] = result
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def probe_spectra(self) -> Dict[Tuple[str, int], ProbeSpectrumResult]:
        """Per-probe spectral results."""
        return self._probe_spectra

    @property
    def coherence_matrices(self) -> Dict[str, CoherenceMatrix]:
        """Coherence matrices per field."""
        return self._coherence_matrices

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write probe data, spectra, and coherence matrices."""
        super().write()

        if self._output_path is None:
            return

        # Write per-probe spectra
        for (fname, pidx), spec in self._probe_spectra.items():
            spec_file = self._output_path / f"{fname}_probe{pidx}_spectrum.dat"
            with open(spec_file, "w") as f:
                f.write(f"# Probe spectrum: field={fname}, probe={pidx}\n")
                f.write(f"# dt={spec.dt:.6e}, n_samples={spec.n_samples}\n")
                f.write(f"# Peak frequency: {spec.peak_frequency:.6e} Hz\n")
                f.write(f"# Spectral slope: {spec.spectral_slope:.4f}\n")
                f.write(f"# SNR estimate: {spec.snr_estimate:.4f}\n")
                f.write("# frequency  PSD\n")
                for i in range(spec.frequencies.numel()):
                    f.write(
                        f"{spec.frequencies[i].item():.6e}"
                        f"  {spec.psd[i].item():.6e}\n"
                    )

        # Write coherence matrices
        for fname, cm in self._coherence_matrices.items():
            if cm.matrix is None:
                continue
            cm_file = self._output_path / f"{fname}_coherence_matrix.dat"
            with open(cm_file, "w") as f:
                f.write(f"# Coherence matrix: field={fname}\n")
                f.write(f"# Probes: {cm.probe_indices}\n")
                f.write(f"# Frequencies: {cm.frequencies.numel()}\n")

        logger.info("Wrote ProbesEnhanced3 to %s", self._output_path)


# Register
from pyfoam.postprocessing.function_object import FunctionObjectRegistry
FunctionObjectRegistry.register("probesEnhanced3", ProbesEnhanced3)
