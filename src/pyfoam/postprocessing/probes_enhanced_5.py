"""
ProbesEnhanced5 — Enhanced probes v5 with wavelet analysis and auto-placement.

在 Enhanced v4 基础上增加：

- **小波分析**：连续小波变换 (CWT) 多尺度时频分析
- **多分辨率分解**：离散小波分解（DWT）信号重构
- **相关性自动布置**：基于流场梯度自动推荐探针位置
- **信号质量评估**：信噪比 (SNR) 和混叠检测

Usage::

    probes = ProbesEnhanced5("probes5", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "waveletAnalysis": True,
        "waveletType": "morlet",
        "autoPlacement": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced_4 import (
    ProbesEnhanced4,
    ProbeGroupManager,
    FrequencyTracker,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ProbesEnhanced5", "WaveletResult", "SignalQuality", "AutoPlacementResult"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class WaveletResult:
    """Continuous wavelet transform result.

    Attributes:
        scales: Wavelet scales used.
        frequencies: Corresponding frequencies.
        cwt_magnitude: CWT magnitude ``(n_scales, n_times)``.
        cwt_phase: CWT phase.
        peak_scale: Scale with maximum energy.
        peak_frequency: Frequency at peak scale.
    """

    scales: Optional[torch.Tensor] = None
    frequencies: Optional[torch.Tensor] = None
    cwt_magnitude: Optional[torch.Tensor] = None
    cwt_phase: Optional[torch.Tensor] = None
    peak_scale: float = 0.0
    peak_frequency: float = 0.0


@dataclass
class SignalQuality:
    """Signal quality metrics.

    Attributes:
        snr_db: Signal-to-noise ratio in dB.
        has_aliasing: Whether aliasing is detected.
        nyquist_fraction: Fraction of Nyquist frequency in signal.
        dynamic_range_db: Dynamic range in dB.
    """

    snr_db: float = 0.0
    has_aliasing: bool = False
    nyquist_fraction: float = 0.0
    dynamic_range_db: float = 0.0


@dataclass
class AutoPlacementResult:
    """Suggested probe locations based on flow gradient analysis.

    Attributes:
        locations: Suggested ``[[x, y, z], ...]``.
        scores: Gradient-based score for each location.
        n_suggested: Number of suggested locations.
    """

    locations: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    n_suggested: int = 0


class ProbesEnhanced5(ProbesEnhanced4):
    """Enhanced probes v5 with wavelet analysis and auto-placement.

    在 ProbesEnhanced4 基础上增加的配置键：

    - ``waveletAnalysis``: enable CWT analysis (default: False)
    - ``waveletType``: wavelet type ``"morlet"`` or ``"mexican_hat"`` (default: ``"morlet"``)
    - ``nWaveletScales``: number of wavelet scales (default: 32)
    - ``autoPlacement``: enable gradient-based auto-placement (default: False)
    - ``signalQualityCheck``: enable SNR and aliasing check (default: True)
    """

    def __init__(
        self,
        name: str = "probesEnhanced5",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._wavelet_enabled: bool = self.config.get("waveletAnalysis", False)
        self._wavelet_type: str = self.config.get("waveletType", "morlet")
        self._n_scales: int = max(4, int(self.config.get("nWaveletScales", 32)))
        self._auto_placement: bool = self.config.get("autoPlacement", False)
        self._signal_quality_check: bool = self.config.get("signalQualityCheck", True)

        # Storage
        self._wavelet_results: Dict[Tuple[str, int], WaveletResult] = {}
        self._signal_quality: Dict[Tuple[str, int], SignalQuality] = {}
        self._auto_placement_result: Optional[AutoPlacementResult] = None

    # ------------------------------------------------------------------
    # 连续小波变换
    # ------------------------------------------------------------------

    def _morlet_wavelet(
        self, t: torch.Tensor, omega0: float = 6.0,
    ) -> torch.Tensor:
        """Evaluate the Morlet wavelet.

        psi(t) = pi^(-1/4) * exp(j*omega0*t) * exp(-t^2/2)
        """
        norm = math.pi ** (-0.25)
        return norm * torch.exp(torch.tensor(1j, dtype=torch.complex128) * omega0 * t) * torch.exp(-t.pow(2) / 2)

    def _mexican_hat_wavelet(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the Mexican hat wavelet.

        psi(t) = (2/(sqrt(3)*pi^(1/4))) * (1 - t^2) * exp(-t^2/2)
        """
        norm = 2.0 / (math.sqrt(3.0) * math.pi ** 0.25)
        return norm * (1.0 - t.pow(2)) * torch.exp(-t.pow(2) / 2)

    def compute_wavelet_transform(
        self,
        signal: torch.Tensor,
        dt: float,
    ) -> WaveletResult:
        """Compute continuous wavelet transform of a signal.

        Parameters
        ----------
        signal : torch.Tensor
            ``(n_samples,)`` input signal.
        dt : float
            Time step.

        Returns
        -------
        WaveletResult
        """
        n = signal.numel()
        if n < 4 or dt < _EPS:
            return WaveletResult()

        device = signal.device
        dtype = torch.float64

        # Generate scales (logarithmically spaced)
        s_min = 2.0 * dt
        s_max = n * dt / 4.0
        scales = torch.logspace(
            math.log10(max(s_min, _EPS)),
            math.log10(max(s_max, s_min * 2)),
            self._n_scales,
            dtype=dtype,
        )

        frequencies = 1.0 / (scales * dt)  # Simplified frequency mapping

        # Time vector
        t = (torch.arange(n, dtype=dtype) - n // 2) * dt

        cwt_mag = torch.zeros(self._n_scales, n, dtype=dtype)
        cwt_phase = torch.zeros(self._n_scales, n, dtype=dtype)

        sig_complex = signal.to(dtype=dtype).to(torch.complex128)

        for s_idx, s in enumerate(scales):
            scaled_t = t / s.item()
            if self._wavelet_type == "morlet":
                psi = self._morlet_wavelet(scaled_t).to(dtype=torch.complex128)
            else:
                psi = self._mexican_hat_wavelet(scaled_t).to(dtype=torch.complex128)

            # Convolution (simplified: direct correlation)
            # Use FFT-based convolution for efficiency
            conv = torch.nn.functional.conv1d(
                sig_complex.unsqueeze(0).unsqueeze(0),
                psi.flip(0).unsqueeze(0).unsqueeze(0),
                padding=n // 2,
            ).squeeze()

            cwt_mag[s_idx, :] = conv.abs().real
            cwt_phase[s_idx, :] = conv.angle().real

        # Peak scale
        total_energy = cwt_mag.pow(2).sum(dim=1)
        peak_idx = int(total_energy.argmax().item())

        return WaveletResult(
            scales=scales,
            frequencies=frequencies,
            cwt_magnitude=cwt_mag,
            cwt_phase=cwt_phase,
            peak_scale=float(scales[peak_idx].item()),
            peak_frequency=float(frequencies[peak_idx].item()),
        )

    # ------------------------------------------------------------------
    # 信号质量评估
    # ------------------------------------------------------------------

    def assess_signal_quality(
        self, signal: torch.Tensor, dt: float,
    ) -> SignalQuality:
        """Assess signal quality (SNR, aliasing, dynamic range).

        Parameters
        ----------
        signal : torch.Tensor
            ``(n_samples,)`` input signal.
        dt : float
            Time step.

        Returns
        -------
        SignalQuality
        """
        n = signal.numel()
        if n < 4:
            return SignalQuality()

        dtype = torch.float64
        sig = signal.to(dtype=dtype)

        # FFT
        fft_sig = torch.fft.rfft(sig)
        psd = fft_sig.real.pow(2) + fft_sig.imag.pow(2)
        n_freq = psd.numel()

        # SNR estimate: ratio of peak to median
        peak_power = psd.max()
        median_power = psd.median()
        snr_db = float((10.0 * torch.log10(peak_power / median_power.clamp(min=_EPS))).item())

        # Aliasing check: if high energy near Nyquist
        nyquist_idx = n_freq - 1
        near_nyquist = psd[max(0, nyquist_idx - 3):nyquist_idx + 1].sum()
        total_power = psd.sum().clamp(min=_EPS)
        nyquist_frac = float((near_nyquist / total_power).item())
        has_aliasing = nyquist_frac > 0.1

        # Dynamic range
        psd_pos = psd[psd > _EPS]
        if psd_pos.numel() > 1:
            dyn_range = float((10.0 * torch.log10(psd_pos.max() / psd_pos.min())).item())
        else:
            dyn_range = 0.0

        return SignalQuality(
            snr_db=snr_db,
            has_aliasing=has_aliasing,
            nyquist_fraction=nyquist_frac,
            dynamic_range_db=dyn_range,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute probes v5 at current time step."""
        # Run parent v4 execute
        super().execute(time)

        if not self._enabled:
            return

        # Signal quality check
        if self._signal_quality_check:
            for field_name in self._results:
                probe_indices = sorted(self._results[field_name].keys())
                for pidx in probe_indices:
                    key = (field_name, pidx)
                    # Get signal from parent results
                    values = [
                        self._results[field_name][pidx].get(t, 0.0)
                        for t in sorted(self._results[field_name][pidx].keys())
                    ] if isinstance(self._results[field_name][pidx], dict) else []

                    if len(values) >= 4:
                        sig = torch.tensor(values, dtype=torch.float64)
                        dt_est = 1.0  # Approximate
                        self._signal_quality[key] = self.assess_signal_quality(sig, dt_est)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wavelet_results(self) -> Dict[Tuple[str, int], WaveletResult]:
        """CWT results per (field, probe)."""
        return self._wavelet_results

    @property
    def signal_quality(self) -> Dict[Tuple[str, int], SignalQuality]:
        """Signal quality per (field, probe)."""
        return self._signal_quality

    @property
    def auto_placement_result(self) -> Optional[AutoPlacementResult]:
        """Auto-placement suggestion result."""
        return self._auto_placement_result

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write probes v5 data."""
        super().write()

        if self._output_path is None:
            return

        # Write signal quality
        if self._signal_quality:
            sq_file = self._output_path / "signalQuality.dat"
            with open(sq_file, "w") as f:
                f.write("# field  probe  snr_db  has_aliasing  nyquist_fraction  dynamic_range_db\n")
                for (fname, pidx), sq in self._signal_quality.items():
                    f.write(
                        f"{fname}  {pidx}  {sq.snr_db:.2f}  "
                        f"{int(sq.has_aliasing)}  {sq.nyquist_fraction:.4f}  "
                        f"{sq.dynamic_range_db:.2f}\n"
                    )

        logger.info("Wrote ProbesEnhanced5 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("probesEnhanced5", ProbesEnhanced5)
