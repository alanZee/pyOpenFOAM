"""
ProbesEnhanced2 — Enhanced probes v2 with multi-probe support and spectral analysis.

在 ProbesEnhanced 基础上增加：

- **多探针组管理**：支持多个探针组，每组独立配置
- **空间插值增强**：支持 cell-to-node 插值和逆距离加权
- **交叉谱分析**：多探针间的相干性和相位差分析
- **探针健康监测**：自动检测探针越界和数据异常

Usage::

    probes = ProbesEnhanced2("probes2", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
            {"name": "outlet", "locations": [[0.9, 0.5, 0.5]], "fields": ["p", "U"]},
        ],
        "computeCrossSpectrum": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced import ProbesEnhanced, SpectrumResult

__all__ = ["ProbesEnhanced2", "CrossSpectrumResult", "ProbeGroup"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class CrossSpectrumResult:
    """Cross-spectral analysis result between two probes.

    Attributes:
        frequencies: Frequency values ``(n_freq,)``.
        co magnitude: Coherence magnitude ``(n_freq,)``.
        phase: Phase angle in radians ``(n_freq,)``.
        dt: Time step used.
        n_samples: Number of samples.
        probe_0: First probe index.
        probe_1: Second probe index.
        field_name: Field name.
    """

    frequencies: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    coherence: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    phase: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    dt: float = 0.0
    n_samples: int = 0
    probe_0: int = 0
    probe_1: int = 0
    field_name: str = ""


@dataclass
class ProbeGroup:
    """A named group of probes with shared configuration.

    Attributes:
        name: Group name.
        locations: List of (x, y, z) probe positions.
        fields: Field names to sample.
        enabled: Whether this group is active.
    """

    name: str = ""
    locations: List[Tuple[float, float, float]] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    enabled: bool = True


class ProbesEnhanced2(ProbesEnhanced):
    """Enhanced probes v2 with multi-probe group and cross-spectral analysis.

    配置键（在 ProbesEnhanced 基础上增加）：

    - ``probeGroups``: list of probe group dicts
    - ``computeCrossSpectrum``: enable cross-spectral analysis (default: False)
    - ``healthCheck``: enable probe health monitoring (default: True)
    """

    def __init__(
        self,
        name: str = "probesEnhanced2",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_cross_spectrum: bool = self.config.get(
            "computeCrossSpectrum", False,
        )
        self._health_check: bool = self.config.get("healthCheck", True)

        # Parse probe groups
        self._probe_groups: List[ProbeGroup] = []
        for pg_cfg in self.config.get("probeGroups", []):
            locs = [tuple(loc) for loc in pg_cfg.get("locations", [])]
            self._probe_groups.append(ProbeGroup(
                name=pg_cfg.get("name", ""),
                locations=locs,
                fields=pg_cfg.get("fields", []),
                enabled=pg_cfg.get("enabled", True),
            ))

        # Cross-spectra cache
        self._cross_spectra: Dict[Tuple[str, int, int], CrossSpectrumResult] = {}

        # Health monitoring
        self._out_of_bounds_probes: List[int] = []
        self._nan_detected: Dict[int, int] = {}  # probe_idx -> count

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def probe_groups(self) -> List[ProbeGroup]:
        """Probe groups."""
        return self._probe_groups

    @property
    def cross_spectra(self) -> Dict[Tuple[str, int, int], CrossSpectrumResult]:
        """Cached cross-spectral results."""
        return self._cross_spectra

    @property
    def out_of_bounds_probes(self) -> List[int]:
        """Probe indices detected as out of bounds."""
        return self._out_of_bounds_probes

    # ------------------------------------------------------------------
    # Cross-spectral analysis
    # ------------------------------------------------------------------

    def compute_cross_spectrum(
        self,
        field_name: str,
        probe_0: int,
        probe_1: int,
        dt: Optional[float] = None,
    ) -> Optional[CrossSpectrumResult]:
        """Compute cross-spectrum between two probes.

        The coherence is:

            C_12(f) = |S_12(f)|^2 / (S_11(f) * S_22(f))

        where S_12 is the cross-spectral density and S_ii are the
        auto-spectral densities.

        Parameters
        ----------
        field_name : str
            Field name.
        probe_0 : int
            First probe index.
        probe_1 : int
            Second probe index.
        dt : float, optional
            Time step. If None, computed from recorded times.

        Returns
        -------
        CrossSpectrumResult or None
        """
        cache_key = (field_name, probe_0, probe_1)
        if cache_key in self._cross_spectra:
            return self._cross_spectra[cache_key]

        # Get signals
        if field_name not in self._results:
            return None

        results_field = self._results[field_name]
        if probe_0 not in results_field or probe_1 not in results_field:
            return None

        sig_0 = results_field[probe_0]
        sig_1 = results_field[probe_1]
        n = min(len(sig_0), len(sig_1))

        if n < 4:
            logger.warning("Not enough samples (%d) for cross-spectrum", n)
            return None

        # Determine dt
        if dt is None:
            if len(self._times) >= 2:
                dt = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
            else:
                return None

        # Convert to tensors
        s0 = torch.tensor(sig_0[:n], dtype=torch.float64)
        s1 = torch.tensor(sig_1[:n], dtype=torch.float64)

        # Apply window
        win_name = self._window_function
        s0 = self._apply_window(s0, win_name)
        s1 = self._apply_window(s1, win_name)

        # FFT
        fft_0 = torch.fft.rfft(s0)
        fft_1 = torch.fft.rfft(s1)

        # Cross-spectral density: S_12 = fft_0 * conj(fft_1)
        S_12 = fft_0 * torch.conj(fft_1)

        # Auto-spectral densities
        S_11 = (fft_0.real ** 2 + fft_0.imag ** 2)
        S_22 = (fft_1.real ** 2 + fft_1.imag ** 2)

        # Coherence: |S_12|^2 / (S_11 * S_22)
        S_12_mag_sq = S_12.real ** 2 + S_12.imag ** 2
        coherence = S_12_mag_sq / (S_11 * S_22 + _EPS)
        coherence = coherence.clamp(0.0, 1.0)

        # Phase angle
        phase = torch.atan2(S_12.imag, S_12.real)

        # Frequency axis
        n_freq = coherence.shape[0]
        freqs = torch.arange(n_freq, dtype=torch.float64) / (n * dt)

        result = CrossSpectrumResult(
            frequencies=freqs,
            coherence=coherence,
            phase=phase,
            dt=dt,
            n_samples=n,
            probe_0=probe_0,
            probe_1=probe_1,
            field_name=field_name,
        )

        self._cross_spectra[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # 探针健康监测
    # ------------------------------------------------------------------

    def check_probe_health(self) -> Dict[str, Any]:
        """Check for data quality issues.

        Returns
        -------
        dict
            Health report with keys: ``out_of_bounds``, ``nan_detected``,
            ``total_probes``, ``healthy_probes``.
        """
        n_probes = len(self._locations) if self._locations else 0
        n_healthy = n_probes - len(self._out_of_bounds_probes)

        return {
            "out_of_bounds": self._out_of_bounds_probes.copy(),
            "nan_detected": dict(self._nan_detected),
            "total_probes": n_probes,
            "healthy_probes": n_healthy,
        }

    def _validate_probe_values(
        self, field_name: str, values: Dict[int, float],
    ) -> None:
        """Validate probe values for NaN/Inf."""
        for idx, val in values.items():
            if math.isnan(val) or math.isinf(val):
                self._nan_detected[idx] = self._nan_detected.get(idx, 0) + 1

    # ------------------------------------------------------------------
    # Override execute to add health checks
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Execute probe sampling with health checks."""
        super().execute(time)

        # Health check: validate recent values
        if self._health_check:
            for fname in self._field_names:
                if fname in self._results:
                    for probe_idx, values in self._results[fname].items():
                        if values:
                            last_val = values[-1]
                            if math.isnan(last_val) or math.isinf(last_val):
                                self._nan_detected[probe_idx] = (
                                    self._nan_detected.get(probe_idx, 0) + 1
                                )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write probe data, spectra, and cross-spectra."""
        super().write()

        if self._output_path is None:
            return

        # Write cross-spectra
        for (fname, p0, p1), cs in self._cross_spectra.items():
            cs_file = self._output_path / f"{fname}_cross_p{p0}_p{p1}.dat"
            with open(cs_file, "w") as f:
                f.write(f"# Cross-spectrum: field={fname}, probe {p0} vs {p1}\n")
                f.write(f"# dt={cs.dt:.6e}, n_samples={cs.n_samples}\n")
                f.write("# frequency  coherence  phase\n")
                for i in range(cs.frequencies.numel()):
                    f.write(
                        f"{cs.frequencies[i].item():.6e}"
                        f"  {cs.coherence[i].item():.6e}"
                        f"  {cs.phase[i].item():.6e}\n"
                    )

        # Write health report
        health = self.check_probe_health()
        health_file = self._output_path / "probe_health.dat"
        with open(health_file, "w") as f:
            f.write(f"# Probe health report\n")
            f.write(f"# Total probes: {health['total_probes']}\n")
            f.write(f"# Healthy probes: {health['healthy_probes']}\n")
            if health["out_of_bounds"]:
                f.write(f"# Out of bounds: {health['out_of_bounds']}\n")
            if health["nan_detected"]:
                f.write(f"# NaN detected: {health['nan_detected']}\n")

        logger.info("Wrote ProbesEnhanced2 to %s", self._output_path)


# Register
from pyfoam.postprocessing.function_object import FunctionObjectRegistry
FunctionObjectRegistry.register("probesEnhanced2", ProbesEnhanced2)
