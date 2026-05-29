"""
ProbesEnhanced4 — Enhanced probes v4 with multi-probe support and spectral analysis.

在 Enhanced v3 基础上增加：

- **多探针管理器**：分层探针组管理，支持动态添加/删除
- **探针级联谱分析**：组内探针间的级联频谱
- **主频追踪**：自动追踪主频的时间变化
- **信号滤波**：带通/低通/高通数字滤波

Usage::

    probes = ProbesEnhanced4("probes4", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "trackDominantFrequency": True,
        "filterType": "lowpass",
        "filterCutoff": 100.0,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced_3 import (
    ProbesEnhanced3,
    ProbeSpectrumResult,
    CoherenceMatrix,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ProbesEnhanced4", "ProbeGroupManager", "FrequencyTracker"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class ProbeGroupManager:
    """Manages hierarchical probe groups.

    Attributes:
        groups: Dict mapping group name to list of probe indices.
        active_probes: Set of currently active probe indices.
    """

    groups: Dict[str, List[int]] = field(default_factory=dict)
    active_probes: set = field(default_factory=set)

    def add_group(self, name: str, probe_indices: List[int]) -> None:
        """Add a new probe group."""
        self.groups[name] = list(probe_indices)
        self.active_probes.update(probe_indices)

    def remove_group(self, name: str) -> None:
        """Remove a probe group."""
        if name in self.groups:
            removed = set(self.groups.pop(name))
            # Only deactivate if not in another group
            still_active = set()
            for indices in self.groups.values():
                still_active.update(indices)
            self.active_probes = still_active

    def get_group_indices(self, name: str) -> List[int]:
        """Get probe indices for a group."""
        return self.groups.get(name, [])

    def get_groups_for_probe(self, probe_idx: int) -> List[str]:
        """Get all group names containing a given probe."""
        return [name for name, indices in self.groups.items() if probe_idx in indices]


@dataclass
class FrequencyTracker:
    """Tracks dominant frequency evolution over time.

    Attributes:
        times: List of times.
        dominant_frequencies: Dominant frequency at each time.
        frequency_history: Full frequency-time map.
    """

    times: List[float] = field(default_factory=list)
    dominant_frequencies: List[float] = field(default_factory=list)
    frequency_history: Dict[float, List[float]] = field(default_factory=dict)

    def update(self, time: float, dominant_freq: float, all_freqs: Optional[List[float]] = None) -> None:
        """Update tracker with new time step data."""
        self.times.append(time)
        self.dominant_frequencies.append(dominant_freq)
        if all_freqs is not None:
            for f in all_freqs:
                if f not in self.frequency_history:
                    self.frequency_history[f] = []
                self.frequency_history[f].append(time)


class ProbesEnhanced4(ProbesEnhanced3):
    """Enhanced probes v4 with multi-probe management and signal processing.

    在 ProbesEnhanced3 基础上增加的配置键：

    - ``trackDominantFrequency``: track dominant frequency over time (default: True)
    - ``filterType``: signal filter type (``"none"``, ``"lowpass"``, ``"highpass"``, ``"bandpass"``)
    - ``filterCutoff``: filter cutoff frequency in Hz (default: None)
    - ``filterOrder``: filter order (default: 4)
    """

    def __init__(
        self,
        name: str = "probesEnhanced4",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._track_dominant_freq: bool = self.config.get(
            "trackDominantFrequency", True,
        )
        self._filter_type: str = self.config.get("filterType", "none")
        self._filter_cutoff: Optional[float] = self.config.get("filterCutoff", None)
        self._filter_order: int = max(1, int(self.config.get("filterOrder", 4)))

        # Managers
        self._group_manager = ProbeGroupManager()
        self._frequency_trackers: Dict[Tuple[str, int], FrequencyTracker] = {}

        # Initialize groups from config
        groups_cfg = self.config.get("probeGroups", [])
        for i, g in enumerate(groups_cfg):
            gname = g.get("name", f"group_{i}")
            n_probes = len(g.get("locations", []))
            self._group_manager.add_group(gname, list(range(n_probes)))

    @property
    def group_manager(self) -> ProbeGroupManager:
        """Probe group manager."""
        return self._group_manager

    @property
    def frequency_trackers(self) -> Dict[Tuple[str, int], FrequencyTracker]:
        """Frequency trackers per (field, probe)."""
        return self._frequency_trackers

    # ------------------------------------------------------------------
    # 信号滤波
    # ------------------------------------------------------------------

    def _apply_filter(self, signal: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply digital filter to a signal.

        Uses a simple moving average for lowpass, difference for highpass,
        and cascaded for bandpass.

        Parameters
        ----------
        signal : torch.Tensor
            ``(n_samples,)`` input signal.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Filtered signal.
        """
        if self._filter_type == "none" or self._filter_cutoff is None:
            return signal

        n = signal.numel()
        if n < 2:
            return signal

        # Compute filter window size from cutoff frequency
        fs = 1.0 / max(dt, _EPS)
        # Window length: fs / cutoff samples
        window = max(2, int(fs / max(self._filter_cutoff, _EPS)))
        window = min(window, n // 2)

        if self._filter_type == "lowpass":
            # Moving average lowpass
            kernel = torch.ones(window, device=signal.device, dtype=signal.dtype) / window
            # Pad signal for convolution
            padded = torch.nn.functional.pad(
                signal.unsqueeze(0).unsqueeze(0),
                (window // 2, window // 2),
                mode="replicate",
            )
            filtered = torch.nn.functional.conv1d(
                padded, kernel.unsqueeze(0).unsqueeze(0),
            ).squeeze()
            return filtered[:n]

        elif self._filter_type == "highpass":
            # Highpass = original - lowpass
            kernel = torch.ones(window, device=signal.device, dtype=signal.dtype) / window
            padded = torch.nn.functional.pad(
                signal.unsqueeze(0).unsqueeze(0),
                (window // 2, window // 2),
                mode="replicate",
            )
            lowpass = torch.nn.functional.conv1d(
                padded, kernel.unsqueeze(0).unsqueeze(0),
            ).squeeze()
            return signal - lowpass[:n]

        elif self._filter_type == "bandpass":
            # Two-pass: lowpass then highpass
            lp_cutoff = self._filter_cutoff
            hp_cutoff = self.config.get("filterCutoffLow", lp_cutoff * 0.5)
            # Highpass first
            window_hp = max(2, int(fs / max(hp_cutoff, _EPS)))
            window_hp = min(window_hp, n // 2)
            kernel_hp = torch.ones(window_hp, device=signal.device, dtype=signal.dtype) / window_hp
            padded_hp = torch.nn.functional.pad(
                signal.unsqueeze(0).unsqueeze(0),
                (window_hp // 2, window_hp // 2),
                mode="replicate",
            )
            lp_hp = torch.nn.functional.conv1d(
                padded_hp, kernel_hp.unsqueeze(0).unsqueeze(0),
            ).squeeze()
            hp_result = signal - lp_hp[:n]
            # Then lowpass
            kernel_lp = torch.ones(window, device=signal.device, dtype=signal.dtype) / window
            padded_lp = torch.nn.functional.pad(
                hp_result.unsqueeze(0).unsqueeze(0),
                (window // 2, window // 2),
                mode="replicate",
            )
            filtered = torch.nn.functional.conv1d(
                padded_lp, kernel_lp.unsqueeze(0).unsqueeze(0),
            ).squeeze()
            return filtered[:n]

        return signal

    # ------------------------------------------------------------------
    # 主频追踪
    # ------------------------------------------------------------------

    def track_dominant_frequency(
        self,
        field_name: str,
        probe_idx: int,
        time: float,
        spectrum: ProbeSpectrumResult,
    ) -> None:
        """Track dominant frequency evolution for a probe.

        Parameters
        ----------
        field_name : str
            Field name.
        probe_idx : int
            Probe index.
        time : float
            Current time.
        spectrum : ProbeSpectrumResult
            Spectral analysis result.
        """
        key = (field_name, probe_idx)
        if key not in self._frequency_trackers:
            self._frequency_trackers[key] = FrequencyTracker()

        tracker = self._frequency_trackers[key]

        # Find top-3 frequencies
        if spectrum.psd.numel() > 0:
            top_k = min(3, spectrum.psd.numel())
            _, top_indices = torch.topk(spectrum.psd, top_k)
            top_freqs = [float(spectrum.frequencies[idx].item()) for idx in top_indices]
            tracker.update(time, spectrum.peak_frequency, top_freqs)
        else:
            tracker.update(time, spectrum.peak_frequency)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute probes v4 at current time step."""
        # Run parent execute
        super().execute(time)

        if not self._enabled:
            return

        # Track dominant frequencies for each probe/field
        if self._track_dominant_freq:
            for field_name in self._results:
                probe_indices = sorted(self._results[field_name].keys())
                for pidx in probe_indices:
                    # Try to get cached spectrum or compute
                    spectrum = self.compute_probe_spectrum(field_name, pidx)
                    if spectrum is not None:
                        self.track_dominant_frequency(field_name, pidx, time, spectrum)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write probe data, spectra, coherence, and frequency tracking."""
        super().write()

        if self._output_path is None:
            return

        # Write frequency tracking
        for (fname, pidx), tracker in self._frequency_trackers.items():
            if not tracker.times:
                continue
            ft_file = self._output_path / f"{fname}_probe{pidx}_freq_tracking.dat"
            with open(ft_file, "w") as f:
                f.write(f"# Frequency tracking: field={fname}, probe={pidx}\n")
                f.write("# time  dominant_frequency\n")
                for t, freq in zip(tracker.times, tracker.dominant_frequencies):
                    f.write(f"{t:.6e}  {freq:.6e}\n")

        # Write group summary
        if self._group_manager.groups:
            group_file = self._output_path / "probeGroups.dat"
            with open(group_file, "w") as f:
                f.write("# Probe group summary\n")
                f.write("# group  n_probes  probe_indices\n")
                for gname, indices in self._group_manager.groups.items():
                    idx_str = ",".join(str(i) for i in indices)
                    f.write(f"{gname}  {len(indices)}  {idx_str}\n")

        logger.info("Wrote ProbesEnhanced4 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("probesEnhanced4", ProbesEnhanced4)
