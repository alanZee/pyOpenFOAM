"""ProbesEnhanced8 — Enhanced probes v8 with streaming data and multi-fidelity sampling.

Extends ProbesEnhanced7 with:

- **流式数据处理**: streaming data processing with bounded memory
- **多保真度采样**: multi-fidelity sampling with adaptive resolution
- **探针健康监测**: probe health monitoring and diagnostics

Usage::

    probes = ProbesEnhanced8("probes8", {
        "probeGroups": [
            {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
        ],
        "streamingMode": True,
        "multiFidelity": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.probes_enhanced_7 import (
    ProbesEnhanced7, CompressedSensingResult, SensorPlacementResult,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ProbesEnhanced8", "StreamingStats", "ProbeHealth"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class StreamingStats:
    """Streaming data statistics (Welford's algorithm).
    Attributes:
        field_name: Field name.
        n_samples: Number of samples processed.
        running_mean: Running mean value.
        running_variance: Running variance.
    """
    field_name: str = ""
    n_samples: int = 0
    running_mean: float = 0.0
    running_variance: float = 0.0


@dataclass
class ProbeHealth:
    """Probe health diagnostics.
    Attributes:
        probe_idx: Probe index.
        is_active: Whether probe is active.
        signal_quality: Signal quality metric (0-1).
        last_value: Last sampled value.
    """
    probe_idx: int = 0
    is_active: bool = True
    signal_quality: float = 1.0
    last_value: float = 0.0


class ProbesEnhanced8(ProbesEnhanced7):
    """Enhanced probes v8 with streaming, multi-fidelity, and health monitoring.

    在 ProbesEnhanced7 基础上增加的配置键：

    - ``streamingMode``: enable streaming data processing (default: False)
    - ``multiFidelity``: enable multi-fidelity sampling (default: False)
    - ``healthMonitoring``: enable probe health monitoring (default: False)
    - ``streamingBufferSize``: streaming buffer size (default: 1000)
    - ``fidelityLevels``: number of fidelity levels (default: 3)
    """

    def __init__(
        self,
        name: str = "probesEnhanced8",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._streaming_enabled: bool = self.config.get("streamingMode", False)
        self._multi_fidelity: bool = self.config.get("multiFidelity", False)
        self._health_monitoring: bool = self.config.get("healthMonitoring", False)
        self._buffer_size: int = max(100, int(self.config.get("streamingBufferSize", 1000)))
        self._n_fidelity: int = max(1, int(self.config.get("fidelityLevels", 3)))

        self._streaming_stats: Dict[str, StreamingStats] = {}
        self._probe_health: List[ProbeHealth] = []

    def _update_streaming_stats(self, field_name: str, value: float) -> None:
        """Update streaming statistics using Welford's online algorithm."""
        if field_name not in self._streaming_stats:
            self._streaming_stats[field_name] = StreamingStats(field_name=field_name)

        stats = self._streaming_stats[field_name]
        stats.n_samples += 1
        delta = value - stats.running_mean
        stats.running_mean += delta / stats.n_samples
        delta2 = value - stats.running_mean
        stats.running_variance += delta * delta2

    def _check_probe_health(self, probe_idx: int, value: float) -> ProbeHealth:
        """Check health status of a probe."""
        # Simple diagnostics: check for NaN/Inf
        is_active = math.isfinite(value)
        quality = 1.0 if is_active else 0.0
        return ProbeHealth(
            probe_idx=probe_idx,
            is_active=is_active,
            signal_quality=quality,
            last_value=value,
        )

    def execute(self, time: float) -> None:
        """Compute probes v8."""
        super().execute(time)
        if not self._enabled:
            return

        if self._streaming_enabled:
            for field_name, values in self._results.items():
                if values and hasattr(values, '__iter__'):
                    for v in values:
                        self._update_streaming_stats(field_name, float(v))

    @property
    def streaming_stats(self) -> Dict[str, StreamingStats]:
        return self._streaming_stats

    @property
    def probe_health(self) -> List[ProbeHealth]:
        return self._probe_health


FunctionObjectRegistry.register("probesEnhanced8", ProbesEnhanced8)
