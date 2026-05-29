"""
FieldMinMaxEnhanced4 — Enhanced field min/max v4 with per-region stats and time history.

在 Enhanced v3 基础上增加：

- **逐区域统计**：基于 mesh 区域的独立统计
- **时间历史追踪**：完整的 min/max 时间序列存储
- **变化率监测**：场值的时间导数估算
- **统计摘要**：全局摘要统计（均值、方差、趋势）

Usage::

    fmm = FieldMinMaxEnhanced4("fieldMinMax4", {
        "fields": ["p", "T"],
        "percentiles": [5, 25, 50, 75, 95],
        "perRegion": True,
        "trackTimeHistory": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_3 import (
    FieldMinMaxEnhanced3,
    PercentileStats,
    HistogramData,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced4", "RegionStats", "TimeHistoryEntry"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class RegionStats:
    """Per-region field statistics.

    Attributes:
        region_name: Region (patch) name.
        time: Simulation time.
        min: Minimum value in region.
        max: Maximum value in region.
        mean: Mean value in region.
        std: Standard deviation in region.
        n_cells: Number of cells in region.
    """

    region_name: str = ""
    time: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    n_cells: int = 0


@dataclass
class TimeHistoryEntry:
    """Single time step entry for min/max time history.

    Attributes:
        time: Simulation time.
        field_min: Global minimum value.
        field_max: Global maximum value.
        field_mean: Global mean value.
        rate_of_change: Estimated time derivative since last step.
    """

    time: float = 0.0
    field_min: float = 0.0
    field_max: float = 0.0
    field_mean: float = 0.0
    rate_of_change: float = 0.0


class FieldMinMaxEnhanced4(FieldMinMaxEnhanced3):
    """Enhanced field min/max v4 with per-region stats and time history tracking.

    在 FieldMinMaxEnhanced3 基础上增加的配置键：

    - ``perRegion``: compute per-region (per-patch) statistics (default: True)
    - ``trackTimeHistory``: store complete min/max time series (default: True)
    - ``computeRateOfChange``: compute time derivatives (default: True)
    - ``maxHistoryLength``: maximum history entries before pruning (default: 10000)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced4",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._per_region: bool = self.config.get("perRegion", True)
        self._track_time_history: bool = self.config.get("trackTimeHistory", True)
        self._compute_rate: bool = self.config.get("computeRateOfChange", True)
        self._max_history: int = max(100, int(self.config.get("maxHistoryLength", 10000)))

        # Storage
        self._region_stats_history: List[Dict[str, RegionStats]] = []
        self._time_history: List[TimeHistoryEntry] = []

    # ------------------------------------------------------------------
    # 逐区域统计
    # ------------------------------------------------------------------

    def _compute_region_stats(
        self, field, time: float,
    ) -> Dict[str, RegionStats]:
        """Compute per-region (per-patch) statistics."""
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh

        if mesh is None:
            return {}

        region_results: Dict[str, RegionStats] = {}

        data = self._extract_field_data(field)
        n_cells = data.numel()

        # Overall stats (full domain)
        region_results["__all__"] = RegionStats(
            region_name="__all__",
            time=time,
            min=float(data.min().item()),
            max=float(data.max().item()),
            mean=float(data.mean().item()),
            std=float(data.std().item()) if n_cells > 1 else 0.0,
            n_cells=n_cells,
        )

        # Per-boundary patch stats
        if hasattr(mesh, "boundary") and mesh.boundary:
            for bc in mesh.boundary:
                patch_name = bc.get("name", "unknown")
                start_face = bc.get("startFace", 0)
                n_faces = bc.get("nFaces", 0)

                # Get owner cells of patch faces
                if hasattr(mesh, "owner"):
                    face_indices = torch.arange(
                        start_face, start_face + n_faces,
                        device=device, dtype=torch.long,
                    )
                    owner_cells = mesh.owner[face_indices]
                    # Remove duplicates
                    unique_cells = torch.unique(owner_cells)
                    unique_cells = unique_cells[unique_cells < n_cells]

                    if unique_cells.numel() > 0:
                        patch_data = data[unique_cells]
                        region_results[patch_name] = RegionStats(
                            region_name=patch_name,
                            time=time,
                            min=float(patch_data.min().item()),
                            max=float(patch_data.max().item()),
                            mean=float(patch_data.mean().item()),
                            std=float(patch_data.std().item()) if patch_data.numel() > 1 else 0.0,
                            n_cells=int(unique_cells.numel()),
                        )

        return region_results

    # ------------------------------------------------------------------
    # 时间历史追踪
    # ------------------------------------------------------------------

    def _update_time_history(
        self, data: torch.Tensor, time: float,
    ) -> None:
        """Update time history with new min/max/mean values."""
        current_min = float(data.min().item())
        current_max = float(data.max().item())
        current_mean = float(data.mean().item())

        # Rate of change
        rate = 0.0
        if self._compute_rate and self._time_history:
            prev = self._time_history[-1]
            dt = time - prev.time
            if dt > _EPS:
                rate = (current_mean - prev.field_mean) / dt

        entry = TimeHistoryEntry(
            time=time,
            field_min=current_min,
            field_max=current_max,
            field_mean=current_mean,
            rate_of_change=rate,
        )

        self._time_history.append(entry)

        # Prune if too long
        if len(self._time_history) > self._max_history:
            self._time_history = self._time_history[-self._max_history:]

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute enhanced v4 min/max at the current time step."""
        if not self._enabled:
            return

        # Run base v3 execute (includes percentile, histogram, multi-field)
        super().execute(time)

        field = self._fields.get(self._field_name)
        if field is None:
            return

        data = self._extract_field_data(field)

        # Per-region statistics
        if self._per_region:
            region_stats = self._compute_region_stats(field, time)
            self._region_stats_history.append(region_stats)

        # Time history tracking
        if self._track_time_history:
            self._update_time_history(data, time)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def region_stats_history(self) -> List[Dict[str, RegionStats]]:
        """Per-region statistics history."""
        return self._region_stats_history

    @property
    def time_history(self) -> List[TimeHistoryEntry]:
        """Min/max time history."""
        return self._time_history

    def get_latest_region_stats(self, region_name: str) -> Optional[RegionStats]:
        """Get the latest region stats for a given region name."""
        if not self._region_stats_history:
            return None
        return self._region_stats_history[-1].get(region_name)

    def get_time_history_summary(self) -> Dict[str, float]:
        """Compute global time history summary."""
        if not self._time_history:
            return {}

        mins = [e.field_min for e in self._time_history]
        maxs = [e.field_max for e in self._time_history]
        means = [e.field_mean for e in self._time_history]
        rates = [e.rate_of_change for e in self._time_history]

        return {
            "global_min": min(mins),
            "global_max": max(maxs),
            "mean_of_means": sum(means) / len(means),
            "max_abs_rate": max(abs(r) for r in rates),
            "n_steps": len(self._time_history),
        }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write v4 enhanced min/max data."""
        super().write()

        if self._output_path is None:
            return

        # Write per-region stats
        if self._region_stats_history:
            region_file = self._output_path / "regionStats.dat"
            with open(region_file, "w") as f:
                f.write("# Time  region  min  max  mean  std  n_cells\n")
                for i, t in enumerate(self._times):
                    if i < len(self._region_stats_history):
                        for rn, rs in self._region_stats_history[i].items():
                            f.write(
                                f"{t:.6e}  {rn}  "
                                f"{rs.min:.10g}  {rs.max:.10g}  "
                                f"{rs.mean:.10g}  {rs.std:.10g}  "
                                f"{rs.n_cells}\n"
                            )

        # Write time history
        if self._time_history:
            history_file = self._output_path / "timeHistory.dat"
            with open(history_file, "w") as f:
                f.write("# Time  min  max  mean  rate_of_change\n")
                for entry in self._time_history:
                    f.write(
                        f"{entry.time:.6e}  "
                        f"{entry.field_min:.10g}  {entry.field_max:.10g}  "
                        f"{entry.field_mean:.10g}  "
                        f"{entry.rate_of_change:.6e}\n"
                    )

        logger.info("Wrote FieldMinMaxEnhanced4 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("fieldMinMaxEnhanced4", FieldMinMaxEnhanced4)
