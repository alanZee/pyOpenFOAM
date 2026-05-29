"""
FieldMinMaxEnhanced3 — Enhanced field min/max v3 with percentile and histogram.

在 Enhanced v2 基础上增加：

- **百分位数统计**：P5, P25, P50, P75, P95 等分位数
- **直方图统计**：值域分布直方图
- **异常值检测**：基于 IQR 的离群点标记
- **多场统计**：同时统计多个标量场

Usage::

    fmm = FieldMinMaxEnhanced3("fieldMinMax3", {
        "fields": ["p", "T"],
        "percentiles": [5, 25, 50, 75, 95],
        "nHistogramBins": 50,
        "outlierDetection": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced_2 import (
    FieldMinMaxEnhanced2,
    RegionMinMaxResult,
    ConvergenceInfo,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced3", "PercentileStats", "HistogramData"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class PercentileStats:
    """Percentile statistics for a field quantity.

    Attributes:
        field_name: Field name.
        time: Simulation time.
        percentiles: Dict mapping percentile level to value.
        iqr: Inter-quartile range (P75 - P25).
        outlier_count: Number of outlier cells detected.
    """

    field_name: str = ""
    time: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)
    iqr: float = 0.0
    outlier_count: int = 0


@dataclass
class HistogramData:
    """Histogram of field values.

    Attributes:
        field_name: Field name.
        time: Simulation time.
        bin_edges: Left edges of histogram bins.
        counts: Number of cells per bin.
        bin_width: Width of each bin.
    """

    field_name: str = ""
    time: float = 0.0
    bin_edges: Optional[torch.Tensor] = None
    counts: Optional[torch.Tensor] = None
    bin_width: float = 0.0


class FieldMinMaxEnhanced3(FieldMinMaxEnhanced2):
    """Enhanced field min/max v3 with percentile, histogram, and outlier detection.

    在 FieldMinMaxEnhanced2 基础上增加的配置键：

    - ``percentiles``: list of percentile levels (default: [5, 25, 50, 75, 95])
    - ``nHistogramBins``: number of histogram bins (default: 50)
    - ``outlierDetection``: enable IQR-based outlier detection (default: True)
    - ``multiFields``: list of additional field names to track
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced3",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._percentile_levels: List[int] = self.config.get(
            "percentiles", [5, 25, 50, 75, 95],
        )
        self._n_histogram_bins: int = max(5, int(self.config.get("nHistogramBins", 50)))
        self._outlier_detection: bool = self.config.get("outlierDetection", True)

        # Multi-field support
        extra_fields = self.config.get("multiFields", [])
        self._multi_field_names: List[str] = list(extra_fields)

        # Storage
        self._percentile_history: List[PercentileStats] = []
        self._histogram_history: List[HistogramData] = []

        # Per-field storage for multi-field
        self._multi_field_results: Dict[str, List[PercentileStats]] = {
            f: [] for f in extra_fields
        }

    def execute(self, time: float) -> None:
        """Compute enhanced v3 min/max at the current time step."""
        if not self._enabled:
            return

        # Run base v2 execute
        super().execute(time)

        field = self._fields.get(self._field_name)
        if field is None:
            return

        data = self._extract_field_data(field)

        # Percentile statistics
        pct_stats = self._compute_percentiles(data, self._field_name, time)
        self._percentile_history.append(pct_stats)

        # Histogram
        hist = self._compute_histogram(data, self._field_name, time)
        self._histogram_history.append(hist)

        # Multi-field analysis
        for fname in self._multi_field_names:
            mfield = self._fields.get(fname)
            if mfield is not None:
                mdata = self._extract_field_data(mfield)
                m_pct = self._compute_percentiles(mdata, fname, time)
                self._multi_field_results[fname].append(m_pct)

    def _extract_field_data(self, field) -> torch.Tensor:
        """Extract tensor data from a field."""
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(field, "internal_field"):
            data = field.internal_field.to(device=device, dtype=dtype)
        elif hasattr(field, "data"):
            data = field.data.to(device=device, dtype=dtype)
        else:
            data = field.to(device=device, dtype=dtype)

        # Flatten to 1D (scalar) or norm (vector)
        if data.dim() > 1:
            data = data.norm(dim=1)

        return data

    def _compute_percentiles(
        self, data: torch.Tensor, field_name: str, time: float,
    ) -> PercentileStats:
        """Compute percentile statistics."""
        sorted_data = data.sort().values
        n = sorted_data.numel()

        percentiles: Dict[int, float] = {}
        for p in self._percentile_levels:
            idx = int(p / 100.0 * (n - 1))
            idx = max(0, min(idx, n - 1))
            percentiles[p] = float(sorted_data[idx].item())

        iqr = percentiles.get(75, 0.0) - percentiles.get(25, 0.0)

        # Outlier detection
        outlier_count = 0
        if self._outlier_detection and iqr > _EPS:
            lower_fence = percentiles.get(25, 0.0) - 1.5 * iqr
            upper_fence = percentiles.get(75, 0.0) + 1.5 * iqr
            outliers = (data < lower_fence) | (data > upper_fence)
            outlier_count = int(outliers.sum().item())

        return PercentileStats(
            field_name=field_name,
            time=time,
            percentiles=percentiles,
            iqr=iqr,
            outlier_count=outlier_count,
        )

    def _compute_histogram(
        self, data: torch.Tensor, field_name: str, time: float,
    ) -> HistogramData:
        """Compute histogram of field values."""
        data_min = float(data.min().item())
        data_max = float(data.max().item())

        if abs(data_max - data_min) < _EPS:
            # All values are the same
            bin_edges = torch.tensor([data_min - 0.5, data_min + 0.5])
            counts = torch.tensor([data.numel()], dtype=torch.long)
            return HistogramData(
                field_name=field_name,
                time=time,
                bin_edges=bin_edges,
                counts=counts,
                bin_width=1.0,
            )

        bin_width = (data_max - data_min) / self._n_histogram_bins
        bin_edges = torch.linspace(
            data_min, data_max, self._n_histogram_bins + 1,
            device=data.device, dtype=data.dtype,
        )

        # Compute histogram manually
        counts = torch.zeros(self._n_histogram_bins, dtype=torch.long, device=data.device)
        for i in range(self._n_histogram_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            if i == self._n_histogram_bins - 1:
                mask = (data >= lower) & (data <= upper)
            else:
                mask = (data >= lower) & (data < upper)
            counts[i] = mask.sum()

        return HistogramData(
            field_name=field_name,
            time=time,
            bin_edges=bin_edges.detach().cpu(),
            counts=counts.detach().cpu(),
            bin_width=float(bin_width),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def percentile_history(self) -> List[PercentileStats]:
        """Percentile statistics history."""
        return self._percentile_history

    @property
    def histogram_history(self) -> List[HistogramData]:
        """Histogram history."""
        return self._histogram_history

    @property
    def multi_field_results(self) -> Dict[str, List[PercentileStats]]:
        """Multi-field percentile results."""
        return self._multi_field_results

    def get_latest_percentiles(self) -> Optional[PercentileStats]:
        """Get the latest percentile statistics."""
        if not self._percentile_history:
            return None
        return self._percentile_history[-1]

    def get_latest_histogram(self) -> Optional[HistogramData]:
        """Get the latest histogram."""
        if not self._histogram_history:
            return None
        return self._histogram_history[-1]

    def write(self) -> None:
        """Write v3 enhanced min/max data."""
        super().write()

        if self._output_path is None:
            return

        # Write percentile history
        if self._percentile_history:
            pct_file = self._output_path / "percentiles.dat"
            with open(pct_file, "w") as f:
                header = "# Time  " + "  ".join(
                    f"P{p}" for p in self._percentile_levels
                ) + "  IQR  outliers"
                f.write(header + "\n")
                for ps in self._percentile_history:
                    vals = "  ".join(
                        f"{ps.percentiles.get(p, 0.0):.10g}"
                        for p in self._percentile_levels
                    )
                    f.write(
                        f"{ps.time:.6e}  {vals}  "
                        f"{ps.iqr:.10g}  {ps.outlier_count}\n"
                    )

        # Write latest histogram
        hist = self.get_latest_histogram()
        if hist is not None and hist.bin_edges is not None and hist.counts is not None:
            hist_file = self._output_path / "histogram.dat"
            with open(hist_file, "w") as f:
                f.write(f"# Histogram of '{hist.field_name}' at t={hist.time:.6e}\n")
                f.write("# bin_left  bin_right  count\n")
                for i in range(hist.counts.numel()):
                    left = hist.bin_edges[i].item()
                    right = hist.bin_edges[i + 1].item() if i + 1 < hist.bin_edges.numel() else left + hist.bin_width
                    f.write(f"{left:.10g}  {right:.10g}  {hist.counts[i].item()}\n")

        logger.info("Wrote FieldMinMaxEnhanced3 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("fieldMinMaxEnhanced3", FieldMinMaxEnhanced3)
