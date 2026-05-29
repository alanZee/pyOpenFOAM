"""
FieldMinMaxEnhanced2 — Enhanced field min/max v2 with per-region stats and time history.

在 FieldMinMaxEnhanced 基础上增加：

- **Per-region 统计**：按 mesh region 或 cell zone 分别计算 min/max
- **时间历史追踪**：完整的时间序列记录，含收敛监控
- **时间导数追踪**：记录 min/max 值的时间变化率
- **收敛判据**：基于 min/max 变化的收敛监测

Usage::

    fmm = FieldMinMaxEnhanced2("fieldMinMax2", {
        "field": "p",
        "regions": ["region0", "region1"],
        "trackConvergence": True,
        "convergenceTol": 1e-4,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.field_min_max_enhanced import (
    FieldMinMaxEnhanced,
    EnhancedMinMaxResult,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["FieldMinMaxEnhanced2", "RegionMinMaxResult"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class RegionMinMaxResult:
    """Min/max result for a single region/zone.

    Attributes:
        region_name: Region or zone name.
        time: Simulation time.
        min_value: Minimum value in the region.
        max_value: Maximum value in the region.
        min_location: Cell index of minimum (global).
        max_location: Cell index of maximum (global).
        n_cells: Number of cells in the region.
        mean: Region mean value.
        std: Region standard deviation.
    """

    region_name: str = ""
    time: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    min_location: int = -1
    max_location: int = -1
    n_cells: int = 0
    mean: float = 0.0
    std: float = 0.0


@dataclass
class ConvergenceInfo:
    """Convergence tracking for a field quantity.

    Attributes:
        quantity_name: Name of the tracked quantity.
        is_converged: Whether convergence criterion is met.
        residual: Current residual (change between time steps).
        n_steps: Number of time steps tracked.
        tol: Convergence tolerance.
    """

    quantity_name: str = ""
    is_converged: bool = False
    residual: float = float("inf")
    n_steps: int = 0
    tol: float = 1e-4


class FieldMinMaxEnhanced2(FieldMinMaxEnhanced):
    """Enhanced field min/max v2 with per-region stats and time history.

    在 FieldMinMaxEnhanced 基础上增加的配置键：

    - ``regions``: list of region/zone names for per-region analysis
    - ``trackConvergence``: enable convergence monitoring (default: True)
    - ``convergenceTol``: convergence tolerance (default: 1e-4)
    - ``trackTimeDerivative``: track time derivative of min/max (default: True)
    """

    def __init__(
        self,
        name: str = "fieldMinMaxEnhanced2",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._regions: List[str] = self.config.get("regions", [])
        self._track_convergence: bool = self.config.get("trackConvergence", True)
        self._convergence_tol: float = float(self.config.get("convergenceTol", 1e-4))
        self._track_time_derivative: bool = self.config.get("trackTimeDerivative", True)

        # Region results
        self._region_results: Dict[str, List[RegionMinMaxResult]] = {}

        # Time derivative tracking
        self._dmin_dt: List[float] = []
        self._dmax_dt: List[float] = []
        self._prev_min: Optional[float] = None
        self._prev_max: Optional[float] = None
        self._prev_time: Optional[float] = None

        # Convergence
        self._convergence = ConvergenceInfo(
            quantity_name=self._field_name,
            tol=self._convergence_tol,
        )

    def execute(self, time: float) -> None:
        """Compute enhanced v2 min/max at the current time step."""
        if not self._enabled:
            return

        field = self._fields.get(self._field_name)
        if field is None:
            logger.warning("Field '%s' not found. Skipping.", self._field_name)
            return

        # Compute base enhanced result
        base_result = self._compute_min_max(field, time)
        enhanced = self._enhance_result(field, base_result, time)

        if self._track_history:
            self._enhanced_results.append(enhanced)
        self._results.append(base_result)

        # Time derivative
        if self._track_time_derivative:
            self._compute_time_derivative(enhanced, time)

        # Per-region analysis
        if self._regions and self._mesh is not None:
            self._compute_per_region(field, time)

        # Convergence check
        if self._track_convergence:
            self._check_convergence(enhanced, time)

        if self._do_log:
            self._log.info(
                "t=%g  field='%s'  min=%.6g  max=%.6g  mean=%.6g  "
                "dmin/dt=%.6g  converged=%s",
                time, self._field_name,
                enhanced.min_value, enhanced.max_value, enhanced.mean,
                self._dmin_dt[-1] if self._dmin_dt else 0.0,
                self._convergence.is_converged,
            )

    def _compute_time_derivative(
        self, result: EnhancedMinMaxResult, time: float,
    ) -> None:
        """Compute time derivative of min and max values."""
        if self._prev_time is not None and self._prev_min is not None:
            dt = time - self._prev_time
            if dt > _EPS:
                dmin = (result.min_value - self._prev_min) / dt
                dmax = (result.max_value - self._prev_max) / dt
            else:
                dmin = 0.0
                dmax = 0.0
            self._dmin_dt.append(dmin)
            self._dmax_dt.append(dmax)
        else:
            self._dmin_dt.append(0.0)
            self._dmax_dt.append(0.0)

        self._prev_min = result.min_value
        self._prev_max = result.max_value
        self._prev_time = time

    def _check_convergence(
        self, result: EnhancedMinMaxResult, time: float,
    ) -> None:
        """Check convergence of min/max values."""
        self._convergence.n_steps += 1

        if len(self._enhanced_results) >= 2:
            prev = self._enhanced_results[-2]
            delta_min = abs(result.min_value - prev.min_value)
            delta_max = abs(result.max_value - prev.max_value)
            self._convergence.residual = max(delta_min, delta_max)
            self._convergence.is_converged = (
                self._convergence.residual < self._convergence_tol
            )

    def _compute_per_region(self, field, time: float) -> None:
        """Compute min/max per region/zone."""
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(field, "internal_field"):
            data = field.internal_field.to(device=device, dtype=dtype)
        elif hasattr(field, "data"):
            data = field.data.to(device=device, dtype=dtype)
        else:
            data = field.to(device=device, dtype=dtype)

        mesh = self._mesh

        # Use cell zones or simple cell ranges for regions
        cell_zones = getattr(mesh, "cell_zones", None)

        for region_name in self._regions:
            if cell_zones and region_name in cell_zones:
                cell_indices = cell_zones[region_name]
            elif hasattr(mesh, "boundary"):
                # Fallback: use boundary patch owner cells
                cell_indices = self._get_region_cells(mesh, region_name)
            else:
                continue

            if not cell_indices:
                continue

            idx_t = torch.tensor(cell_indices, device=device, dtype=torch.long)
            valid = idx_t[idx_t < data.shape[0]]
            if valid.numel() == 0:
                continue

            if data.dim() == 1:
                region_data = data[valid]
            else:
                region_data = data[valid].norm(dim=1)

            min_val = float(region_data.min().item())
            max_val = float(region_data.max().item())
            min_loc = int(valid[region_data.argmin()].item())
            max_loc = int(valid[region_data.argmax()].item())

            result = RegionMinMaxResult(
                region_name=region_name,
                time=time,
                min_value=min_val,
                max_value=max_val,
                min_location=min_loc,
                max_location=max_loc,
                n_cells=len(cell_indices),
                mean=float(region_data.mean().item()),
                std=float(region_data.std().item()),
            )

            if region_name not in self._region_results:
                self._region_results[region_name] = []
            self._region_results[region_name].append(result)

    def _get_region_cells(self, mesh, region_name: str) -> List[int]:
        """Get cell indices for a region by patch owner cells."""
        cells = set()
        if hasattr(mesh, "boundary") and mesh.boundary:
            for p in mesh.boundary:
                if p.get("name") == region_name:
                    start = p["startFace"]
                    n_faces = p["nFaces"]
                    for fi in range(start, start + n_faces):
                        if fi < mesh.owner.shape[0]:
                            cells.add(int(mesh.owner[fi].item()))
        return list(cells)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def region_results(self) -> Dict[str, List[RegionMinMaxResult]]:
        """Per-region min/max results."""
        return self._region_results

    @property
    def dmin_dt(self) -> List[float]:
        """Time derivative of min value."""
        return self._dmin_dt

    @property
    def dmax_dt(self) -> List[float]:
        """Time derivative of max value."""
        return self._dmax_dt

    @property
    def convergence(self) -> ConvergenceInfo:
        """Convergence information."""
        return self._convergence

    def write(self) -> None:
        """Write v2 enhanced min/max data."""
        super().write()

        if self._output_path is None:
            return

        # Write convergence info
        if self._track_convergence and self._convergence.n_steps > 0:
            conv_file = self._output_path / "convergence.dat"
            with open(conv_file, "w") as f:
                f.write(f"# Convergence for field '{self._field_name}'\n")
                f.write(f"# Tolerance: {self._convergence_tol}\n")
                f.write(f"# Residual: {self._convergence.residual:.6e}\n")
                f.write(f"# Converged: {self._convergence.is_converged}\n")
                f.write(f"# Steps: {self._convergence.n_steps}\n")

        # Write per-region results
        if self._region_results:
            for region_name, results in self._region_results.items():
                region_file = self._output_path / f"fieldMinMax2_{region_name}.dat"
                with open(region_file, "w") as f:
                    f.write("# Time  min  minCell  max  maxCell  mean  std  n_cells\n")
                    for r in results:
                        f.write(
                            f"{r.time:.6e}  {r.min_value:.10g}  {r.min_location}  "
                            f"{r.max_value:.10g}  {r.max_location}  "
                            f"{r.mean:.10g}  {r.std:.10g}  {r.n_cells}\n"
                        )

        logger.info("Wrote FieldMinMaxEnhanced2 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("fieldMinMaxEnhanced2", FieldMinMaxEnhanced2)
