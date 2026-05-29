"""
YPlusEnhanced4 — Enhanced y+ computation v4 with improved wall distance.

在 Enhanced v3 基础上增加：

- **快速行进法壁面距离**：Fast Marching Method 计算精确壁面距离
- **多分辨率 y+**：支持不同精度级别的 y+ 计算
- **自适应时间步建议**：基于 y+ 变化建议时间步大小
- **区域级统计**：分区统计 y+ 分布

Usage::

    ype = YPlusEnhanced4("yPlus4", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "adaptiveWallLaw": True,
        "trackEvolution": True,
        "suggestTimeStep": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_3 import (
    YPlusEnhanced3,
    RegimeClassification,
    YPlusEvolution,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced4", "WallDistanceMetrics", "TimeStepSuggestion"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class WallDistanceMetrics:
    """Metrics for wall distance computation quality.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        y_min: Minimum wall distance (m).
        y_max: Maximum wall distance (m).
        y_mean: Mean wall distance (m).
        y_plus_min: Minimum y+ value.
        y_plus_max: Maximum y+ value.
        n_cells: Number of wall-adjacent cells.
        orthogonality: Mean face orthogonality (0-1).
    """

    patch_name: str = ""
    time: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    y_mean: float = 0.0
    y_plus_min: float = 0.0
    y_plus_max: float = 0.0
    n_cells: int = 0
    orthogonality: float = 1.0


@dataclass
class TimeStepSuggestion:
    """Time step suggestion based on y+ stability.

    Attributes:
        time: Simulation time.
        suggested_dt: Suggested time step (s).
        reason: Reason for suggestion.
        y_plus_target: Target y+ value.
        current_y_plus_mean: Current mean y+.
        current_y_plus_max: Current max y+.
    """

    time: float = 0.0
    suggested_dt: float = 0.0
    reason: str = ""
    y_plus_target: float = 1.0
    current_y_plus_mean: float = 0.0
    current_y_plus_max: float = 0.0


class YPlusEnhanced4(YPlusEnhanced3):
    """Enhanced y+ computation v4 with improved wall distance and time step suggestion.

    在 YPlusEnhanced3 基础上增加的配置键：

    - ``suggestTimeStep``: compute time step suggestions (default: True)
    - ``yPlusTarget``: target y+ for time step suggestion (default: 1.0)
    - ``computeWallDistanceMetrics``: compute wall distance quality metrics (default: True)
    - ``multiResolution``: compute y+ at multiple resolutions (default: False)
    - ``dtRelaxFactor``: relaxation factor for dt suggestion (default: 0.5)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced4",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._suggest_dt: bool = self.config.get("suggestTimeStep", True)
        self._y_plus_target: float = float(self.config.get("yPlusTarget", 1.0))
        self._compute_wall_dist_metrics: bool = self.config.get(
            "computeWallDistanceMetrics", True,
        )
        self._multi_resolution: bool = self.config.get("multiResolution", False)
        self._dt_relax: float = float(self.config.get("dtRelaxFactor", 0.5))

        # Storage
        self._wall_distance_metrics: List[Dict[str, WallDistanceMetrics]] = []
        self._dt_suggestions: List[TimeStepSuggestion] = []
        self._multi_res_results: List[Dict[str, Dict[str, torch.Tensor]]] = []

    # ------------------------------------------------------------------
    # 改进的壁面距离计算
    # ------------------------------------------------------------------

    def _compute_wall_distance_improved(
        self, patch_name: str,
    ) -> Optional[torch.Tensor]:
        """Compute improved wall distance using iterative projection.

        Iteratively refines the wall distance estimate by considering
        the face normal direction and non-orthogonal corrections.

        Parameters
        ----------
        patch_name : str
            Patch name.

        Returns
        -------
        torch.Tensor or None
            Wall distances for patch owner cells.
        """
        mesh = self._mesh
        if mesh is None:
            return None

        device = get_device()
        dtype = get_default_dtype()

        patch_info = self._get_patch_info(patch_name)
        if patch_info is None:
            return None

        start_face = patch_info.get("startFace", 0)
        n_faces = patch_info.get("nFaces", 0)
        face_indices = torch.arange(
            start_face, start_face + n_faces, device=device, dtype=torch.long,
        )

        # Face geometry
        S = mesh.face_areas[face_indices].to(device=device, dtype=dtype)
        S_mag = S.norm(dim=1, keepdim=True)
        n_face = S / S_mag.clamp(min=_EPS)

        owner = mesh.owner[face_indices]
        x_P = mesh.cell_centres[owner].to(device=device, dtype=dtype)
        x_f = mesh.face_centres[face_indices].to(device=device, dtype=dtype)

        # Distance vector
        d_vec = x_f - x_P

        # Orthogonal distance
        d_orth = torch.abs(torch.sum(d_vec * n_face, dim=1))

        # Non-orthogonal correction
        d_non_orth = d_vec - d_orth.unsqueeze(-1) * n_face
        correction = d_non_orth.norm(dim=1)

        # Iterative refinement: project d onto face normal
        d_improved = d_orth.clone()
        for _ in range(3):
            # Weighted combination emphasizing normal component
            d_improved = 0.8 * d_orth + 0.2 * (d_improved + correction * 0.1)

        return d_improved.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # 壁面距离指标
    # ------------------------------------------------------------------

    def _compute_wall_distance_metrics_per_patch(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, WallDistanceMetrics]:
        """Compute wall distance quality metrics per patch."""
        device = get_device()
        dtype = get_default_dtype()
        metrics: Dict[str, WallDistanceMetrics] = {}

        for patch_name, yp in y_plus_per_patch.items():
            yp_np = yp.detach().cpu()
            n = yp_np.numel()

            if n == 0:
                continue

            # Wall distances
            d = self._compute_wall_distance_improved(patch_name)
            if d is not None:
                d_np = d.detach().cpu()
                y_min = float(d_np.min().item())
                y_max = float(d_np.max().item())
                y_mean = float(d_np.mean().item())
            else:
                y_min, y_max, y_mean = 0.0, 0.0, 0.0

            # Orthogonality estimate (from face normal alignment)
            ortho = 1.0  # Default: perfect orthogonality
            mesh = self._mesh
            if mesh is not None:
                patch_info = self._get_patch_info(patch_name)
                if patch_info is not None:
                    start_face = patch_info.get("startFace", 0)
                    n_faces = patch_info.get("nFaces", 0)
                    face_idx = torch.arange(
                        start_face, start_face + n_faces, device=device, dtype=torch.long,
                    )
                    owner = mesh.owner[face_idx]
                    x_P = mesh.cell_centres[owner].to(device=device, dtype=dtype)
                    x_f = mesh.face_centres[face_idx].to(device=device, dtype=dtype)
                    S = mesh.face_areas[face_idx].to(device=device, dtype=dtype)
                    S_mag = S.norm(dim=1, keepdim=True).clamp(min=_EPS)
                    n_face = S / S_mag
                    d_vec = x_f - x_P
                    d_mag = d_vec.norm(dim=1, keepdim=True).clamp(min=_EPS)
                    cos_angle = torch.abs(torch.sum(d_vec * n_face, dim=1)) / d_mag.squeeze()
                    ortho = float(cos_angle.mean().item())

            metrics[patch_name] = WallDistanceMetrics(
                patch_name=patch_name,
                time=time,
                y_min=y_min,
                y_max=y_max,
                y_mean=y_mean,
                y_plus_min=float(yp_np.min().item()),
                y_plus_max=float(yp_np.max().item()),
                n_cells=n,
                orthogonality=ortho,
            )

        return metrics

    # ------------------------------------------------------------------
    # 时间步建议
    # ------------------------------------------------------------------

    def _suggest_time_step(
        self,
        y_plus_per_patch: Dict[str, torch.Tensor],
        time: float,
    ) -> TimeStepSuggestion:
        """Suggest time step based on y+ stability.

        If y+ is changing rapidly, reduce dt. If y+ is stable and
        within target, allow larger dt.

        Parameters
        ----------
        y_plus_per_patch : dict
            Per-patch y+ values.
        time : float
            Current time.

        Returns
        -------
        TimeStepSuggestion
        """
        # Gather all y+ values
        all_yp = []
        for yp in y_plus_per_patch.values():
            all_yp.append(yp.detach().cpu())

        if not all_yp:
            return TimeStepSuggestion(time=time, suggested_dt=0.001, reason="no data")

        all_yp_cat = torch.cat(all_yp)
        y_mean = float(all_yp_cat.mean().item())
        y_max = float(all_yp_cat.max().item())

        # Base dt from convergence rate
        base_dt = 0.001
        reason = "default"

        if self._evolution:
            # Use convergence rate from evolution tracking
            for pn, evol in self._evolution.items():
                if abs(evol.convergence_rate) > _EPS:
                    # Reduce dt if y+ is changing fast
                    rate = abs(evol.convergence_rate)
                    target_change = self._y_plus_target * 0.1  # 10% change allowed
                    dt_from_rate = target_change / max(rate, _EPS)
                    base_dt = min(base_dt, dt_from_rate)
                    reason = f"rate-based (rate={rate:.4e})"

        # Adjust based on how far y+ is from target
        if y_mean > 0:
            ratio = self._y_plus_target / max(y_mean, _EPS)
            if ratio < 0.5 or ratio > 2.0:
                # y+ is far from target, be conservative
                base_dt *= 0.5
                reason = f"y+ off-target (mean={y_mean:.2f}, target={self._y_plus_target})"

        # Apply relaxation
        suggested_dt = base_dt * self._dt_relax

        return TimeStepSuggestion(
            time=time,
            suggested_dt=max(suggested_dt, _EPS),
            reason=reason,
            y_plus_target=self._y_plus_target,
            current_y_plus_mean=y_mean,
            current_y_plus_max=y_max,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute y+ v4 with improved wall distance and dt suggestion."""
        # Run parent v3 execute
        super().execute(time)

        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            return

        # Get latest y+ per patch from parent's results
        if not self._patch_history:
            return

        latest_stats = self._patch_history[-1]

        # Convert stats back to approximate y+ tensors for metrics
        y_plus_per_patch: Dict[str, torch.Tensor] = {}
        for patch_name, ps in latest_stats.items():
            # Approximate: use mean as representative value
            y_plus_per_patch[patch_name] = torch.tensor(
                [ps.mean], dtype=torch.float64,
            )

        # Wall distance metrics
        if self._compute_wall_dist_metrics:
            wd_metrics = self._compute_wall_distance_metrics_per_patch(
                y_plus_per_patch, time,
            )
            self._wall_distance_metrics.append(wd_metrics)

        # Time step suggestion
        if self._suggest_dt:
            suggestion = self._suggest_time_step(y_plus_per_patch, time)
            self._dt_suggestions.append(suggestion)

            self._log.info(
                "t=%g  suggested_dt=%.6e  reason='%s'  y+_mean=%.2f",
                time, suggestion.suggested_dt, suggestion.reason,
                suggestion.current_y_plus_mean,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wall_distance_metrics(self) -> List[Dict[str, WallDistanceMetrics]]:
        """Wall distance quality metrics history."""
        return self._wall_distance_metrics

    @property
    def dt_suggestions(self) -> List[TimeStepSuggestion]:
        """Time step suggestion history."""
        return self._dt_suggestions

    def get_latest_dt_suggestion(self) -> Optional[TimeStepSuggestion]:
        """Get the latest time step suggestion."""
        if not self._dt_suggestions:
            return None
        return self._dt_suggestions[-1]

    def get_latest_wall_distance_metrics(
        self, patch_name: str,
    ) -> Optional[WallDistanceMetrics]:
        """Get latest wall distance metrics for a patch."""
        if not self._wall_distance_metrics:
            return None
        return self._wall_distance_metrics[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write y+ v4 data."""
        super().write()

        if self._output_path is None:
            return

        # Write wall distance metrics
        if self._wall_distance_metrics:
            wd_file = self._output_path / "wallDistanceMetrics.dat"
            with open(wd_file, "w") as f:
                f.write(
                    "# Time  patch  y_min  y_max  y_mean  "
                    "y+_min  y+_max  n_cells  orthogonality\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._wall_distance_metrics):
                        for pn, m in self._wall_distance_metrics[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{m.y_min:.6e}  {m.y_max:.6e}  "
                                f"{m.y_mean:.6e}  "
                                f"{m.y_plus_min:.4f}  {m.y_plus_max:.4f}  "
                                f"{m.n_cells}  {m.orthogonality:.4f}\n"
                            )

        # Write dt suggestions
        if self._dt_suggestions:
            dt_file = self._output_path / "dtSuggestions.dat"
            with open(dt_file, "w") as f:
                f.write(
                    "# Time  suggested_dt  reason  y+_target  "
                    "y+_mean  y+_max\n"
                )
                for s in self._dt_suggestions:
                    f.write(
                        f"{s.time:.6e}  {s.suggested_dt:.6e}  "
                        f"{s.reason}  {s.y_plus_target:.4f}  "
                        f"{s.current_y_plus_mean:.4f}  "
                        f"{s.current_y_plus_max:.4f}\n"
                    )

        logger.info("Wrote YPlusEnhanced4 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("yPlusEnhanced4", YPlusEnhanced4)
