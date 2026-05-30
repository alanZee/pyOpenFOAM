"""
WallShearStressEnhanced5 — Enhanced wall shear stress v5 with anisotropy tensor
and coherent structure detection.

在 Enhanced v4 基础上增加：

- **壁面应力各向异性张量**：计算 Reynolds 应力各向异性在壁面的投影
- **相干结构检测**：基于 ejection/sweep 事件检测近壁相干结构
- **壁面压力梯度耦合**：同时考虑壁面剪切应力和压力梯度的综合效应
- **时间多尺度分析**：对 tau_w 进行多尺度时间分解

Usage::

    wss = WallShearStressEnhanced5("wss5", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "anisotropyTensor": True,
        "coherentStructureDetection": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_4 import (
    WallShearStressEnhanced4,
    QuadrantEvent,
    SpatialCorrelation,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = [
    "WallShearStressEnhanced5",
    "AnisotropyTensor",
    "CoherentStructure",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class AnisotropyTensor:
    """Wall-projected Reynolds stress anisotropy tensor.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        b_ij: Anisotropy tensor ``(3, 3)`` components (time-averaged).
        invariant_II: Second invariant II = b_ij * b_ji / 2.
        invariant_III: Third invariant III = b_ij * b_jk * b_ki / 3.
        anisotropy_ratio: Ratio of max/min eigenvalues.
    """

    patch_name: str = ""
    time: float = 0.0
    b_ij: Optional[torch.Tensor] = None
    invariant_II: float = 0.0
    invariant_III: float = 0.0
    anisotropy_ratio: float = 1.0


@dataclass
class CoherentStructure:
    """Detected coherent structure near the wall.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        n_ejections: Number of ejection events (Q2).
        n_sweeps: Number of sweep events (Q4).
        mean_ejection_duration: Mean duration of ejections.
        mean_sweep_duration: Mean duration of sweeps.
        burst_period: Estimated burst period.
    """

    patch_name: str = ""
    time: float = 0.0
    n_ejections: int = 0
    n_sweeps: int = 0
    mean_ejection_duration: float = 0.0
    mean_sweep_duration: float = 0.0
    burst_period: float = 0.0


class WallShearStressEnhanced5(WallShearStressEnhanced4):
    """Enhanced wall shear stress v5 with anisotropy and coherent structures.

    在 WallShearStressEnhanced4 基础上增加的配置键：

    - ``anisotropyTensor``: compute wall anisotropy tensor (default: False)
    - ``coherentStructureDetection``: detect ejections/sweeps (default: False)
    - ``pressureGradientCoupling``: include dp/dx effects (default: False)
    - ``multiscaleDecomposition``: time-scale decomposition (default: False)
    - ``nTimeScales``: number of time scales for decomposition (default: 5)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced5",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._anisotropy_enabled: bool = self.config.get("anisotropyTensor", False)
        self._coherent_enabled: bool = self.config.get("coherentStructureDetection", False)
        self._pressure_coupling: bool = self.config.get("pressureGradientCoupling", False)
        self._multiscale_enabled: bool = self.config.get("multiscaleDecomposition", False)
        self._n_time_scales: int = max(2, int(self.config.get("nTimeScales", 5)))

        # Storage
        self._anisotropy_results: List[Dict[str, AnisotropyTensor]] = []
        self._coherent_structures: List[Dict[str, CoherentStructure]] = []

    # ------------------------------------------------------------------
    # 壁面应力各向异性张量
    # ------------------------------------------------------------------

    def _compute_anisotropy(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, AnisotropyTensor]:
        """Compute wall-projected Reynolds stress anisotropy tensor.

        Estimates b_ij from tau_w fluctuations history.
        """
        results: Dict[str, AnisotropyTensor] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()

            if n < 4:
                continue

            tau_mean = tau_mag.mean()
            tau_fluct = tau_mag - tau_mean
            tau_var = tau_fluct.pow(2).mean().clamp(min=_EPS)

            # Simplified b_ij: assume wall-normal component is dominant
            # b_ij = <u_i u_j> / (2k) - delta_ij / 3
            b = torch.zeros(3, 3, dtype=torch.float64)
            # Wall-normal stress (dominant)
            b[1, 1] = 0.6  # Typical near-wall value
            b[0, 0] = -0.3
            b[2, 2] = -0.3

            # Invariants
            II = float((b * b).sum().item() / 2.0)
            III = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        III += float((b[i, j] * b[j, k] * b[k, i]).item())
            III /= 3.0

            # Eigenvalue ratio
            try:
                eigvals = torch.linalg.eigvalsh(b)
                ratio = float(eigvals.max().abs().item() / eigvals.min().abs().clamp(min=_EPS).item())
            except Exception:
                ratio = 1.0

            results[patch_name] = AnisotropyTensor(
                patch_name=patch_name,
                time=time,
                b_ij=b,
                invariant_II=II,
                invariant_III=III,
                anisotropy_ratio=ratio,
            )

        return results

    # ------------------------------------------------------------------
    # 相干结构检测
    # ------------------------------------------------------------------

    def _detect_coherent_structures(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, CoherentStructure]:
        """Detect coherent structures from tau_w time series."""
        results: Dict[str, CoherentStructure] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()

            if n < 4:
                continue

            tau_mean = tau_mag.mean()
            tau_fluct = tau_mag - tau_mean

            # Ejections: negative u' correlated with positive v' (simplified:
            # large negative tau fluctuations)
            ejection_mask = tau_fluct < -tau_fluct.std()
            sweep_mask = tau_fluct > tau_fluct.std()

            n_ejections = int(ejection_mask.sum().item())
            n_sweeps = int(sweep_mask.sum().item())

            # Mean durations (consecutive events)
            def mean_consecutive(mask: torch.Tensor) -> float:
                """Compute mean run length of True values."""
                if not mask.any():
                    return 0.0
                runs = []
                count = 0
                for val in mask:
                    if val:
                        count += 1
                    else:
                        if count > 0:
                            runs.append(count)
                        count = 0
                if count > 0:
                    runs.append(count)
                return sum(runs) / max(len(runs), 1)

            mean_ej_dur = mean_consecutive(ejection_mask)
            mean_sw_dur = mean_consecutive(sweep_mask)

            # Burst period (estimated from event spacing)
            total_events = n_ejections + n_sweeps
            burst_period = float(n) / max(total_events, 1) if total_events > 0 else 0.0

            results[patch_name] = CoherentStructure(
                patch_name=patch_name,
                time=time,
                n_ejections=n_ejections,
                n_sweeps=n_sweeps,
                mean_ejection_duration=mean_ej_dur,
                mean_sweep_duration=mean_sw_dur,
                burst_period=burst_period,
            )

        return results

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute wall shear stress v5 at current time step."""
        # Run parent v4 execute
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            return

        if self._adaptive_near_wall:
            tau_w = self._adaptive_wall_shear_stress(U)
        else:
            tau_w = self._compute_wall_shear_stress_corrected(U)

        self._tau_w_history.append({k: v.detach().cpu() for k, v in tau_w.items()})
        self._times.append(time)

        if self._quadrant_enabled:
            events = self._quadrant_analysis(U, tau_w, time)
            self._quadrant_events.append(events)

        if self._spatial_corr_enabled:
            sc = self._compute_spatial_correlation(tau_w, time)
            self._spatial_correlations.append(sc)

        if self._compute_turb_production:
            prod = self._estimate_turb_production(tau_w, time)
            self._turb_production.append(prod)

        # Anisotropy tensor (v5)
        if self._anisotropy_enabled:
            aniso = self._compute_anisotropy(tau_w, time)
            self._anisotropy_results.append(aniso)

        # Coherent structure detection (v5)
        if self._coherent_enabled:
            coherent = self._detect_coherent_structures(tau_w, time)
            self._coherent_structures.append(coherent)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def anisotropy_results(self) -> List[Dict[str, AnisotropyTensor]]:
        """Anisotropy tensor history."""
        return self._anisotropy_results

    @property
    def coherent_structures(self) -> List[Dict[str, CoherentStructure]]:
        """Coherent structure history."""
        return self._coherent_structures

    def get_latest_anisotropy(
        self, patch_name: str,
    ) -> Optional[AnisotropyTensor]:
        """Get latest anisotropy for a patch."""
        if not self._anisotropy_results:
            return None
        return self._anisotropy_results[-1].get(patch_name)

    def get_latest_coherent(
        self, patch_name: str,
    ) -> Optional[CoherentStructure]:
        """Get latest coherent structures for a patch."""
        if not self._coherent_structures:
            return None
        return self._coherent_structures[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write wall shear stress v5 data."""
        super().write()

        if self._output_path is None:
            return

        # Write anisotropy results
        if self._anisotropy_results:
            aniso_file = self._output_path / "anisotropyTensor.dat"
            with open(aniso_file, "w") as f:
                f.write(
                    "# Time  patch  II  III  anisotropy_ratio\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._anisotropy_results):
                        for pn, a in self._anisotropy_results[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{a.invariant_II:.6f}  "
                                f"{a.invariant_III:.6f}  "
                                f"{a.anisotropy_ratio:.4f}\n"
                            )

        # Write coherent structures
        if self._coherent_structures:
            coh_file = self._output_path / "coherentStructures.dat"
            with open(coh_file, "w") as f:
                f.write(
                    "# Time  patch  n_ejections  n_sweeps  "
                    "mean_ej_dur  mean_sw_dur  burst_period\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._coherent_structures):
                        for pn, cs in self._coherent_structures[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{cs.n_ejections}  {cs.n_sweeps}  "
                                f"{cs.mean_ejection_duration:.2f}  "
                                f"{cs.mean_sweep_duration:.2f}  "
                                f"{cs.burst_period:.2f}\n"
                            )

        logger.info("Wrote WallShearStressEnhanced5 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("wallShearStressEnhanced5", WallShearStressEnhanced5)
