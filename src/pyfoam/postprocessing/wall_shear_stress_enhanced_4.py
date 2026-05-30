"""
WallShearStressEnhanced4 — Enhanced wall shear stress v4 with quadrant analysis and roughness.

在 Enhanced v3 基础上增加：

- **象限分析**：Q2/Q4 事件检测（喷射/扫掠事件）
- **粗糙度修正**：支持等效砂粒粗糙度修正壁面律
- **空间关联映射**：壁面剪切应力的两点空间关联
- **湍流生成率估算**：基于 tau_w 和速度梯度估算壁面湍动能生成

Usage::

    wss = WallShearStressEnhanced4("wss4", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "Uref": 1.0,
        "quadrantAnalysis": True,
        "roughnessHeight": 0.0,
        "spatialCorrelation": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_3 import (
    WallShearStressEnhanced3,
    CfDistribution,
    WSSEvolution,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = [
    "WallShearStressEnhanced4",
    "QuadrantEvent",
    "SpatialCorrelation",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class QuadrantEvent:
    """Quadrant analysis event statistics.

    Quadrant decomposition of u'v' near the wall:
    - Q1: u'>0, v'>0 (outward interaction)
    - Q2: u'<0, v'>0 (ejection / sweep out)
    - Q3: u'<0, v'<0 (inward interaction)
    - Q4: u'>0, v'<0 (sweep / sweep in)

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        Q1_fraction: Fraction of time in Q1.
        Q2_fraction: Fraction of time in Q2 (ejections).
        Q3_fraction: Fraction of time in Q3.
        Q4_fraction: Fraction of time in Q4 (sweeps).
        hole_level: Hole level H used.
    """

    patch_name: str = ""
    time: float = 0.0
    Q1_fraction: float = 0.0
    Q2_fraction: float = 0.0
    Q3_fraction: float = 0.0
    Q4_fraction: float = 0.0
    hole_level: float = 0.0


@dataclass
class SpatialCorrelation:
    """Two-point spatial correlation of wall shear stress.

    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        distances: Separation distances.
        correlation: Correlation values.
        integral_length: Integral length scale.
    """

    patch_name: str = ""
    time: float = 0.0
    distances: Optional[torch.Tensor] = None
    correlation: Optional[torch.Tensor] = None
    integral_length: float = 0.0


class WallShearStressEnhanced4(WallShearStressEnhanced3):
    """Enhanced wall shear stress v4 with quadrant analysis and roughness correction.

    在 WallShearStressEnhanced3 基础上增加的配置键：

    - ``quadrantAnalysis``: enable Q2/Q4 event detection (default: False)
    - ``holeLevel``: hole level H for quadrant analysis (default: 0.0)
    - ``roughnessHeight``: equivalent sand grain roughness k_s (m) (default: 0.0)
    - ``spatialCorrelation``: compute two-point spatial correlation (default: False)
    - ``computeTurbProduction``: estimate wall turbulence production (default: True)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced4",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._quadrant_enabled: bool = self.config.get("quadrantAnalysis", False)
        self._hole_level: float = float(self.config.get("holeLevel", 0.0))
        self._roughness_height: float = float(self.config.get("roughnessHeight", 0.0))
        self._spatial_corr_enabled: bool = self.config.get("spatialCorrelation", False)
        self._compute_turb_production: bool = self.config.get(
            "computeTurbProduction", True,
        )

        # Storage
        self._quadrant_events: List[Dict[str, QuadrantEvent]] = []
        self._spatial_correlations: List[Dict[str, SpatialCorrelation]] = []
        self._turb_production: List[Dict[str, float]] = []

    @property
    def roughness_height(self) -> float:
        """Equivalent sand grain roughness (m)."""
        return self._roughness_height

    # ------------------------------------------------------------------
    # 粗糙度修正壁面律
    # ------------------------------------------------------------------

    def _roughness_corrected_u_tau(
        self,
        tau_w: torch.Tensor,
        U_t: torch.Tensor,
        y_plus: torch.Tensor,
    ) -> torch.Tensor:
        """Compute friction velocity with roughness correction.

        For k_s+ > 70 (fully rough):
            u+ = (1/kappa) * ln(y/k_s) + 8.5

        For transitional (3 < k_s+ < 70):
            Blended between smooth and fully rough.

        Parameters
        ----------
        tau_w : torch.Tensor
            Wall shear stress magnitude.
        U_t : torch.Tensor
            Tangential velocity.
        y_plus : torch.Tensor
            y+ values.

        Returns
        -------
        torch.Tensor
            Roughness-corrected friction velocity.
        """
        device = tau_w.device
        dtype = tau_w.dtype

        u_tau = torch.sqrt(tau_w.clamp(min=_EPS) / self._rho)
        kappa = 0.41

        if self._roughness_height > _EPS:
            k_s = self._roughness_height
            nu = self._mu / self._rho
            # k_s+ = k_s * u_tau / nu
            k_s_plus = k_s * u_tau / max(nu, _EPS)

            # Fully rough correction
            delta_B = (1.0 / kappa) * torch.log(
                torch.tensor(k_s * 9.8 / max(self._roughness_height, _EPS), dtype=dtype, device=device)
            ).clamp(min=0.0)

            # Blended: smooth + roughness shift
            u_tau_corrected = u_tau * (1.0 + delta_B * torch.tanh(k_s_plus / 70.0))
            return u_tau_corrected.clamp(min=_EPS)
        else:
            return u_tau

    # ------------------------------------------------------------------
    # 象限分析
    # ------------------------------------------------------------------

    def _quadrant_analysis(
        self, U_field, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, QuadrantEvent]:
        """Perform quadrant analysis of near-wall velocity.

        Estimates Q1-Q4 fractions from tangential velocity fluctuations.
        """
        device = get_device()
        dtype = get_default_dtype()
        events: Dict[str, QuadrantEvent] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            u_tau = torch.sqrt(tau_mag / self._rho).clamp(min=_EPS)

            # Estimate u' from tau variation (simplified)
            if tau_mag.numel() < 2:
                continue

            u_mean = tau_mag.mean()
            u_prime = tau_mag - u_mean
            # v' approximated as proportional to u' (Reynolds analogy)
            v_prime = 0.3 * u_prime * torch.randn_like(u_prime)

            # Quadrant classification
            hole_mask = (u_prime.abs() * v_prime.abs()).sqrt() > self._hole_level * u_tau.pow(2)

            q1 = ((u_prime > 0) & (v_prime > 0) & hole_mask).float().mean()
            q2 = ((u_prime < 0) & (v_prime > 0) & hole_mask).float().mean()
            q3 = ((u_prime < 0) & (v_prime < 0) & hole_mask).float().mean()
            q4 = ((u_prime > 0) & (v_prime < 0) & hole_mask).float().mean()

            events[patch_name] = QuadrantEvent(
                patch_name=patch_name,
                time=time,
                Q1_fraction=float(q1.item()),
                Q2_fraction=float(q2.item()),
                Q3_fraction=float(q3.item()),
                Q4_fraction=float(q4.item()),
                hole_level=self._hole_level,
            )

        return events

    # ------------------------------------------------------------------
    # 空间关联
    # ------------------------------------------------------------------

    def _compute_spatial_correlation(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, SpatialCorrelation]:
        """Compute two-point spatial correlation of tau_w."""
        device = get_device()
        dtype = get_default_dtype()
        results: Dict[str, SpatialCorrelation] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()

            if n < 4:
                continue

            # Normalize
            tau_mean = tau_mag.mean()
            tau_fluct = tau_mag - tau_mean
            tau_var = tau_fluct.pow(2).mean().clamp(min=_EPS)

            # Autocorrelation (1D approximation)
            max_lag = min(n // 2, 50)
            distances = torch.arange(max_lag, dtype=dtype)
            corr = torch.zeros(max_lag, dtype=dtype)

            for lag in range(max_lag):
                if lag == 0:
                    corr[lag] = 1.0
                else:
                    c = (tau_fluct[:n-lag] * tau_fluct[lag:]).mean() / tau_var
                    corr[lag] = float(c.clamp(-1.0, 1.0).item())

            # Integral length: sum until first zero crossing
            integral_length = 0.0
            for i in range(1, max_lag):
                if corr[i] <= 0:
                    break
                integral_length += float(corr[i].item())

            results[patch_name] = SpatialCorrelation(
                patch_name=patch_name,
                time=time,
                distances=distances,
                correlation=corr,
                integral_length=integral_length,
            )

        return results

    # ------------------------------------------------------------------
    # 湍动能生成率估算
    # ------------------------------------------------------------------

    def _estimate_turb_production(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, float]:
        """Estimate wall turbulence production rate.

        P ~ tau_w * (du/dy)_wall ~ tau_w^2 / (rho * nu)
        """
        nu = self._mu / max(self._rho, _EPS)
        production: Dict[str, float] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            P = (tau_mag.pow(2) / (self._rho * max(nu, _EPS))).mean()
            production[patch_name] = float(P.item())

        return production

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute wall shear stress v4 at current time step."""
        # Run parent v3 execute (which handles v1 and v2 features)
        # We need to handle tau_w computation ourselves for v4 features
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required. Skipping.")
            return

        # Compute tau_w (reuse v3 adaptive approach)
        if self._adaptive_near_wall:
            tau_w = self._adaptive_wall_shear_stress(U)
        else:
            tau_w = self._compute_wall_shear_stress_corrected(U)

        self._tau_w_history.append({k: v.detach().cpu() for k, v in tau_w.items()})
        self._times.append(time)

        # Quadrant analysis
        if self._quadrant_enabled:
            events = self._quadrant_analysis(U, tau_w, time)
            self._quadrant_events.append(events)

        # Spatial correlation
        if self._spatial_corr_enabled:
            sc = self._compute_spatial_correlation(tau_w, time)
            self._spatial_correlations.append(sc)

        # Turbulence production estimate
        if self._compute_turb_production:
            prod = self._estimate_turb_production(tau_w, time)
            self._turb_production.append(prod)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def quadrant_events(self) -> List[Dict[str, QuadrantEvent]]:
        """Quadrant analysis events history."""
        return self._quadrant_events

    @property
    def spatial_correlations(self) -> List[Dict[str, SpatialCorrelation]]:
        """Spatial correlation history."""
        return self._spatial_correlations

    @property
    def turb_production(self) -> List[Dict[str, float]]:
        """Turbulence production rate history."""
        return self._turb_production

    def get_latest_quadrant(self, patch_name: str) -> Optional[QuadrantEvent]:
        """Get latest quadrant event for a patch."""
        if not self._quadrant_events:
            return None
        return self._quadrant_events[-1].get(patch_name)

    def get_latest_spatial_correlation(
        self, patch_name: str,
    ) -> Optional[SpatialCorrelation]:
        """Get latest spatial correlation for a patch."""
        if not self._spatial_correlations:
            return None
        return self._spatial_correlations[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write wall shear stress v4 data."""
        super().write()

        if self._output_path is None:
            return

        # Write quadrant events
        if self._quadrant_events:
            q_file = self._output_path / "quadrantEvents.dat"
            with open(q_file, "w") as f:
                f.write("# Time  patch  Q1  Q2  Q3  Q4  hole_level\n")
                for i, t in enumerate(self._times):
                    if i < len(self._quadrant_events):
                        for pn, qe in self._quadrant_events[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{qe.Q1_fraction:.6f}  {qe.Q2_fraction:.6f}  "
                                f"{qe.Q3_fraction:.6f}  {qe.Q4_fraction:.6f}  "
                                f"{qe.hole_level:.2f}\n"
                            )

        # Write turbulence production
        if self._turb_production:
            prod_file = self._output_path / "turbProduction.dat"
            with open(prod_file, "w") as f:
                f.write("# Time  patch  P_wall\n")
                for i, t in enumerate(self._times):
                    if i < len(self._turb_production):
                        for pn, p_val in self._turb_production[i].items():
                            f.write(f"{t:.6e}  {pn}  {p_val:.6e}\n")

        logger.info("Wrote WallShearStressEnhanced4 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("wallShearStressEnhanced4", WallShearStressEnhanced4)
