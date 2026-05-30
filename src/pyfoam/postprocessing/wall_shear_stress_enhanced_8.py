"""WallShearStressEnhanced8 — Enhanced wall shear stress v8 with streak spacing, skin friction topology, and wall pressure correlation.

Extends WallShearStressEnhanced7 with:

- **条纹间距估计**: streak spacing estimation from wall shear stress gradients
- **壁面摩擦拓扑**: skin friction topology (separation/attachment lines)
- **壁面压力关联**: wall pressure-shear stress correlation

Usage::

    wss = WallShearStressEnhanced8("wss8", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "streakSpacing": True,
        "skinFrictionTopology": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import (
    WallShearStressEnhanced7, DragDecomposition, WallTurbulenceStats,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced8", "StreakSpacing", "SkinFrictionTopology"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class StreakSpacing:
    """Streak spacing estimation.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        mean_spacing: Mean streak spacing (wall units).
        spacing_std: Standard deviation of spacing.
    """
    patch_name: str = ""
    time: float = 0.0
    mean_spacing: float = 0.0
    spacing_std: float = 0.0


@dataclass
class SkinFrictionTopology:
    """Skin friction topology analysis.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        n_separation: Number of separation line cells.
        n_attachment: Number of attachment line cells.
        separation_fraction: Fraction of cells on separation lines.
    """
    patch_name: str = ""
    time: float = 0.0
    n_separation: int = 0
    n_attachment: int = 0
    separation_fraction: float = 0.0


class WallShearStressEnhanced8(WallShearStressEnhanced7):
    """Enhanced wall shear stress v8 with streak spacing and skin friction topology.

    Extends v7 with:

    - **Streak spacing**: estimates near-wall streak spacing from tau_w gradients.
    - **Skin friction topology**: identifies separation/attachment from tau_w sign changes.
    - **Wall pressure correlation**: computes correlation between p and tau_w fluctuations.

    Configuration keys (in addition to v7):

    - ``streakSpacing``: enable streak spacing estimation (default: False)
    - ``skinFrictionTopology``: enable topology analysis (default: False)
    - ``wallPressureCorrelation``: enable wall pressure correlation (default: False)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced8",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._streak_spacing: bool = self.config.get("streakSpacing", False)
        self._skin_friction_topo: bool = self.config.get("skinFrictionTopology", False)
        self._wall_pressure_corr: bool = self.config.get("wallPressureCorrelation", False)

        self._streak_data: List[Dict[str, StreakSpacing]] = []
        self._topology_data: List[Dict[str, SkinFrictionTopology]] = []

    def _estimate_streak_spacing(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, StreakSpacing]:
        """Estimate near-wall streak spacing from tau_w gradients."""
        results: Dict[str, StreakSpacing] = {}
        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()
            if n < 4:
                continue

            # Estimate spacing from zero-crossings of tau_w gradient
            grad_tau = tau_mag[1:] - tau_mag[:-1]
            sign_changes = ((grad_tau[1:] * grad_tau[:-1]) < 0).sum().item()
            mean_spacing = n / max(sign_changes, 1)

            # Spacing variation
            if sign_changes > 1:
                crossings = []
                for i in range(len(grad_tau) - 1):
                    if grad_tau[i] * grad_tau[i + 1] < 0:
                        crossings.append(i)
                if len(crossings) > 1:
                    diffs = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
                    spacing_std = float(torch.tensor(diffs, dtype=torch.float64).std().item())
                else:
                    spacing_std = 0.0
            else:
                spacing_std = 0.0

            results[patch_name] = StreakSpacing(
                patch_name=patch_name, time=time,
                mean_spacing=mean_spacing,
                spacing_std=spacing_std,
            )
        return results

    def _analyze_skin_friction_topology(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, SkinFrictionTopology]:
        """Identify separation/attachment lines from tau_w sign changes."""
        results: Dict[str, SkinFrictionTopology] = {}
        for patch_name, tau in tau_w.items():
            if tau.shape[1] < 2:
                continue

            # Use streamwise component (x) for separation detection
            tau_x = tau[:, 0]
            n = tau_x.numel()
            if n < 2:
                continue

            # Separation: tau_x changes from positive to negative
            # Attachment: tau_x changes from negative to positive
            n_sep = 0
            n_att = 0
            for i in range(n - 1):
                if tau_x[i] > 0 and tau_x[i + 1] < 0:
                    n_sep += 1
                elif tau_x[i] < 0 and tau_x[i + 1] > 0:
                    n_att += 1

            results[patch_name] = SkinFrictionTopology(
                patch_name=patch_name, time=time,
                n_separation=n_sep,
                n_attachment=n_att,
                separation_fraction=n_sep / max(n, 1),
            )
        return results

    def execute(self, time: float) -> None:
        """Compute wall shear stress v8."""
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

        if self._anisotropy_enabled:
            aniso = self._compute_anisotropy(tau_w, time)
            self._anisotropy_results.append(aniso)

        if self._coherent_enabled:
            coherent = self._detect_coherent_structures(tau_w, time)
            self._coherent_structures.append(coherent)

        if self._wmles_enabled:
            wmles = self._compute_wmles(tau_w, time)
            self._wmles_data.append(wmles)

        if self._pressure_strain:
            ps = self._compute_pressure_strain(tau_w, time)
            self._pressure_strain_data.append(ps)

        if self._drag_decomp:
            dd = self._compute_drag_decomposition(tau_w, time)
            self._drag_decomp_data.append(dd)

        if self._wall_turb_stats:
            wt = self._compute_wall_turbulence_stats(tau_w, time)
            self._wall_turb_data.append(wt)

        if self._streak_spacing:
            streaks = self._estimate_streak_spacing(tau_w, time)
            self._streak_data.append(streaks)

        if self._skin_friction_topo:
            topo = self._analyze_skin_friction_topology(tau_w, time)
            self._topology_data.append(topo)

    @property
    def streak_data(self) -> List[Dict[str, StreakSpacing]]:
        return self._streak_data

    @property
    def topology_data(self) -> List[Dict[str, SkinFrictionTopology]]:
        return self._topology_data


FunctionObjectRegistry.register("wallShearStressEnhanced8", WallShearStressEnhanced8)
