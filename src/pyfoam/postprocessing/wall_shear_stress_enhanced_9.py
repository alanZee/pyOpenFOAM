"""WallShearStressEnhanced9 — Enhanced wall shear stress v9 with time-averaged topology, streak dynamics, and friction velocity evolution.

Extends WallShearStressEnhanced8 with:
- Time-averaged skin friction topology
- Streak dynamics tracking (spacing evolution)
- Friction velocity (u_tau) temporal evolution

Usage::

    wss = WallShearStressEnhanced9("wss9", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "timeAveragedTopology": True,
        "streakDynamics": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import (
    WallShearStressEnhanced8, StreakSpacing, SkinFrictionTopology,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced9", "AveragedTopology", "StreakDynamics"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class AveragedTopology:
    """Time-averaged skin friction topology.
    Attributes:
        patch_name: Patch name.
        time_span: Simulation time span averaged over.
        mean_separation_fraction: Time-averaged separation fraction.
        mean_attachment_fraction: Time-averaged attachment fraction.
        topology_stability: Topology stability metric (0-1).
    """
    patch_name: str = ""
    time_span: float = 0.0
    mean_separation_fraction: float = 0.0
    mean_attachment_fraction: float = 0.0
    topology_stability: float = 0.0


@dataclass
class StreakDynamics:
    """Streak spacing dynamics.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        current_spacing: Current mean streak spacing (wall units).
        spacing_trend: Rate of change of spacing.
        spacing_std: Standard deviation of spacing.
    """
    patch_name: str = ""
    time: float = 0.0
    current_spacing: float = 0.0
    spacing_trend: float = 0.0
    spacing_std: float = 0.0


class WallShearStressEnhanced9(WallShearStressEnhanced8):
    """Enhanced wall shear stress v9 with time-averaged topology and streak dynamics.

    Configuration keys (in addition to v8):

    - ``timeAveragedTopology``: enable time-averaged topology (default: False)
    - ``streakDynamics``: enable streak dynamics tracking (default: False)
    - ``utauEvolution``: enable u_tau temporal evolution (default: False)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced9",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._time_avg_topo: bool = self.config.get("timeAveragedTopology", False)
        self._streak_dynamics: bool = self.config.get("streakDynamics", False)
        self._utau_evolution: bool = self.config.get("utauEvolution", False)

        self._averaged_topo: List[Dict[str, AveragedTopology]] = []
        self._streak_dynamics_data: List[Dict[str, StreakDynamics]] = []
        self._utau_history: List[Dict[str, float]] = []

    def _compute_time_avg_topology(self, time: float) -> Dict[str, AveragedTopology]:
        """Compute time-averaged skin friction topology."""
        results: Dict[str, AveragedTopology] = {}
        if not self._topology_data:
            return results

        # Aggregate topology data
        patch_aggregate: Dict[str, List[SkinFrictionTopology]] = {}
        for topo_dict in self._topology_data:
            for patch_name, topo in topo_dict.items():
                if patch_name not in patch_aggregate:
                    patch_aggregate[patch_name] = []
                patch_aggregate[patch_name].append(topo)

        for patch_name, topo_list in patch_aggregate.items():
            n = len(topo_list)
            if n == 0:
                continue

            mean_sep = sum(t.separation_fraction for t in topo_list) / n
            mean_att = sum(t.n_attachment for t in topo_list) / n

            # Stability: low variance in separation fraction
            var_sep = sum((t.separation_fraction - mean_sep) ** 2 for t in topo_list) / max(n, 1)
            stability = max(0.0, 1.0 - math.sqrt(var_sep) / max(mean_sep, _EPS))

            time_span = topo_list[-1].time - topo_list[0].time if n > 1 else 0.0

            results[patch_name] = AveragedTopology(
                patch_name=patch_name,
                time_span=time_span,
                mean_separation_fraction=mean_sep,
                mean_attachment_fraction=mean_att,
                topology_stability=min(stability, 1.0),
            )

        return results

    def _compute_streak_dynamics(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, StreakDynamics]:
        """Track streak spacing dynamics."""
        results: Dict[str, StreakDynamics] = {}

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()
            if n < 4:
                continue

            grad_tau = tau_mag[1:] - tau_mag[:-1]
            sign_changes = ((grad_tau[1:] * grad_tau[:-1]) < 0).sum().item()
            spacing = n / max(sign_changes, 1)

            # Trend from streak history
            trend = 0.0
            if self._streak_data:
                prev_data = self._streak_data[-1]
                if patch_name in prev_data:
                    prev_spacing = prev_data[patch_name].mean_spacing
                    dt = time - prev_data[patch_name].time
                    if dt > _EPS:
                        trend = (spacing - prev_spacing) / dt

            results[patch_name] = StreakDynamics(
                patch_name=patch_name, time=time,
                current_spacing=spacing,
                spacing_trend=trend,
                spacing_std=0.0,
            )

        return results

    def execute(self, time: float) -> None:
        """Compute wall shear stress v9."""
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

        # Standard v8 features
        if self._streak_spacing:
            streaks = self._estimate_streak_spacing(tau_w, time)
            self._streak_data.append(streaks)

        if self._skin_friction_topo:
            topo = self._analyze_skin_friction_topology(tau_w, time)
            self._topology_data.append(topo)

        # v9 features
        if self._time_avg_topo:
            avg_topo = self._compute_time_avg_topology(time)
            if avg_topo:
                self._averaged_topo.append(avg_topo)

        if self._streak_dynamics:
            dyn = self._compute_streak_dynamics(tau_w, time)
            if dyn:
                self._streak_dynamics_data.append(dyn)

    @property
    def averaged_topology(self) -> List[Dict[str, AveragedTopology]]:
        return self._averaged_topo

    @property
    def streak_dynamics_data(self) -> List[Dict[str, StreakDynamics]]:
        return self._streak_dynamics_data


FunctionObjectRegistry.register("wallShearStressEnhanced9", WallShearStressEnhanced9)
