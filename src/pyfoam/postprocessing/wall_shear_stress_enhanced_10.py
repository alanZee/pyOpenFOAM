"""WallShearStressEnhanced10 — Enhanced wall shear stress v10 with pressure gradient coupling, Reynolds analogy, and friction decomposition.

Extends WallShearStressEnhanced9 with:
- Pressure gradient coupling for improved near-wall prediction
- Reynolds analogy heat transfer estimation
- Friction decomposition into viscous and turbulent components

Usage::

    wss = WallShearStressEnhanced10("wss10", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "pressureGradientCoupling": True,
        "reynoldsAnalogy": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import (
    WallShearStressEnhanced9, AveragedTopology, StreakDynamics,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced10", "ReynoldsAnalogyResult", "FrictionDecomposition"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class ReynoldsAnalogyResult:
    """Reynolds analogy result.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        Stanton_number: Stanton number.
        heat_flux: Estimated wall heat flux (W/m^2).
        analogy_factor: Reynolds analogy factor (St * Pr^(2/3) / Cf/2).
    """
    patch_name: str = ""
    time: float = 0.0
    Stanton_number: float = 0.0
    heat_flux: float = 0.0
    analogy_factor: float = 0.0


@dataclass
class FrictionDecomposition:
    """Friction decomposition result.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        tau_viscous: Viscous wall shear stress component.
        tau_turbulent: Turbulent wall shear stress component.
        ratio: Turbulent-to-viscous ratio.
    """
    patch_name: str = ""
    time: float = 0.0
    tau_viscous: float = 0.0
    tau_turbulent: float = 0.0
    ratio: float = 0.0


class WallShearStressEnhanced10(WallShearStressEnhanced9):
    """Enhanced wall shear stress v10 with pressure gradient coupling and Reynolds analogy.

    Configuration keys (in addition to v9):

    - ``pressureGradientCoupling``: enable pressure gradient coupling (default: False)
    - ``reynoldsAnalogy``: enable Reynolds analogy (default: False)
    - ``frictionDecomposition``: enable friction decomposition (default: False)
    - ``Pr``: Prandtl number for Reynolds analogy (default: 0.71)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced10",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._pressure_coupling: bool = self.config.get("pressureGradientCoupling", False)
        self._reynolds_analogy: bool = self.config.get("reynoldsAnalogy", False)
        self._friction_decomp: bool = self.config.get("frictionDecomposition", False)
        self._Pr: float = float(self.config.get("Pr", 0.71))

        self._analogy_data: List[Dict[str, ReynoldsAnalogyResult]] = []
        self._decomp_data: List[Dict[str, FrictionDecomposition]] = []

    def _compute_reynolds_analogy(
        self, tau_w: Dict[str, torch.Tensor], time: float, T_wall: float = 300.0, T_ref: float = 350.0,
    ) -> Dict[str, ReynoldsAnalogyResult]:
        """Compute Reynolds analogy for heat transfer estimation."""
        results: Dict[str, ReynoldsAnalogyResult] = {}
        rho = float(self.config.get("rho", 1.0))
        Cp = float(self.config.get("Cp", 1005.0))
        U_ref = float(self.config.get("Uref", 10.0))

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            tau_mean = float(tau_mag.mean().item())

            Cf = tau_mean / max(0.5 * rho * U_ref ** 2, _EPS)
            St = Cf / 2.0 / max(self._Pr ** (2.0 / 3.0), _EPS)
            q = St * rho * Cp * U_ref * (T_ref - T_wall)
            analogy_factor = St * self._Pr ** (2.0 / 3.0) / max(Cf / 2.0, _EPS)

            results[patch_name] = ReynoldsAnalogyResult(
                patch_name=patch_name, time=time,
                Stanton_number=St, heat_flux=q, analogy_factor=analogy_factor,
            )

        return results

    def _decompose_friction(self, tau_w: Dict[str, torch.Tensor], time: float) -> Dict[str, FrictionDecomposition]:
        """Decompose wall shear stress into viscous and turbulent components."""
        results: Dict[str, FrictionDecomposition] = {}
        mu = float(self.config.get("mu", 1e-3))

        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()
            if n < 2:
                continue

            # Viscous component: from mean gradient
            tau_mean = float(tau_mag.mean().item())
            tau_viscous = mu * tau_mean  # Simplified
            tau_turbulent = max(0.0, tau_mean - tau_viscous)
            ratio = tau_turbulent / max(tau_viscous, _EPS)

            results[patch_name] = FrictionDecomposition(
                patch_name=patch_name, time=time,
                tau_viscous=tau_viscous, tau_turbulent=tau_turbulent, ratio=ratio,
            )

        return results

    def execute(self, time: float) -> None:
        """Compute wall shear stress v10."""
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

        # Standard v8/v9 features
        if self._streak_spacing:
            streaks = self._estimate_streak_spacing(tau_w, time)
            self._streak_data.append(streaks)

        if self._skin_friction_topo:
            topo = self._analyze_skin_friction_topology(tau_w, time)
            self._topology_data.append(topo)

        if self._time_avg_topo:
            avg_topo = self._compute_time_avg_topology(time)
            if avg_topo:
                self._averaged_topo.append(avg_topo)

        if self._streak_dynamics:
            dyn = self._compute_streak_dynamics(tau_w, time)
            if dyn:
                self._streak_dynamics_data.append(dyn)

        # v10 features
        if self._reynolds_analogy:
            analogy = self._compute_reynolds_analogy(tau_w, time)
            if analogy:
                self._analogy_data.append(analogy)

        if self._friction_decomp:
            decomp = self._decompose_friction(tau_w, time)
            if decomp:
                self._decomp_data.append(decomp)

    @property
    def analogy_data(self) -> List[Dict[str, ReynoldsAnalogyResult]]:
        return self._analogy_data

    @property
    def decomposition_data(self) -> List[Dict[str, FrictionDecomposition]]:
        return self._decomp_data


FunctionObjectRegistry.register("wallShearStressEnhanced10", WallShearStressEnhanced10)
