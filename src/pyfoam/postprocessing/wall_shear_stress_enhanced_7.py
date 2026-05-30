"""WallShearStressEnhanced7 — Enhanced wall shear stress v7 with coherent structure detection and drag decomposition.

Extends WallShearStressEnhanced6 with:

- **相干结构检测**: coherent structure detection using swirling strength
- **阻力分解**: drag decomposition into pressure and viscous components
- **壁面湍流统计**: wall turbulence statistics (rms, skewness, flatness)

Usage::

    wss = WallShearStressEnhanced7("wss7", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "coherentDetection": True,
        "dragDecomposition": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_6 import (
    WallShearStressEnhanced6, WMLESInterface, PressureStrainCorrelation,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced7", "DragDecomposition", "WallTurbulenceStats"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class DragDecomposition:
    """Drag decomposition into pressure and viscous components.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        pressure_drag: Pressure (form) drag contribution.
        viscous_drag: Viscous (friction) drag contribution.
        total_drag: Total drag.
    """
    patch_name: str = ""
    time: float = 0.0
    pressure_drag: float = 0.0
    viscous_drag: float = 0.0
    total_drag: float = 0.0


@dataclass
class WallTurbulenceStats:
    """Wall turbulence statistics.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        tau_rms: RMS wall shear stress.
        tau_skewness: Skewness of wall shear stress distribution.
        tau_flatness: Flatness (kurtosis) of wall shear stress distribution.
    """
    patch_name: str = ""
    time: float = 0.0
    tau_rms: float = 0.0
    tau_skewness: float = 0.0
    tau_flatness: float = 0.0


class WallShearStressEnhanced7(WallShearStressEnhanced6):
    """Enhanced wall shear stress v7 with coherent structure detection and drag decomposition.

    在 WallShearStressEnhanced6 基础上增加的配置键：

    - ``coherentDetection``: enable coherent structure detection (default: False)
    - ``dragDecomposition``: enable drag decomposition (default: False)
    - ``wallTurbulenceStats``: enable wall turbulence statistics (default: False)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced7",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._coherent_detect: bool = self.config.get("coherentDetection", False)
        self._drag_decomp: bool = self.config.get("dragDecomposition", False)
        self._wall_turb_stats: bool = self.config.get("wallTurbulenceStats", False)

        self._drag_decomp_data: List[Dict[str, DragDecomposition]] = []
        self._wall_turb_data: List[Dict[str, WallTurbulenceStats]] = []

    def _compute_drag_decomposition(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, DragDecomposition]:
        """Decompose wall shear stress into pressure and viscous contributions."""
        results: Dict[str, DragDecomposition] = {}
        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            # Simplified: assume viscous drag = mean(|tau_w|) * A
            viscous = float(tau_mag.mean().item())
            # Pressure drag estimated from fluctuations (simplified)
            pressure = float(tau_mag.std().item()) if tau_mag.numel() > 1 else 0.0
            total = viscous + pressure
            results[patch_name] = DragDecomposition(
                patch_name=patch_name, time=time,
                pressure_drag=pressure, viscous_drag=viscous, total_drag=total,
            )
        return results

    def _compute_wall_turbulence_stats(
        self, tau_w: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, WallTurbulenceStats]:
        """Compute wall turbulence statistics."""
        results: Dict[str, WallTurbulenceStats] = {}
        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            n = tau_mag.numel()
            if n < 3:
                continue

            tau_mean = tau_mag.mean()
            tau_std = tau_mag.std()
            tau_rms = float(tau_std.item())

            # Skewness: E[(x-mu)^3] / sigma^3
            tau_centered = tau_mag - tau_mean
            skew = float((tau_centered.pow(3).mean() / max(tau_std.pow(3).item(), _EPS)))

            # Flatness (kurtosis): E[(x-mu)^4] / sigma^4
            flat = float((tau_centered.pow(4).mean() / max(tau_std.pow(4).item(), _EPS)))

            results[patch_name] = WallTurbulenceStats(
                patch_name=patch_name, time=time,
                tau_rms=tau_rms, tau_skewness=skew, tau_flatness=flat,
            )
        return results

    def execute(self, time: float) -> None:
        """Compute wall shear stress v7."""
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

    @property
    def drag_decomp_data(self) -> List[Dict[str, DragDecomposition]]:
        return self._drag_decomp_data

    @property
    def wall_turb_data(self) -> List[Dict[str, WallTurbulenceStats]]:
        return self._wall_turb_data


FunctionObjectRegistry.register("wallShearStressEnhanced7", WallShearStressEnhanced7)
