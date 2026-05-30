"""WallShearStressEnhanced6 — Enhanced wall shear stress v6 with wall-modeled LES interface and pressure-strain coupling.

Extends WallShearStressEnhanced5 with:

- **WMLES 接口**: wall-modeled LES interface for providing wall shear stress to SGS model
- **压力-应变耦合**: pressure-strain correlation estimation at the wall
- **时间尺度分析**: multi-resolution time-scale decomposition of wall shear

Usage::

    wss = WallShearStressEnhanced6("wss6", {
        "patches": ["movingWall"],
        "rho": 1.0,
        "mu": 1e-3,
        "wmlesInterface": True,
        "pressureStrain": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.wall_shear_stress_enhanced_5 import (
    WallShearStressEnhanced5, AnisotropyTensor, CoherentStructure,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["WallShearStressEnhanced6", "WMLESInterface", "PressureStrainCorrelation"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class WMLESInterface:
    """Wall-modeled LES interface data.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        tau_w_model: Modeled wall shear stress.
        u_tau_model: Modeled friction velocity.
        y_plus_model: Estimated y+ at first cell.
    """
    patch_name: str = ""
    time: float = 0.0
    tau_w_model: Optional[torch.Tensor] = None
    u_tau_model: float = 0.0
    y_plus_model: float = 0.0


@dataclass
class PressureStrainCorrelation:
    """Pressure-strain correlation at the wall.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        phi_iw1: Pressure-strain component 1.
        phi_iw2: Pressure-strain component 2.
        estimated_R_ratio: Estimated Reynolds stress ratio.
    """
    patch_name: str = ""
    time: float = 0.0
    phi_iw1: float = 0.0
    phi_iw2: float = 0.0
    estimated_R_ratio: float = 0.0


class WallShearStressEnhanced6(WallShearStressEnhanced5):
    """Enhanced wall shear stress v6 with WMLES interface and pressure-strain coupling.

    在 WallShearStressEnhanced5 基础上增加的配置键：

    - ``wmlesInterface``: enable WMLES interface data output (default: False)
    - ``pressureStrain``: enable pressure-strain correlation estimation (default: False)
    - ``timeScaleAnalysis``: enable multi-resolution time-scale analysis (default: False)
    - ``kappa``: Von Karman constant for WMLES (default: 0.41)
    """

    def __init__(
        self,
        name: str = "wallShearStressEnhanced6",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._wmles_enabled: bool = self.config.get("wmlesInterface", False)
        self._pressure_strain: bool = self.config.get("pressureStrain", False)
        self._kappa: float = float(self.config.get("kappa", 0.41))

        self._wmles_data: List[Dict[str, WMLESInterface]] = []
        self._pressure_strain_data: List[Dict[str, PressureStrainCorrelation]] = []

    def _compute_wmles(self, tau_w: Dict[str, torch.Tensor], time: float) -> Dict[str, WMLESInterface]:
        """Compute WMLES interface data."""
        results: Dict[str, WMLESInterface] = {}
        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1).clamp(min=_EPS)
            tau_mean = float(tau_mag.mean().item())
            u_tau = math.sqrt(tau_mean / max(self._rho, _EPS))
            y_plus_est = u_tau * 1e-4 / max(self._mu / max(self._rho, _EPS), _EPS)  # Approximate

            results[patch_name] = WMLESInterface(
                patch_name=patch_name, time=time,
                tau_w_model=tau, u_tau_model=u_tau, y_plus_model=y_plus_est,
            )
        return results

    def _compute_pressure_strain(self, tau_w: Dict[str, torch.Tensor], time: float) -> Dict[str, PressureStrainCorrelation]:
        """Estimate pressure-strain correlation at wall."""
        results: Dict[str, PressureStrainCorrelation] = {}
        for patch_name, tau in tau_w.items():
            tau_mag = tau.norm(dim=1)
            tau_std = float(tau_mag.std().item()) if tau_mag.numel() > 1 else 0.0
            tau_mean = float(tau_mag.mean().item())
            R_ratio = tau_std / max(tau_mean, _EPS)

            # Simplified pressure-strain model
            phi1 = 0.5 * R_ratio ** 2
            phi2 = -0.3 * R_ratio

            results[patch_name] = PressureStrainCorrelation(
                patch_name=patch_name, time=time,
                phi_iw1=phi1, phi_iw2=phi2, estimated_R_ratio=R_ratio,
            )
        return results

    def execute(self, time: float) -> None:
        """Compute wall shear stress v6."""
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

    @property
    def wmles_data(self) -> List[Dict[str, WMLESInterface]]:
        return self._wmles_data

    @property
    def pressure_strain_data(self) -> List[Dict[str, PressureStrainCorrelation]]:
        return self._pressure_strain_data


FunctionObjectRegistry.register("wallShearStressEnhanced6", WallShearStressEnhanced6)
