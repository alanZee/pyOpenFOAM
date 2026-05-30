"""
ForcesEnhanced4 — Enhanced forces v4 with aeroacoustic sources and unsteady analysis.

在 Enhanced v3 基础上增加：

- **非定常力统计**：力的脉动均方根值和峰峰值
- **气动声学源积分**：Curle 声类比源项
- **压力/粘性力分解的稳定性指标**：力分解的数值稳定性监测
- **力的时间导数**：dF/dt 追踪

Usage::

    forces = ForcesEnhanced4("forces4", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "liftDir": [0.0, 1.0, 0.0],
        "dragDir": [1.0, 0.0, 0.0],
        "computeAeroacousticSources": True,
        "computeUnsteadyStats": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_3 import (
    ForcesEnhanced3,
    MomentCoefficients,
    ForceSpectrum,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced4", "UnsteadyForceStats", "AeroacousticSource"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class UnsteadyForceStats:
    """Unsteady force statistics.

    Attributes:
        time: Simulation time.
        drag_rms: RMS of drag force.
        lift_rms: RMS of lift force.
        drag_pp: Peak-to-peak drag.
        lift_pp: Peak-to-peak lift.
        d_drag_dt: Time derivative of drag.
        d_lift_dt: Time derivative of lift.
        n_samples: Number of samples in window.
    """

    time: float = 0.0
    drag_rms: float = 0.0
    lift_rms: float = 0.0
    drag_pp: float = 0.0
    lift_pp: float = 0.0
    d_drag_dt: float = 0.0
    d_lift_dt: float = 0.0
    n_samples: int = 0


@dataclass
class AeroacousticSource:
    """Curle aeroacoustic source term.

    Based on Curle's analogy:
        p'(x, t) = 1/(4*pi*a0) * d/dt integral(S, pn * cos(theta) / r) dS

    Attributes:
        time: Simulation time.
        source_strength: Integrated source strength (Pa*m^2).
        source_power: Acoustic source power (W).
        directivity_factor: Directivity factor.
        reference_distance: Reference distance for SPL (m).
        spl_estimate: Estimated SPL at reference distance (dB).
    """

    time: float = 0.0
    source_strength: float = 0.0
    source_power: float = 0.0
    directivity_factor: float = 0.5
    reference_distance: float = 1.0
    spl_estimate: float = 0.0


class ForcesEnhanced4(ForcesEnhanced3):
    """Enhanced forces v4 with aeroacoustic sources and unsteady analysis.

    在 ForcesEnhanced3 基础上增加的配置键：

    - ``computeUnsteadyStats``: compute unsteady force statistics (default: True)
    - ``computeAeroacousticSources``: compute Curle acoustic sources (default: False)
    - ``unsteadyWindowSize``: window size for RMS computation (default: 100)
    - ``aeroacousticRefDistance``: reference distance for SPL (default: 1.0)
    - ``speedOfSound``: speed of sound for acoustic analogy (default: 343.0)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced4",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_unsteady: bool = self.config.get("computeUnsteadyStats", True)
        self._compute_aeroacoustic: bool = self.config.get(
            "computeAeroacousticSources", False,
        )
        self._unsteady_window: int = max(10, int(self.config.get("unsteadyWindowSize", 100)))
        self._aa_ref_dist: float = float(
            self.config.get("aeroacousticRefDistance", 1.0),
        )
        self._c0: float = float(self.config.get("speedOfSound", 343.0))

        # Storage
        self._unsteady_stats: List[UnsteadyForceStats] = []
        self._aeroacoustic_sources: List[AeroacousticSource] = []
        self._force_derivatives: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # 非定常力统计
    # ------------------------------------------------------------------

    def _compute_unsteady_force_stats(self, time: float) -> Optional[UnsteadyForceStats]:
        """Compute unsteady force statistics over a sliding window."""
        if not self._projected_forces or len(self._projected_forces) < 2:
            return None

        window = self._projected_forces[-self._unsteady_window:]
        n = len(window)

        drag_vals = torch.tensor([pf.drag for pf in window], dtype=torch.float64)
        lift_vals = torch.tensor([pf.lift for pf in window], dtype=torch.float64)

        # RMS
        drag_rms = float(drag_vals.std().item()) if n > 1 else 0.0
        lift_rms = float(lift_vals.std().item()) if n > 1 else 0.0

        # Peak-to-peak
        drag_pp = float((drag_vals.max() - drag_vals.min()).item())
        lift_pp = float((lift_vals.max() - lift_vals.min()).item())

        # Time derivatives (last two steps)
        d_drag_dt = 0.0
        d_lift_dt = 0.0
        if len(self._projected_forces) >= 2:
            pf_curr = self._projected_forces[-1]
            pf_prev = self._projected_forces[-2]
            dt = pf_curr.time - pf_prev.time
            if dt > _EPS:
                d_drag_dt = (pf_curr.drag - pf_prev.drag) / dt
                d_lift_dt = (pf_curr.lift - pf_prev.lift) / dt

        return UnsteadyForceStats(
            time=time,
            drag_rms=drag_rms,
            lift_rms=lift_rms,
            drag_pp=drag_pp,
            lift_pp=lift_pp,
            d_drag_dt=d_drag_dt,
            d_lift_dt=d_lift_dt,
            n_samples=n,
        )

    # ------------------------------------------------------------------
    # 气动声学源
    # ------------------------------------------------------------------

    def _compute_aeroacoustic_source(self, time: float) -> Optional[AeroacousticSource]:
        """Compute Curle aeroacoustic source term.

        S = d/dt (integral p' * n * dS)

        SPL = 10*log10(S^2 / (4*pi*rho0*c0*r_ref)^2 / p_ref^2)
        """
        if not self._projected_forces or len(self._projected_forces) < 2:
            return None

        rho0 = self.config.get("rhoInf", 1.225)
        p_ref = 2e-5  # Reference pressure (Pa)

        # Source strength: rate of change of pressure force
        pf_curr = self._projected_forces[-1]
        pf_prev = self._projected_forces[-2]
        dt = pf_curr.time - pf_prev.time

        if dt < _EPS:
            return None

        # Drag and lift source terms
        dF_drag = (pf_curr.drag - pf_prev.drag) / dt
        dF_lift = (pf_curr.lift - pf_prev.lift) / dt

        source_strength = math.sqrt(dF_drag ** 2 + dF_lift ** 2)

        # Acoustic power (Curle)
        r = max(self._aa_ref_dist, _EPS)
        denom = 4.0 * math.pi * rho0 * self._c0 * r
        source_power = source_strength ** 2 / max(denom ** 2, _EPS)

        # SPL estimate
        if source_strength > _EPS and p_ref > _EPS:
            spl = 10.0 * math.log10(
                source_strength ** 2 / max(denom ** 2 * p_ref ** 2, _EPS),
            )
        else:
            spl = 0.0

        # Directivity (simplified: 0.5 for dipole)
        directivity = 0.5

        return AeroacousticSource(
            time=time,
            source_strength=source_strength,
            source_power=source_power,
            directivity_factor=directivity,
            reference_distance=self._aa_ref_dist,
            spl_estimate=spl,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute forces v4 at current time step."""
        # Run parent v3 execute
        super().execute(time)

        if not self._enabled:
            return

        if not self._force_total:
            return

        # Unsteady force statistics
        if self._compute_unsteady:
            stats = self._compute_unsteady_force_stats(time)
            if stats is not None:
                self._unsteady_stats.append(stats)

        # Aeroacoustic sources
        if self._compute_aeroacoustic:
            source = self._compute_aeroacoustic_source(time)
            if source is not None:
                self._aeroacoustic_sources.append(source)

                self._log.info(
                    "t=%g  SPL_est=%.2f dB  source_power=%.6e W",
                    time, source.spl_estimate, source.power if hasattr(source, 'power') else source.source_power,
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unsteady_stats(self) -> List[UnsteadyForceStats]:
        """Unsteady force statistics history."""
        return self._unsteady_stats

    @property
    def aeroacoustic_sources(self) -> List[AeroacousticSource]:
        """Aeroacoustic source history."""
        return self._aeroacoustic_sources

    def get_latest_unsteady(self) -> Optional[UnsteadyForceStats]:
        """Get latest unsteady statistics."""
        if not self._unsteady_stats:
            return None
        return self._unsteady_stats[-1]

    def get_latest_aeroacoustic(self) -> Optional[AeroacousticSource]:
        """Get latest aeroacoustic source."""
        if not self._aeroacoustic_sources:
            return None
        return self._aeroacoustic_sources[-1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write forces v4 data."""
        super().write()

        if self._output_path is None:
            return

        # Write unsteady stats
        if self._unsteady_stats:
            us_file = self._output_path / "unsteadyForceStats.dat"
            with open(us_file, "w") as f:
                f.write(
                    "# Time  drag_rms  lift_rms  drag_pp  lift_pp  "
                    "d_drag_dt  d_lift_dt  n_samples\n"
                )
                for us in self._unsteady_stats:
                    f.write(
                        f"{us.time:.6e}  "
                        f"{us.drag_rms:.6e}  {us.lift_rms:.6e}  "
                        f"{us.drag_pp:.6e}  {us.lift_pp:.6e}  "
                        f"{us.d_drag_dt:.6e}  {us.d_lift_dt:.6e}  "
                        f"{us.n_samples}\n"
                    )

        # Write aeroacoustic sources
        if self._aeroacoustic_sources:
            aa_file = self._output_path / "aeroacousticSources.dat"
            with open(aa_file, "w") as f:
                f.write(
                    "# Time  source_strength  source_power  "
                    "directivity  ref_distance  SPL_dB\n"
                )
                for aa in self._aeroacoustic_sources:
                    f.write(
                        f"{aa.time:.6e}  "
                        f"{aa.source_strength:.6e}  {aa.source_power:.6e}  "
                        f"{aa.directivity_factor:.4f}  "
                        f"{aa.reference_distance:.6e}  "
                        f"{aa.spl_estimate:.2f}\n"
                    )

        logger.info("Wrote ForcesEnhanced4 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("forcesEnhanced4", ForcesEnhanced4)
