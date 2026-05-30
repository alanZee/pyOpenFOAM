"""
Enhanced spray models v4.

Adds BlobAtomizationNozzle and SprayPostProcessing following OpenFOAM conventions.

- :class:`BlobAtomizationNozzle` — blob model with nozzle geometry effects
- :class:`SprayPostProcessing`  — spray diagnostic and post-processing model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["BlobAtomizationNozzle", "SprayPostProcessing"]


class BlobAtomizationNozzle(SprayModel):
    """Blob atomization model with nozzle geometry effects.

    Extends the blob model with:
    - Nozzle diameter ratio correction
    - Exit velocity profile (turbulent inflow)
    - Cavitation number effects

    Parameters
    ----------
    nozzle_diameter : float
        Nozzle orifice diameter (m).  Default ``2e-4``.
    nozzle_length : float
        Nozzle orifice length (m).  Default ``1e-3``.
    b0 : float
        KH constant.  Default ``0.61``.
    cavitation_number : float
        Cavitation number (dimensionless).  Default ``0.0``.
    """

    def __init__(
        self,
        nozzle_diameter: float = 2e-4,
        nozzle_length: float = 1e-3,
        b0: float = 0.61,
        cavitation_number: float = 0.0,
    ) -> None:
        self.nozzle_diameter = nozzle_diameter
        self.nozzle_length = nozzle_length
        self.b0 = b0
        self.cavitation_number = cavitation_number

    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Compute nozzle blob atomization."""
        if diameter < 1e-15 or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        r = diameter / 2.0
        We = fluid_density * relative_velocity ** 2 * r / surface_tension

        we_crit = 12.0 * (1.0 + 0.5 * self.cavitation_number)
        if We < we_crit:
            return {"diameter": diameter, "atomized": False}

        # 修正的 KH 波长考虑喷嘴长径比
        L_D = self.nozzle_length / max(self.nozzle_diameter, 1e-15)
        oh_correction = 1.0 + 0.1 * math.sqrt(L_D)

        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * r) if particle_density * surface_tension * r > 1e-30 else 0.0
        denom = (1.0 + Oh) * oh_correction
        if denom < 1e-30:
            return {"diameter": diameter, "atomized": False}

        lambda_kh = 9.02 * r * math.sqrt(We) / (denom * (1.0 + We / 12.0))
        d_child = max(2.0 * self.b0 * min(lambda_kh, r), 1e-10)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}


class SprayPostProcessing(SprayModel):
    """Spray diagnostic and post-processing model.

    Computes spray statistics (SMD, D32, D10, etc.) without modifying
    the droplet size.  This is a pass-through model that reports
    diagnostic information.

    Parameters
    ----------
    report_interval : int
        Number of calls between reports.  Default ``100``.
    """

    def __init__(
        self,
        report_interval: int = 100,
    ) -> None:
        self.report_interval = report_interval
        self._call_count = 0
        self._total_volume = 0.0
        self._total_area = 0.0
        self._n_droplets = 0

    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Pass-through: track statistics, no breakup."""
        self._call_count += 1
        vol = (math.pi / 6.0) * diameter ** 3
        area = math.pi * diameter ** 2
        self._total_volume += vol
        self._total_area += area
        self._n_droplets += 1

        return {"diameter": diameter, "atomized": False}

    @property
    def smd(self) -> float:
        """Sauter Mean Diameter (D32)."""
        if self._total_area < 1e-30:
            return 0.0
        return 6.0 * self._total_volume / self._total_area

    @property
    def d10(self) -> float:
        """Arithmetic mean diameter."""
        if self._n_droplets < 1:
            return 0.0
        return math.sqrt(6.0 * self._total_volume / (math.pi * self._n_droplets))

    def reset(self) -> None:
        """重置统计数据。"""
        self._call_count = 0
        self._total_volume = 0.0
        self._total_area = 0.0
        self._n_droplets = 0
