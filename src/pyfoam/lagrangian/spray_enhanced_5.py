"""
Enhanced spray models v5.

Adds NozzleFlowModel and SheetAtomization following OpenFOAM conventions.

- :class:`NozzleFlowModel`    — nozzle internal flow model
- :class:`SheetAtomization`   — sheet atomization model for flat sprays
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["NozzleFlowModel", "SheetAtomization"]


class NozzleFlowModel(SprayModel):
    """Nozzle internal flow model for spray initialization.

    Models the flow inside the nozzle orifice to determine the
    initial spray conditions.  Accounts for:
    - Discharge coefficient
    - Nozzle turbulence effects
    - Flash boiling correction

    Parameters
    ----------
    discharge_coefficient : float
        Nozzle discharge coefficient Cd.  Default ``0.7``.
    nozzle_diameter : float
        Nozzle orifice diameter (m).  Default ``2e-4``.
    nozzle_length : float
        Nozzle orifice length (m).  Default ``1e-3``.
    flash_boiling_correction : float
        Flash boiling correction factor.  Default ``1.0`` (no flash boiling).
    """

    def __init__(
        self,
        discharge_coefficient: float = 0.7,
        nozzle_diameter: float = 2e-4,
        nozzle_length: float = 1e-3,
        flash_boiling_correction: float = 1.0,
    ) -> None:
        self.discharge_coefficient = discharge_coefficient
        self.nozzle_diameter = nozzle_diameter
        self.nozzle_length = nozzle_length
        self.flash_boiling_correction = flash_boiling_correction

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
        """Compute nozzle-initialized atomization."""
        if diameter < 1e-15 or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}

        # 初始液滴直径 = nozzle_diameter * Cd
        d_initial = self.nozzle_diameter * self.discharge_coefficient
        d_initial *= self.flash_boiling_correction
        d_initial = max(d_initial, 1e-10)

        if d_initial < diameter:
            return {"diameter": d_initial, "atomized": True}
        return {"diameter": diameter, "atomized": False}


class SheetAtomization(SprayModel):
    """Sheet atomization model for flat fan sprays.

    Models the breakup of a liquid sheet into droplets using the
    Dombrowski & Johns (1963) correlation:

    .. math::

        d_{child} = C_1 \\cdot h^{2/3} \\cdot \\sigma^{1/3}
                    / (\\rho_g U^2)^{1/3}

    where h is the sheet thickness at breakup.

    Parameters
    ----------
    sheet_thickness : float
        Initial sheet thickness (m).  Default ``5e-5``.
    C1 : float
        Empirical coefficient.  Default ``1.88``.
    breakup_length : float
        Sheet breakup length (m).  Default ``1e-2``.
    """

    def __init__(
        self,
        sheet_thickness: float = 5e-5,
        C1: float = 1.88,
        breakup_length: float = 1e-2,
    ) -> None:
        self.sheet_thickness = sheet_thickness
        self.C1 = C1
        self.breakup_length = breakup_length

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
        """Compute sheet atomization."""
        if diameter < 1e-15 or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        h = self.sheet_thickness
        rho_g = fluid_density

        denom = rho_g * relative_velocity ** 2
        if denom < 1e-30:
            return {"diameter": diameter, "atomized": False}

        d_child = self.C1 * h ** (2.0 / 3.0) * surface_tension ** (1.0 / 3.0) / denom ** (1.0 / 3.0)
        d_child = max(d_child, 1e-10)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}
