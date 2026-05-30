"""
Enhanced reacting models v3.

Adds ReactingMultiphaseCloud and TwoPhaseReacting following OpenFOAM conventions.

- :class:`ReactingMultiphaseCloud` — reacting cloud with multiple phases
- :class:`TwoPhaseReacting`        — two-phase heterogeneous reaction model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.reacting_models import ReactingModel

__all__ = ["ReactingMultiphaseCloud", "TwoPhaseReacting"]


class ReactingMultiphaseCloud(ReactingModel):
    """Reacting cloud model for multiphase particle systems.

    Handles particles containing multiple phases (solid, liquid, gas)
    with phase-specific reaction kinetics.  The total reaction rate is
    the sum of individual phase contributions.

    Parameters
    ----------
    phases : list[dict]
        Phase definitions:
        - ``"name"``: phase name
        - ``"Y0"``: initial mass fraction
        - ``"rho"``: phase density (kg/m³)
        - ``"A"``: pre-exponential factor
        - ``"Ea"``: activation energy (J/mol)
        - ``"heat"``: heat of reaction (J/kg)
        - ``"T_onset"``: onset temperature (K)
    r_gas : float
        Universal gas constant.  Default ``8.314``.
    """

    def __init__(
        self,
        phases: list[dict] | None = None,
        r_gas: float = 8.314,
    ) -> None:
        if phases is None:
            phases = [
                {"name": "solid", "Y0": 0.5, "rho": 2000.0, "A": 1e3, "Ea": 8e4, "heat": -3e7, "T_onset": 600.0},
                {"name": "liquid", "Y0": 0.3, "rho": 800.0, "A": 1e4, "Ea": 5e4, "heat": -2e7, "T_onset": 400.0},
                {"name": "gas", "Y0": 0.2, "rho": 1.0, "A": 1e5, "Ea": 3e4, "heat": -4e7, "T_onset": 300.0},
            ]
        self.phases = phases
        self.r_gas = r_gas

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute multiphase reacting cloud."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        total_dm = 0.0
        total_heat = 0.0

        RT = self.r_gas * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        for phase in self.phases:
            Y = phase.get("Y0", 0.0)
            A = phase.get("A", 0.0)
            Ea = phase.get("Ea", 0.0)
            heat = phase.get("heat", 0.0)
            T_onset = phase.get("T_onset", 0.0)

            if temperature < T_onset or Y <= 0 or A <= 0:
                continue

            k = A * math.exp(-Ea / RT)
            dm_i = math.pi * diameter ** 2 * k * dt * Y * species_mass_fraction
            dm_i = max(dm_i, 0.0)

            total_dm += dm_i
            total_heat += dm_i * abs(heat)

        total_dm = min(total_dm, m_particle)

        if total_dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - total_dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": total_dm, "heat_release": total_heat}


class TwoPhaseReacting(ReactingModel):
    """Two-phase heterogeneous reaction model.

    Models the reaction between a solid/liquid particle and a gas-phase
    oxidiser with separate rates for the two phases:

    - Gas-phase diffusion to particle surface
    - Surface chemical reaction
    - Product desorption and removal

    Parameters
    ----------
    A_gas : float
        Gas-phase diffusion pre-factor.  Default ``5e-12``.
    A_surface : float
        Surface reaction pre-factor (m/s).  Default ``1.0``.
    E_a : float
        Surface reaction activation energy (J/mol).  Default ``8.0e4``.
    n_order : float
        Reaction order.  Default ``0.5``.
    """

    def __init__(
        self,
        A_gas: float = 5e-12,
        A_surface: float = 1.0,
        E_a: float = 8.0e4,
        n_order: float = 0.5,
    ) -> None:
        self.A_gas = A_gas
        self.A_surface = A_surface
        self.E_a = E_a
        self.n_order = n_order

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute two-phase heterogeneous reaction."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        if species_mass_fraction <= 0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        RT = 8.314 * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        # 气相扩散
        k_gas = self.A_gas * temperature ** 0.75 / max(diameter, 1e-15)

        # 表面反应
        k_surface = self.A_surface * math.exp(-self.E_a / RT)
        k_surface *= species_mass_fraction ** self.n_order

        # 串联
        if k_gas + k_surface < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}
        k_total = k_gas * k_surface / (k_gas + k_surface)

        dm = math.pi * diameter ** 2 * k_total * dt
        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = max(min(dm, m_particle), 0.0)

        if dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": dm, "heat_release": dm * 3e7}
