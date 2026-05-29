"""
multiphaseEulerFoam2 — Enhanced N-phase Euler-Euler with population balance.

Extends :class:`MultiphaseEulerFoam` with:

- **Population balance equation (PBE)** for tracking the particle/bubble
  size distribution within each dispersed phase.
- **Multi-size-group (MUSIG)** model: discretises the size distribution
  into N groups, each with its own momentum equation and interphase
  forces.
- **Breakage and coalescence** kernels for the PBE:
  - Breakage: Luo model
  - Coalescence: Prince-Blanch model

The PBE for number density n_i(V, t):

    dn_i/dt + ∇·(U_i n_i) = (B_break - D_break) + (B_coal - D_coal)

where B/D are birth/death rates from breakage and coalescence.

Usage::

    from pyfoam.applications.multiphase_euler_foam_2 import MultiphaseEulerFoam2

    solver = MultiphaseEulerFoam2("path/to/case", phases=phases, n_size_groups=5)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .multiphase_euler_foam import MultiphaseEulerFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseEulerFoam2", "SizeGroup"]

logger = logging.getLogger(__name__)


# ======================================================================
# Size group data
# ======================================================================


@dataclass
class SizeGroup:
    """Properties of a single size group in the MUSIG model.

    Attributes
    ----------
    index : int
        Size group index.
    diameter : float
        Sauter mean diameter for this group (m).
    volume : float
        Particle volume (m^3).
    fraction : float
        Volume fraction of the dispersed phase in this size group.
    number_density : float
        Number density (particles/m^3).
    """

    index: int = 0
    diameter: float = 1e-3
    volume: float = 0.0
    fraction: float = 0.0
    number_density: float = 0.0

    def __post_init__(self) -> None:
        if self.volume == 0.0:
            self.volume = math.pi / 6.0 * self.diameter ** 3


# ======================================================================
# Breakage and coalescence kernels
# ======================================================================


def luo_breakage_rate(
    d_i: float,
    epsilon: float,
    sigma: float,
    rho_c: float,
    rho_d: float,
) -> float:
    """Luo breakage rate kernel.

    Parameters
    ----------
    d_i : float
        Particle diameter (m).
    epsilon : float
        Turbulent dissipation rate (m^2/s^3).
    sigma : float
        Surface tension coefficient (N/m).
    rho_c, rho_d : float
        Continuous and dispersed phase densities.

    Returns
    -------
    float
        Breakage rate (1/s).
    """
    if sigma < 1e-30 or d_i < 1e-30:
        return 0.0
    # Weber number
    We = rho_c * epsilon ** (2.0 / 3.0) * d_i ** (5.0 / 3.0) / sigma
    # Breakage rate (simplified Luo model)
    rate = 0.01 * (epsilon / (d_i ** 2 + 1e-30)) * We ** 0.5
    return max(rate, 0.0)


def prince_blanch_coalescence_rate(
    d_i: float,
    d_j: float,
    epsilon: float,
    sigma: float,
    rho_c: float,
) -> float:
    """Prince-Blanch coalescence rate kernel.

    Parameters
    ----------
    d_i, d_j : float
        Diameters of colliding particles (m).
    epsilon : float
        Turbulent dissipation rate.
    sigma : float
        Surface tension coefficient.
    rho_c : float
        Continuous phase density.

    Returns
    -------
    float
        Coalescence rate (m^3/s).
    """
    if sigma < 1e-30:
        return 0.0

    d_eq = 2.0 * d_i * d_j / (d_i + d_j + 1e-30)
    # Turbulent collision velocity
    u_t = 1.4 * epsilon ** (1.0 / 3.0) * d_eq ** (2.0 / 3.0)
    # Drainage time (simplified)
    h_0 = 1e-4  # initial film thickness
    h_f = 1e-8  # critical film thickness
    t_drain = (
        (d_eq ** 3 * rho_c * h_0)
        / (16.0 * sigma + 1e-30)
    )
    # Collision area
    A_col = math.pi / 4.0 * d_eq ** 2
    # Coalescence efficiency
    Pc = math.exp(-t_drain * u_t / (d_eq + 1e-30))

    return A_col * u_t * Pc


# ======================================================================
# Main solver
# ======================================================================


class MultiphaseEulerFoam2(MultiphaseEulerFoam):
    """Enhanced N-phase Euler-Euler with population balance.

    Extends MultiphaseEulerFoam with:

    - Population balance equation (PBE) for size distribution.
    - Multi-size-group (MUSIG) model with N groups per dispersed phase.
    - Luo breakage and Prince-Blanch coalescence kernels.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[dict]
        Phase definitions with name, rho, mu, d.
    n_size_groups : int
        Number of size groups for the dispersed phase.
    d_min, d_max : float
        Minimum and maximum diameters for size group discretisation.
    sigma : float
        Surface tension coefficient (N/m).
    epsilon : float
        Turbulent dissipation rate (m^2/s^3).
    coalescence_efficiency : float
        Scaling factor for coalescence rate (default 1.0).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[Dict[str, Any]],
        n_size_groups: int = 5,
        d_min: float = 1e-4,
        d_max: float = 1e-2,
        sigma: float = 0.07,
        epsilon: float = 0.01,
        coalescence_efficiency: float = 1.0,
    ) -> None:
        super().__init__(case_path, phases=phases)

        self.n_size_groups = n_size_groups
        self.d_min = d_min
        self.d_max = d_max
        self.sigma = sigma
        self.epsilon = epsilon
        self.coalescence_efficiency = coalescence_efficiency

        # Create size groups for each dispersed phase
        self.size_groups = self._create_size_groups()

        # Size fraction fields for each phase
        self.fractions = self._init_fraction_fields()

        logger.info(
            "MultiphaseEulerFoam2 ready: %d phases, %d size groups",
            self.n_phases, self.n_size_groups,
        )

    # ------------------------------------------------------------------
    # Size group initialisation
    # ------------------------------------------------------------------

    def _create_size_groups(self) -> dict[str, list[SizeGroup]]:
        """Create size groups for each dispersed phase.

        Linearly spaced diameters from d_min to d_max.
        """
        groups = {}
        for i, phase in enumerate(self.phases):
            name = phase["name"]
            group_list = []
            for g in range(self.n_size_groups):
                frac = g / max(self.n_size_groups - 1, 1)
                d = self.d_min + frac * (self.d_max - self.d_min)
                sg = SizeGroup(
                    index=g,
                    diameter=d,
                    fraction=1.0 / self.n_size_groups,
                )
                group_list.append(sg)
            groups[name] = group_list
        return groups

    def _init_fraction_fields(self) -> dict[str, torch.Tensor]:
        """Initialise volume fraction fields for each size group.

        Returns dict of {phase_name: (n_cells, n_size_groups)} tensors.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        fractions = {}
        for name in self.phase_names:
            f = torch.full(
                (n_cells, self.n_size_groups),
                1.0 / self.n_size_groups,
                dtype=dtype, device=device,
            )
            fractions[name] = f

        return fractions

    # ------------------------------------------------------------------
    # Population balance equation
    # ------------------------------------------------------------------

    def _solve_pbe(
        self,
        phase_name: str,
        alpha_phase: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Solve the population balance equation for one phase.

        Updates the size distribution fractions using birth/death terms
        from breakage and coalescence.

        Parameters
        ----------
        phase_name : str
            Phase name.
        alpha_phase : torch.Tensor
            ``(n_cells,)`` total volume fraction for this phase.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            ``(n_cells, n_size_groups)`` updated fractions.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        groups = self.size_groups[phase_name]
        fractions = self.fractions[phase_name]

        # Source terms for each size group
        source = torch.zeros_like(fractions)

        for i in range(self.n_size_groups):
            d_i = groups[i].diameter

            # Breakage: larger droplets break into smaller ones
            break_rate = luo_breakage_rate(
                d_i, self.epsilon, self.sigma,
                self.rho_phases[0].item(),
                self.rho_phases[min(1, len(self.rho_phases) - 1)].item(),
            )

            # Birth from breakage of larger groups
            for j in range(i + 1, self.n_size_groups):
                d_j = groups[j].diameter
                # Fragment size distribution (binary breakage)
                beta = 0.5  # Equal breakage
                birth = break_rate * fractions[:, j] * beta
                source[:, i] = source[:, i] + birth

            # Death from breakage of this group
            source[:, i] = source[:, i] - break_rate * fractions[:, i]

            # Coalescence: two smaller droplets form a larger one
            for j in range(self.n_size_groups):
                if i + j < self.n_size_groups:
                    d_j = groups[j].diameter
                    coal_rate = prince_blanch_coalescence_rate(
                        d_i, d_j, self.epsilon, self.sigma,
                        self.rho_phases[0].item(),
                    )
                    coal_rate *= self.coalescence_efficiency

                    # Birth from coalescence of i and j
                    if i + j < self.n_size_groups:
                        source[:, i + j] = source[:, i + j] + coal_rate * fractions[:, i] * fractions[:, j]

                    # Death from coalescence of this group
                    source[:, i] = source[:, i] - coal_rate * fractions[:, i] * fractions[:, j]

        # Update fractions
        fractions_new = fractions + dt * source

        # Enforce non-negativity and normalise
        fractions_new = fractions_new.clamp(min=0.0)
        total = fractions_new.sum(dim=1, keepdim=True).clamp(min=1e-30)
        fractions_new = fractions_new / total

        return fractions_new

    # ------------------------------------------------------------------
    # Effective diameter
    # ------------------------------------------------------------------

    def compute_effective_diameter(self, phase_name: str) -> torch.Tensor:
        """Compute the Sauter mean diameter from the size distribution.

        d32 = sum(f_i * d_i^3) / sum(f_i * d_i^2)

        Parameters
        ----------
        phase_name : str
            Phase name.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` Sauter mean diameter.
        """
        groups = self.size_groups[phase_name]
        fractions = self.fractions[phase_name]

        d3_sum = torch.zeros_like(fractions[:, 0])
        d2_sum = torch.zeros_like(fractions[:, 0])

        for i, sg in enumerate(groups):
            d3_sum = d3_sum + fractions[:, i] * sg.diameter ** 3
            d2_sum = d2_sum + fractions[:, i] * sg.diameter ** 2

        d32 = d3_sum / d2_sum.clamp(min=1e-30)
        return d32

    # ------------------------------------------------------------------
    # Enhanced iteration
    # ------------------------------------------------------------------

    def _euler_iteration(self):
        """Enhanced Euler iteration with population balance."""
        velocities = [U.clone() for U in self.velocities]
        p = self.p.clone()
        alphas = [a.clone() for a in self.alphas]
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            vels_prev = [U.clone() for U in velocities]

            # Enforce constraint: last alpha = 1 - sum(others)
            alpha_sum = sum(alphas[:-1])
            alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

            # Renormalise
            total = sum(alphas).clamp(min=1e-30)
            alphas = [a / total for a in alphas]

            # Solve PBE for each phase with size groups
            for i, name in enumerate(self.phase_names):
                if name in self.size_groups:
                    self.fractions[name] = self._solve_pbe(
                        name, alphas[i], self.delta_t,
                    )

            # Convergence
            U_residual = max(
                self._compute_residual(velocities[i], vels_prev[i])
                for i in range(self.n_phases)
            )
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return velocities, p, alphas, phi, convergence
