"""
One-equation k subgrid-scale model for LES (Schumann 1975, Yoshizawa 1982).

Implements the one-equation model that solves a transport equation for
the subgrid-scale turbulent kinetic energy k_sgs.  The SGS viscosity
is then computed as:

    ν_sgs = C_k Δ √k_sgs

This model provides a more physical representation of the SGS
dissipation than the algebraic Smagorinsky model, particularly for
flows with significant backscatter or non-equilibrium effects.

Transport equation:

    ∂k_sgs/∂t + ∇·(U k_sgs) = ∇·((ν + ν_sgs/σ_k) ∇k_sgs)
                              + P_sgs - ε_sgs

where:
    P_sgs = -2 ν_sgs S_ij S_ij (production)
    ε_sgs = C_ε k_sgs^{3/2} / Δ (dissipation)

References
----------
Schumann, U. (1975). Subgrid scale model for finite difference
simulations of turbulent flows in plane channels and annuli.
Journal of Computational Physics, 18(4), 376–404.

Yoshizawa, A. (1982). Statistical modelling of a transport equation
for the kinetic energy dissipation rate. Physics of Fluids, 25(9),
1532–1538.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .les_model import LESModel
from .filter_width import compute_filter_width

__all__ = ["KEqnModel", "KEqnConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KEqnConstants:
    """Constants for the one-equation k SGS model.

    Attributes:
        C_k: Eddy-viscosity constant (default: 0.094).
        C_epsilon: Dissipation constant (default: 1.048).
        sigma_k: Turbulent Prandtl number for k (default: 1.0).
    """

    C_k: float = 0.094
    C_epsilon: float = 1.048
    sigma_k: float = 1.0


_DEFAULT_CONSTANTS = KEqnConstants()


# ---------------------------------------------------------------------------
# One-equation k model
# ---------------------------------------------------------------------------


class KEqnModel(LESModel):
    """One-equation k SGS model for LES.

    Solves a transport equation for the subgrid-scale turbulent kinetic
    energy and computes ν_sgs = C_k Δ √k_sgs.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    constants : KEqnConstants, optional
        Model constants.

    Examples::

        >>> model = KEqnModel(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.nut()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        constants: KEqnConstants | None = None,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._C = constants or _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Molecular viscosity (default for air at STP)
        self._nu: float = 1.5e-5

        # SGS turbulent kinetic energy
        self._k_sgs = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

    @property
    def k_sgs_field(self) -> torch.Tensor:
        """SGS turbulent kinetic energy ``(n_cells,)``."""
        return self._k_sgs

    @k_sgs_field.setter
    def k_sgs_field(self, value: torch.Tensor) -> None:
        self._k_sgs = value.to(device=self._device, dtype=self._dtype)

    def nut(self) -> torch.Tensor:
        """Compute the SGS turbulent viscosity.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity:
            ν_sgs = C_k Δ √k_sgs
        """
        return self._C.C_k * self._delta * torch.sqrt(self._k_sgs.clamp(min=1e-16))

    def k(self) -> torch.Tensor:
        """Return SGS turbulent kinetic energy (same as k_sgs_field)."""
        return self._k_sgs

    def correct(self) -> None:
        """Update the model with the current velocity field.

        Recomputes velocity gradients and solves the k_sgs transport equation.
        """
        # Compute velocity gradient and strain rate
        self._compute_gradients()

        # Solve k_sgs equation
        self._solve_k_sgs()

    def _solve_k_sgs(self) -> None:
        """Solve the SGS k transport equation.

        ∂k_sgs/∂t + ∇·(U k_sgs) = ∇·((ν + ν_sgs/σ_k) ∇k_sgs)
                                  + P_sgs - ε_sgs
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        nut = self.nut()

        # Effective diffusivity
        nu_eff = self._nu + nut / C.sigma_k

        # Build equation: convection + diffusion
        eqn = fvm.div(self._phi, self._k_sgs, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._k_sgs, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Production: P_sgs = 2 ν_sgs |S|²
        mag_S = self._mag_S.clamp(min=1e-16)
        P_sgs = 2.0 * nut * mag_S.pow(2)

        # Dissipation: ε_sgs = C_ε k_sgs^{3/2} / Δ
        k_safe = self._k_sgs.clamp(min=1e-16)
        delta_safe = self._delta.clamp(min=1e-10)
        eps_sgs = C.C_epsilon * k_safe.pow(1.5) / delta_safe

        # Source: production - dissipation
        source = P_sgs - eps_sgs
        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k_sgs = k_new.clamp(min=1e-10)
