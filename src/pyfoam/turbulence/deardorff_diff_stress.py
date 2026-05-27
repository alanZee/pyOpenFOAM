"""
Deardorff diffusion stress one-equation LES model.

Implements the Deardorff (1980) one-equation model that solves a
transport equation for the subgrid-scale turbulent kinetic energy
k_sgs.  The SGS viscosity is computed as:

    ν_sgs = C_k Δ √k_sgs

The transport equation is:

    ∂k_sgs/∂t + ∇·(U k_sgs) = P_k + ∇·((ν + ν_sgs) ∇k_sgs)
                                - C_ε k_sgs^{3/2} / Δ

where:
    P_k = 2 ν_sgs |S|²  (production by resolved strain)

This model is similar to KEqnModel but uses the full effective
viscosity (ν + ν_sgs) for diffusion without a turbulent Prandtl
number divisor, following Deardorff's original formulation.

References
----------
Deardorff, J.W. (1980). Stratocumulus-capped mixed layers derived
from a three-dimensional model. Boundary-Layer Meteorology, 18(4),
495–527.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm

from .les_model import LESModel

__all__ = ["DeardorffDiffStressModel", "DeardorffDiffStressConstants"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeardorffDiffStressConstants:
    """Constants for the Deardorff diffusion stress SGS model.

    Attributes:
        C_k: Eddy-viscosity constant (default: 0.1).
        C_epsilon: Dissipation constant (default: 1.0).
    """

    C_k: float = 0.1
    C_epsilon: float = 1.0


_DEFAULT_CONSTANTS = DeardorffDiffStressConstants()


# ---------------------------------------------------------------------------
# Deardorff diffusion stress model
# ---------------------------------------------------------------------------


class DeardorffDiffStressModel(LESModel):
    """Deardorff diffusion stress one-equation SGS model.

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
    constants : DeardorffDiffStressConstants, optional
        Model constants.

    Examples::

        >>> model = DeardorffDiffStressModel(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.nut()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        constants: DeardorffDiffStressConstants | None = None,
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
        self._compute_gradients()
        self._solve_k_sgs()

    def _solve_k_sgs(self) -> None:
        """Solve the SGS k transport equation.

        ∂k_sgs/∂t + ∇·(U k_sgs) = P_k + ∇·((ν + ν_sgs) ∇k_sgs)
                                    - C_ε k_sgs^{3/2} / Δ
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype
        C = self._C

        nut = self.nut()

        # Effective diffusivity: ν + ν_sgs (no sigma_k divisor)
        nu_eff = self._nu + nut

        # Convection: ∇·(U k_sgs)
        eqn = fvm.div(self._phi, self._k_sgs, "Gauss upwind", mesh=mesh)

        # Diffusion: ∇·((ν + ν_sgs) ∇k_sgs)
        diff = fvm.laplacian(
            nu_eff, self._k_sgs, "Gauss linear corrected", mesh=mesh
        )
        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Production: P_k = 2 ν_sgs |S|²
        mag_S = self._mag_S.clamp(min=1e-16)
        P_k = 2.0 * nut * mag_S.pow(2)

        # Dissipation: ε = C_ε k_sgs^{3/2} / Δ
        k_safe = self._k_sgs.clamp(min=1e-16)
        delta_safe = self._delta.clamp(min=1e-10)
        eps = C.C_epsilon * k_safe.pow(1.5) / delta_safe

        # Source = production - dissipation
        source = P_k - eps
        eqn.source = eqn.source + source

        # Solve (explicit relaxation)
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k_sgs = k_new.clamp(min=1e-10)
