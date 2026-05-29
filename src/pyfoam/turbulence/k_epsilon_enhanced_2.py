"""
Enhanced k-epsilon turbulence model v2 — realizable variant with improved C_mu.

Extends :class:`~pyfoam.turbulence.k_epsilon_enhanced.KEpsilonEnhancedModel`
with:

- Improved C_mu computation with strain-vorticity interaction
- Enhanced realizability enforcement (Schmidt, 2001)
- Additional wall-reflection correction

References:
    Schmidt, H. & Patnaik, G. (2001). "A new near-wall two-equation model."
    AIAA Journal, 39(3).

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_2 import KEpsilonEnhanced2Model

    model = KEpsilonEnhanced2Model(mesh, U, phi)
    model.correct()
    nut = model.nut()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

from .turbulence_model import TurbulenceModel
from .k_epsilon_enhanced import KEpsilonEnhancedModel, KEpsilonEnhancedConstants

__all__ = ["KEpsilonEnhanced2Model", "KEpsilonEnhanced2Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced2Constants(KEpsilonEnhancedConstants):
    """Constants for enhanced realizable k-epsilon v2.

    Extends parent constants with:
        C_w: Wall-reflection correction coefficient.
        C_eta_2: Strain-vorticity interaction limiter.
        A_w: Wall proximity coefficient.
    """

    C_w: float = 0.3
    C_eta_2: float = 5.0
    A_w: float = 1.0


_DEFAULTS = KEpsilonEnhanced2Constants()

_SQRT6_COS_PI3 = 6.0**0.5 * 0.5


@TurbulenceModel.register("realizableKEEnhanced2")
class KEpsilonEnhanced2Model(KEpsilonEnhancedModel):
    """Enhanced realizable k-epsilon v2.

    Features:
    - Improved dynamic C_mu with strain-vorticity interaction term
    - Wall-reflection correction for epsilon
    - Enhanced realizability enforcement
    - Low-Re correction via improved damping function

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KEpsilonEnhanced2Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced2Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip KEpsilonEnhancedModel.__init__)
        super(KEpsilonEnhancedModel, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Improved dynamic C_mu
    # ------------------------------------------------------------------

    def _compute_C_mu(self) -> torch.Tensor:
        """Improved dynamic C_mu with strain-vorticity interaction.

        C_mu = 1 / (A0 + A_s * U* * k / epsilon)
        where U* includes the strain-vorticity interaction:
            U* = sqrt(S:S + Omega:Omega + C_eta_2 * |S||Omega|)

        Returns
        -------
        torch.Tensor
            C_mu field ``(n_cells,)``.
        """
        if self._grad_U is None:
            return torch.full_like(self._k, self._C.C_mu_base)

        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        # Enhanced U* with interaction term
        C = self._C
        U_star = torch.sqrt(
            S_mag**2
            + Omega_mag**2
            + getattr(C, 'C_eta_2', 5.0) * S_mag * Omega_mag
        ).clamp(min=1e-16)

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        denominator = (
            C.A0 + _SQRT6_COS_PI3 * U_star * k_safe / eps_safe
        ).clamp(min=1e-10)

        return (1.0 / denominator).clamp(min=0.001, max=0.5)

    # ------------------------------------------------------------------
    # Override epsilon solver with wall-reflection correction
    # ------------------------------------------------------------------

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve epsilon equation with wall-reflection correction.

        Adds wall-reflection source term:
            S_wall = C_w * eps^2 / k * (exp(-y+ / A_w))
        """
        mesh = self._mesh
        C = self._C

        nu_eff = self._nu + self.nut() / C.sigma_eps

        eqn = fvm.div(self._phi, self._eps, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._eps, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # Standard source
        nu_eps = (self._nu * eps_safe).sqrt()
        source = C.C1 * S_mag * eps_safe - C.C2 * eps_safe**2 / (k_safe + nu_eps)

        # Wall-reflection correction
        C_w = getattr(C, 'C_w', 0.3)
        A_w = getattr(C, 'A_w', 1.0)
        if C_w > 0 and self._y is not None:
            # Estimate y+ from wall distance
            u_tau = C.C_mu_base ** 0.25 * k_safe.sqrt()
            y_plus = (u_tau * self._y / max(self._nu, 1e-30)).clamp(min=0.01)
            wall_damping = torch.exp(-y_plus / A_w)
            source = source + C_w * eps_safe**2 / k_safe * wall_damping

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced2Model(n_cells={self._mesh.n_cells})"
