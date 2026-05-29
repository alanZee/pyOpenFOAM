"""
Enhanced k-epsilon turbulence model v3 — realizable variant with improved C_mu.

Extends :class:`~pyfoam.turbulence.k_epsilon_enhanced_2.KEpsilonEnhanced2Model`
with:

- Third-order dynamic C_mu with tensor-invariant formulation
- Enhanced realizability with SSS (Schumann-Stephens-Sanders) correction
- Improved wall-reflection with elliptic relaxation

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_3 import KEpsilonEnhanced3Model

    model = KEpsilonEnhanced3Model(mesh, U, phi)
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
from .k_epsilon_enhanced_2 import KEpsilonEnhanced2Model, KEpsilonEnhanced2Constants

__all__ = ["KEpsilonEnhanced3Model", "KEpsilonEnhanced3Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced3Constants(KEpsilonEnhanced2Constants):
    """Constants for enhanced realizable k-epsilon v3.

    Extends parent constants with:
        C_sss: SSS realizability correction coefficient.
        lambda_relax: Elliptic relaxation parameter.
        C_eta_3: Third-order strain invariant limiter.
    """

    C_sss: float = 0.3
    lambda_relax: float = 0.1
    C_eta_3: float = 2.0


_DEFAULTS = KEpsilonEnhanced3Constants()

_SQRT6_COS_PI3 = 6.0**0.5 * 0.5


@TurbulenceModel.register("realizableKEEnhanced3")
class KEpsilonEnhanced3Model(KEpsilonEnhanced2Model):
    """Enhanced realizable k-epsilon v3.

    Features:
    - Third-order dynamic C_mu with tensor invariants (S, Omega, S:Omega)
    - SSS realizability correction for epsilon production
    - Elliptic-relaxation wall-reflection correction
    - Low-Re correction via improved damping function

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KEpsilonEnhanced3Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced3Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip KEpsilonEnhanced2Model.__init__)
        super(KEpsilonEnhanced2Model, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()
        # Elliptic relaxation field (initialised to 1)
        self._f_relax = torch.ones(n_cells, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Third-order dynamic C_mu
    # ------------------------------------------------------------------

    def _compute_C_mu(self) -> torch.Tensor:
        """Third-order dynamic C_mu with full tensor invariants.

        C_mu = 1 / (A0 + A_s * U* * k / epsilon + A_3 * III_S * k^2 / eps^2)

        where III_S is the third invariant of the strain rate tensor.
        """
        if self._grad_U is None:
            return torch.full_like(self._k, self._C.C_mu_base)

        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        # Enhanced U* with third-order term
        C = self._C
        U_star = torch.sqrt(
            S_mag**2
            + Omega_mag**2
            + getattr(C, 'C_eta_2', 5.0) * S_mag * Omega_mag
        ).clamp(min=1e-16)

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # Third invariant: det(S) approximation via diagonal product
        S_diag = S.diagonal(dim1=-2, dim2=-1)
        III_S = (S_diag[:, 0] * S_diag[:, 1] * S_diag[:, 2]).abs()

        C_eta_3 = getattr(C, 'C_eta_3', 2.0)
        denominator = (
            C.A0
            + _SQRT6_COS_PI3 * U_star * k_safe / eps_safe
            + C_eta_3 * III_S * k_safe.pow(2) / eps_safe.pow(2)
        ).clamp(min=1e-10)

        return (1.0 / denominator).clamp(min=0.001, max=0.5)

    # ------------------------------------------------------------------
    # Elliptic relaxation for wall-reflection
    # ------------------------------------------------------------------

    def _update_relaxation(self) -> None:
        """Update elliptic relaxation field.

        f_relax = 1 - exp(-y * sqrt(eps / nu^3) * lambda_relax)
        """
        C = self._C
        eps_safe = self._eps.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)
        y = self._y.clamp(min=1e-10)

        length_ratio = y * torch.sqrt(eps_safe / nu ** 3)
        lam = getattr(C, 'lambda_relax', 0.1)
        self._f_relax = (1.0 - torch.exp(-length_ratio * lam)).clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Override epsilon solver
    # ------------------------------------------------------------------

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve epsilon with SSS correction and elliptic-relaxation wall model.

        SSS: Production limited by realizability constraint:
            eps_prod = C1 * S_mag * eps * f_SSS
            where f_SSS = max(1 - C_sss * |Omega|^2 / |S|^2, 0.3)
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

        # SSS realizability correction
        if self._grad_U is not None:
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            C_sss = getattr(C, 'C_sss', 0.3)
            f_sss = (1.0 - C_sss * Omega_mag.pow(2) / S_mag.pow(2).clamp(min=1e-30)).clamp(min=0.3, max=1.0)
        else:
            f_sss = torch.ones_like(S_mag)

        # Source with SSS correction
        nu_eps = (self._nu * eps_safe).sqrt()
        source = C.C1 * S_mag * eps_safe * f_sss - C.C2 * eps_safe**2 / (k_safe + nu_eps)

        # Elliptic-relaxation wall-reflection
        self._update_relaxation()
        C_w = getattr(C, 'C_w', 0.3)
        if C_w > 0:
            u_tau = C.C_mu_base ** 0.25 * k_safe.sqrt()
            y_plus = (u_tau * self._y / max(self._nu, 1e-30)).clamp(min=0.01)
            wall_damp = torch.exp(-y_plus / getattr(C, 'A_w', 1.0))
            source = source + C_w * eps_safe**2 / k_safe * wall_damp * self._f_relax

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced3Model(n_cells={self._mesh.n_cells})"
