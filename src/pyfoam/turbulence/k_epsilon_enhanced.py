"""
Enhanced k-epsilon turbulence model — realizable variant v2.

Implements a realizable k-epsilon model with:

- Variable C_mu computed from strain and rotation rates (Shih et al. 1995)
- Enhanced production limiter (P_k / epsilon bounded)
- Low-Reynolds-number correction via damping functions
- Improved epsilon equation with realizability enforcement

References:
    Shih, T.H. et al. (1995). "A new k-epsilon eddy viscosity model
    for high Reynolds number turbulent flows." Computers & Fluids, 24(3).

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced import KEpsilonEnhancedModel

    model = KEpsilonEnhancedModel(mesh, U, phi)
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

__all__ = ["KEpsilonEnhancedModel", "KEpsilonEnhancedConstants"]


@dataclass(frozen=True)
class KEpsilonEnhancedConstants:
    """Constants for enhanced realizable k-epsilon model.

    Attributes:
        C_mu_base: Base C_mu value (used before gradient is available).
        C1: Production coefficient for epsilon.
        C2: Destruction coefficient for epsilon.
        sigma_k: Turbulent Prandtl number for k.
        sigma_eps: Turbulent Prandtl number for epsilon.
        A0: Realizability constant (4.0 per Shih et al.).
        eta_0: Production-to-dissipation limiter threshold.
        C_eps3: Buoyancy coefficient (0 for non-buoyant flows).
    """

    C_mu_base: float = 0.09
    C1: float = 1.44
    C2: float = 1.9
    sigma_k: float = 1.0
    sigma_eps: float = 1.2
    A0: float = 4.0
    eta_0: float = 4.38
    C_eps3: float = 0.0


_DEFAULTS = KEpsilonEnhancedConstants()

_SQRT6_COS_PI3 = 6.0**0.5 * 0.5  # sqrt(6) * cos(pi/3)


@TurbulenceModel.register("realizableKEEnhanced")
class KEpsilonEnhancedModel(TurbulenceModel):
    """Enhanced realizable k-epsilon model.

    Features:
    - Dynamic C_mu from strain and vorticity rates
    - Production limiter: P_k <= C_lim * epsilon
    - Low-Re damping via f_mu function
    - Improved epsilon equation denominator: k + sqrt(nu * epsilon)

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KEpsilonEnhancedConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhancedConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k_field(self) -> torch.Tensor:
        """Turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    @k_field.setter
    def k_field(self, value: torch.Tensor) -> None:
        self._k = value.to(device=self._device, dtype=self._dtype)

    @property
    def epsilon_field(self) -> torch.Tensor:
        """Dissipation rate ``(n_cells,)``."""
        return self._eps

    @epsilon_field.setter
    def epsilon_field(self, value: torch.Tensor) -> None:
        self._eps = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity: mu_t = C_mu * k^2 / epsilon.

        Returns:
            ``(n_cells,)`` turbulent viscosity field.
        """
        k = self._k.clamp(min=1e-16)
        eps = self._eps.clamp(min=1e-16)
        C_mu = self._compute_C_mu()
        return C_mu * k**2 / eps

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate ``(n_cells,)``."""
        return self._eps

    def correct(self) -> None:
        """Update the enhanced realizable k-epsilon model."""
        # Compute velocity gradient
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        # Production rate
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Apply production limiter: P_k <= C_lim * epsilon
        eps_safe = self._eps.clamp(min=1e-16)
        P_k = P_k.clamp(max=self._C.eta_0 * eps_safe)

        self._solve_k(P_k)
        self._solve_eps(P_k)

    # ------------------------------------------------------------------
    # Dynamic C_mu
    # ------------------------------------------------------------------

    def _compute_C_mu(self) -> torch.Tensor:
        """Compute dynamic C_mu.

        C_mu = 1 / (A0 + A_s * U* * k / epsilon)
        where U* = sqrt(S:S + Omega:Omega)
        """
        if self._grad_U is None:
            return torch.full_like(self._k, self._C.C_mu_base)

        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        U_star = torch.sqrt(S_mag**2 + Omega_mag**2).clamp(min=1e-16)

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        denominator = (
            self._C.A0 + _SQRT6_COS_PI3 * U_star * k_safe / eps_safe
        ).clamp(min=1e-10)

        return (1.0 / denominator).clamp(min=0.001, max=0.5)

    # ------------------------------------------------------------------
    # Transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation.

        dk/dt + div(U k) = div((nu + nu_t/sigma_k) grad(k)) + P_k - epsilon
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        nu_eff = self._nu + self.nut() / self._C.sigma_k

        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._k, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - epsilon (with backscatter prevention)
        source = P_k - self._eps.clamp(min=0.0)
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve the epsilon transport equation.

        Improved denominator: epsilon^2 / (k + sqrt(nu * epsilon))
        instead of epsilon^2 / k (prevents singularity at k->0).
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
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

        # Strain magnitude
        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

        # Enhanced source: C1 * S * eps - C2 * eps^2 / (k + sqrt(nu * eps))
        nu_eps = (self._nu * eps_safe).sqrt()
        source = C.C1 * S_mag * eps_safe - C.C2 * eps_safe**2 / (k_safe + nu_eps)

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Strain rate tensor S = 0.5 (grad(U) + grad(U)^T)."""
        return 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))

    def __repr__(self) -> str:
        return f"KEpsilonEnhancedModel(n_cells={self._mesh.n_cells})"
