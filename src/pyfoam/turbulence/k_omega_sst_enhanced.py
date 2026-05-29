"""
Enhanced k-omega SST turbulence model — Menter 2003 variant.

Implements the SST model following Menter et al. (2003) with:

- Improved cross-diffusion term with c1 limiter
- Updated blending functions
- Better separated flow behaviour

References:
    Menter, F.R., Kuntz, M., Langtry, R. (2003). "Ten Years of
    Industrial Experience with the SST Turbulence Model." Turbulence,
    Heat and Mass Transfer 4.

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced import KOmegaSSTEnhancedModel

    model = KOmegaSSTEnhancedModel(mesh, U, phi)
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

__all__ = ["KOmegaSSTEnhancedModel", "KOmegaSSTEnhancedConstants"]


@dataclass(frozen=True)
class KOmegaSSTEnhancedConstants:
    """Constants for enhanced k-omega SST 2003 model.

    Attributes:
        sigma_k1: Turbulent Prandtl number for k (inner).
        sigma_k2: Turbulent Prandtl number for k (outer).
        sigma_omega1: Turbulent Prandtl number for omega (inner).
        sigma_omega2: Turbulent Prandtl number for omega (outer).
        beta1: Destruction coefficient for omega (inner).
        beta2: Destruction coefficient for omega (outer).
        gamma1: Production coefficient for omega (inner).
        gamma2: Production coefficient for omega (outer).
        a1: SST blending constant.
        beta_star: k destruction coefficient.
        kappa: von Karman constant.
        c1: Cross-diffusion limiter constant (2003 addition).
    """

    sigma_k1: float = 0.85
    sigma_k2: float = 1.0
    sigma_omega1: float = 0.5
    sigma_omega2: float = 0.856
    beta1: float = 0.075
    beta2: float = 0.0828
    gamma1: float = 5.0 / 9.0
    gamma2: float = 0.44
    a1: float = 0.31
    beta_star: float = 0.09
    kappa: float = 0.41
    c1: float = 10.0


_DEFAULTS = KOmegaSSTEnhancedConstants()


@TurbulenceModel.register("kOmegaSST2003Enhanced")
class KOmegaSSTEnhancedModel(TurbulenceModel):
    """Enhanced k-omega SST model (Menter 2003).

    Improvements over 1994 SST:
    - Cross-diffusion term with c1 limiter
    - Improved blending with F3 transition function
    - Better behaviour for separated flows

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTEnhancedConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTEnhancedConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

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
    def omega_field(self) -> torch.Tensor:
        """Specific dissipation rate ``(n_cells,)``."""
        return self._omega

    @omega_field.setter
    def omega_field(self, value: torch.Tensor) -> None:
        self._omega = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # TurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with SST limiter.

        mu_t = rho a1 k / max(a1 omega, S F2)
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)

        if self._grad_U is None:
            return k / omega

        S = self._strain_magnitude()
        F2 = self._F2()

        denominator = (self._C.a1 * omega).max(S * F2)
        return self._C.a1 * k / denominator.clamp(min=1e-16)

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def omega(self) -> torch.Tensor:
        """Return specific dissipation rate ``(n_cells,)``."""
        return self._omega

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate: epsilon = beta* omega k."""
        return self._C.beta_star * self._omega * self._k

    def correct(self) -> None:
        """Update the enhanced k-omega SST model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        self._solve_k(P_k)
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Blending functions
    # ------------------------------------------------------------------

    def _F1(self) -> torch.Tensor:
        """First blending function F1 (inner/outer transition)."""
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y**2 * omega)
        CD_kω = (2.0 * C.sigma_omega2 / omega).clamp(min=1e-10)
        arg3 = 4.0 * C.sigma_omega2 * k / (CD_kω * y**2)

        arg = torch.min(torch.max(arg1, arg2), arg3)
        return torch.tanh(arg**4)

    def _F2(self) -> torch.Tensor:
        """Second blending function F2 (shear stress limiter)."""
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = 2.0 * torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y**2 * omega)

        arg = torch.max(arg1, arg2)
        return torch.tanh(arg**2)

    # ------------------------------------------------------------------
    # Transport equations
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve the k transport equation."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        F1 = self._F1()
        sigma_k = F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2
        nu_eff = self._nu + sigma_k * self.nut()

        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._k, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        source = P_k - C.beta_star * omega_safe * k_safe
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve the omega equation with 2003 cross-diffusion limiter."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        C = self._C

        F1 = self._F1()
        sigma_w = F1 * C.sigma_omega1 + (1.0 - F1) * C.sigma_omega2
        beta = F1 * C.beta1 + (1.0 - F1) * C.beta2
        gamma = F1 * C.gamma1 + (1.0 - F1) * C.gamma2

        nu_eff = self._nu + sigma_w * self.nut()

        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._omega, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        nut = self.nut()
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        source = gamma * P_k / nut.clamp(min=1e-16) - beta * omega_safe**2

        # 2003 cross-diffusion with c1 limiter
        if self._grad_U is not None:
            grad_k = fvc.grad(self._k, "Gauss linear", mesh=self._mesh)
            grad_omega = fvc.grad(
                self._omega, "Gauss linear", mesh=self._mesh
            )
            cross = (grad_k * grad_omega).sum(dim=1)
            CD_kw = (
                2.0 * (1.0 - F1) * C.sigma_omega2 * cross / omega_safe
            ).clamp(min=0.0)

            # c1 limiter: F3 = tanh(y * omega / nu * 0.01)
            F3 = torch.tanh(
                self._y.clamp(min=1e-10) * omega_safe / self._nu * 0.01
            )
            CD_limited = CD_kw * (1.0 + C.c1 * F3)
            source = source + CD_limited

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Strain rate tensor S = 0.5 (grad(U) + grad(U)^T)."""
        return 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))

    def _strain_magnitude(self) -> torch.Tensor:
        """Magnitude of strain rate |S| = sqrt(2 S:S)."""
        S = self._strain_rate()
        return torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhancedModel(n_cells={self._mesh.n_cells})"
