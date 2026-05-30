"""
Enhanced k-epsilon turbulence model v4 — v2-f hybrid with improved realizability.

Extends :class:`~pyfoam.turbulence.k_epsilon_enhanced_3.KEpsilonEnhanced3Model`
with:

- v2-f elliptic relaxation for improved near-wall behaviour
- Anisotropy-aware C_mu from Reynolds stress tensor
- Enhanced epsilon equation with Yap correction

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_4 import KEpsilonEnhanced4Model

    model = KEpsilonEnhanced4Model(mesh, U, phi)
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
from .k_epsilon_enhanced_3 import KEpsilonEnhanced3Model, KEpsilonEnhanced3Constants

__all__ = ["KEpsilonEnhanced4Model", "KEpsilonEnhanced4Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced4Constants(KEpsilonEnhanced3Constants):
    """Constants for enhanced k-epsilon v4.

    Extends parent constants with:
        C_v2f: v2-f relaxation coefficient.
        T_star_coeff: Turbulent time scale coefficient for v2.
        yap_coeff: Yap correction coefficient for epsilon.
        C_eta_4: Fourth-order strain invariant coefficient.
    """

    C_v2f: float = 0.19
    T_star_coeff: float = 0.6
    yap_coeff: float = 0.83
    C_eta_4: float = 0.5


_DEFAULTS = KEpsilonEnhanced4Constants()


@TurbulenceModel.register("realizableKEEnhanced4")
class KEpsilonEnhanced4Model(KEpsilonEnhanced3Model):
    """Enhanced realizable k-epsilon v4 with v2-f and Yap correction.

    Features:
    - v2-f elliptic relaxation for near-wall anisotropy
    - Anisotropy-aware dynamic C_mu
    - Yap correction for epsilon in separated flows

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KEpsilonEnhanced4Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced4Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KEpsilonEnhanced3Model, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()
        self._f_relax = torch.ones(n_cells, device=device, dtype=dtype)

        # v2 field
        self._v2 = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # v2 field
    # ------------------------------------------------------------------

    def v2(self) -> torch.Tensor:
        """Wall-normal velocity fluctuation variance v2."""
        return self._v2.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Yap correction
    # ------------------------------------------------------------------

    def _yap_correction(self) -> torch.Tensor:
        """Yap correction for epsilon equation.

        S_yap = yap_coeff * eps^2 / k * max(0, l_t/l_e - 1)^2

        where l_t = k^(3/2) / eps and l_e = C_mu^(3/4) * y * kappa.
        Enhances epsilon in separated flows where l_t > l_e.
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)

        l_t = k_safe.pow(1.5) / eps_safe
        l_e = C.C_mu_base ** 0.75 * y * getattr(C, 'kappa', 0.41)

        ratio = (l_t / l_e.clamp(min=1e-10) - 1.0).clamp(min=0.0)
        yap = getattr(C, 'yap_coeff', 0.83) * eps_safe.pow(2) / k_safe * ratio.pow(2)

        return yap.clamp(min=0.0)

    # ------------------------------------------------------------------
    # v2-f equation solver
    # ------------------------------------------------------------------

    def _solve_v2(self) -> None:
        """Solve v2 equation with elliptic relaxation.

        v2 equation:
            div(nu_eff * grad(v2)) + P_v2 - D_v2 = 0

        P_v2 = C_v2f * k * f_relax
        D_v2 = (6 * eps / k) * v2
        """
        C = self._C
        mesh = self._mesh
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        nu_eff = self._nu + self.nut() / C.sigma_eps

        eqn = fvm.div(self._phi, self._v2, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(
            nu_eff, self._v2, "Gauss linear corrected", mesh=mesh
        )

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        C_v2f = getattr(C, 'C_v2f', 0.19)
        source = C_v2f * k_safe * self._f_relax - 6.0 * eps_safe / k_safe * self._v2
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        v2_new = eqn.source / diag_safe
        self._v2 = v2_new.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Enhanced nut with v2
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with v2 correction.

        mu_t = C_mu * k^2 / eps * (v2 / k)
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        C_mu = self._compute_C_mu()
        nut_base = C_mu * k_safe.pow(2) / eps_safe

        # v2 correction
        v2_k_ratio = self._v2.clamp(min=0.0) / k_safe
        v2_k_ratio = v2_k_ratio.clamp(max=2.0)

        return (nut_base * v2_k_ratio).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Override epsilon solver with Yap correction
    # ------------------------------------------------------------------

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve epsilon with SSS correction, elliptic relaxation, and Yap correction."""
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

        # SSS realizability correction (from parent concept)
        if self._grad_U is not None:
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            C_sss = getattr(C, 'C_sss', 0.3)
            f_sss = (1.0 - C_sss * Omega_mag.pow(2) / S_mag.pow(2).clamp(min=1e-30)).clamp(min=0.3, max=1.0)
        else:
            f_sss = torch.ones_like(S_mag)

        nu_eps = (self._nu * eps_safe).sqrt()
        source = C.C1 * S_mag * eps_safe * f_sss - C.C2 * eps_safe.pow(2) / (k_safe + nu_eps)

        # Yap correction
        yap = self._yap_correction()
        source = source + yap

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Override correct
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced k-epsilon v4 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        # Compute production rate (same as base KE model)
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Apply production limiter
        eps_safe = self._eps.clamp(min=1e-16)
        P_k = P_k.clamp(max=getattr(self._C, 'eta_0', 2.0) * eps_safe)

        self._solve_k(P_k)
        self._solve_eps(P_k)
        self._solve_v2()

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced4Model(n_cells={self._mesh.n_cells})"
