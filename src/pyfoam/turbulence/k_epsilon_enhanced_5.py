"""
Enhanced k-epsilon turbulence model v5 — realizable with elliptic blending and RNG correction.

Extends :class:`~pyfoam.turbulence.k_epsilon_enhanced_4.KEpsilonEnhanced4Model`
with:

- Elliptic blending for improved near-wall to log-law transition
- RNG-based correction to C1 for better swirling flow predictions
- Strain-vorticity invariant realizability constraint

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_5 import KEpsilonEnhanced5Model

    model = KEpsilonEnhanced5Model(mesh, U, phi)
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
from .k_epsilon_enhanced_4 import KEpsilonEnhanced4Model, KEpsilonEnhanced4Constants

__all__ = ["KEpsilonEnhanced5Model", "KEpsilonEnhanced5Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced5Constants(KEpsilonEnhanced4Constants):
    """Constants for enhanced k-epsilon v5.

    Extends parent constants with:
        alpha_ebl: Elliptic blending coefficient.
        C1_rng: RNG correction factor for C1.
        eta_max: Maximum strain invariant for realizability.
    """

    alpha_ebl: float = 0.3
    C1_rng: float = 1.42
    eta_max: float = 4.38


_DEFAULTS = KEpsilonEnhanced5Constants()


@TurbulenceModel.register("realizableKEEnhanced5")
class KEpsilonEnhanced5Model(KEpsilonEnhanced4Model):
    """Enhanced realizable k-epsilon v5 with elliptic blending and RNG.

    Features:
    - Elliptic blending function for smooth near-wall to far-field transition
    - RNG correction to C1 for improved swirling flow behaviour
    - Strain-vorticity realizability: C_mu bounded by invariants

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KEpsilonEnhanced5Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced5Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KEpsilonEnhanced4Model, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()
        self._f_relax = torch.ones(n_cells, device=device, dtype=dtype)
        self._v2 = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)

        # Elliptic blending field
        self._alpha3 = torch.zeros(n_cells, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Elliptic blending
    # ------------------------------------------------------------------

    def _compute_blending(self) -> None:
        """Compute elliptic blending function alpha^3.

        alpha^3 = tanh((y * sqrt(k) / nu) * alpha_ebl)

        Smoothly blends between near-wall (alpha=0) and far-field (alpha=1).
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        nu = max(self._nu, 1e-30)

        y_star = y * k_safe.sqrt() / nu
        alpha_ebl = getattr(C, 'alpha_ebl', 0.3)

        alpha = torch.tanh((y_star * alpha_ebl).clamp(max=10.0))
        self._alpha3 = alpha.pow(3).clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # RNG correction to C1
    # ------------------------------------------------------------------

    def _C1_rng_corrected(self, S_mag: torch.Tensor) -> torch.Tensor:
        """RNG-corrected C1 coefficient.

        C1_eff = C1 * (1 + C1_rng * (eta / eta_0 - 1))

        where eta = S * k / eps and eta_0 = 4.38.
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        eta = S_mag * k_safe / eps_safe
        eta_0 = getattr(C, 'eta_max', 4.38)

        C1_rng = getattr(C, 'C1_rng', 1.42)
        ratio = (eta / eta_0).clamp(max=2.0)

        C1_eff = C.C1 * (1.0 + C1_rng * (ratio - 1.0).clamp(min=0.0))
        return C1_eff.clamp(min=C.C1 * 0.5, max=C.C1 * 2.0)

    # ------------------------------------------------------------------
    # Enhanced nut with blending
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with elliptic blending and v2 correction."""
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        C_mu = self._compute_C_mu()
        nut_base = C_mu * k_safe.pow(2) / eps_safe

        v2_k_ratio = self._v2.clamp(min=0.0) / k_safe
        v2_k_ratio = v2_k_ratio.clamp(max=2.0)

        # Blend: alpha3=1 -> full v2 correction, alpha3=0 -> standard
        nut = nut_base * (self._alpha3 * v2_k_ratio + (1.0 - self._alpha3) * 1.0)

        return nut.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Override epsilon solver with RNG and blending
    # ------------------------------------------------------------------

    def _solve_eps(self, P_k: torch.Tensor) -> None:
        """Solve epsilon with RNG correction, elliptic blending, and Yap correction."""
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

        # RNG-corrected C1
        C1_eff = self._C1_rng_corrected(S_mag)

        # SSS realizability
        if self._grad_U is not None:
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            C_sss = getattr(C, 'C_sss', 0.3)
            f_sss = (1.0 - C_sss * Omega_mag.pow(2) / S_mag.pow(2).clamp(min=1e-30)).clamp(min=0.3, max=1.0)
        else:
            f_sss = torch.ones_like(S_mag)

        nu_eps = (self._nu * eps_safe).sqrt()
        source = C1_eff * S_mag * eps_safe * f_sss - C.C2 * eps_safe.pow(2) / (k_safe + nu_eps)

        # Yap correction
        yap = self._yap_correction()
        source = source + yap

        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        eps_new = eqn.source / diag_safe
        self._eps = eps_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Override correct with blending update
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced k-epsilon v5 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        # Update blending
        self._compute_blending()

        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        eps_safe = self._eps.clamp(min=1e-16)
        P_k = P_k.clamp(max=getattr(self._C, 'eta_0', 2.0) * eps_safe)

        self._solve_k(P_k)
        self._solve_eps(P_k)
        self._solve_v2()

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced5Model(n_cells={self._mesh.n_cells})"
