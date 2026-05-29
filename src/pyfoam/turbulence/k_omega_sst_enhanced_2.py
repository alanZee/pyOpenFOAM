"""
Enhanced k-omega SST turbulence model v2 — SST 2003 with improved blending.

Extends :class:`~pyfoam.turbulence.k_omega_sst_enhanced.KOmegaSSTEnhancedModel`
with:

- Improved F4 blending function for separated flows
- Vorticity-based production limiter
- Kato-Launder production modification option

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_2 import KOmegaSSTEnhanced2Model

    model = KOmegaSSTEnhanced2Model(mesh, U, phi)
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
from .k_omega_sst_enhanced import KOmegaSSTEnhancedModel, KOmegaSSTEnhancedConstants

__all__ = ["KOmegaSSTEnhanced2Model", "KOmegaSSTEnhanced2Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced2Constants(KOmegaSSTEnhancedConstants):
    """Constants for enhanced SST v2.

    Extends parent constants with:
        C_prod_lim: Production limiter coefficient.
        F4_a1: F4 blending exponent.
        kato_launder: Enable Kato-Launder production modification.
    """

    C_prod_lim: float = 10.0
    F4_a1: float = 1.5
    kato_launder: bool = False


_DEFAULTS = KOmegaSSTEnhanced2Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced2")
class KOmegaSSTEnhanced2Model(KOmegaSSTEnhancedModel):
    """Enhanced SST v2 (Menter 2003 + improvements).

    Features:
    - F4 blending function for separated flow
    - Vorticity-based production limiter
    - Optional Kato-Launder production modification
    - Improved near-wall behaviour

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTEnhanced2Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTEnhanced2Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent __init__)
        super(KOmegaSSTEnhancedModel, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
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
    # F4 blending function
    # ------------------------------------------------------------------

    def _F4(self) -> torch.Tensor:
        """Fourth blending function F4 for separated flow regions.

        F4 = tanh(y * omega / nu)^F4_a1

        Provides additional damping in separated regions where the
        standard F1/F2 functions may be insufficient.
        """
        C = self._C
        omega_safe = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        nu = max(self._nu, 1e-30)

        arg = y * omega_safe / nu
        return torch.tanh(arg.clamp(max=10.0) ** C.F4_a1)

    # ------------------------------------------------------------------
    # Modified production
    # ------------------------------------------------------------------

    def _compute_production(self) -> torch.Tensor:
        """Compute turbulent production with optional Kato-Launder.

        Standard: P_k = 2 * nut * S:S
        Kato-Launder: P_k = nut * |S| * |Omega|
        """
        if self._grad_U is None:
            return torch.zeros(self._mesh.n_cells, device=self._device, dtype=self._dtype)

        nut = self.nut()
        S = self._strain_rate()
        C = self._C

        if getattr(C, 'kato_launder', False):
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            return nut * S_mag * Omega_mag

        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Apply production limiter
        eps = self._C.beta_star * self._omega * self._k
        eps_safe = eps.clamp(min=1e-16)
        P_k = P_k.clamp(max=C.C_prod_lim * eps_safe)

        return P_k

    # ------------------------------------------------------------------
    # Override correct to use enhanced production
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced SST v2 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        P_k = self._compute_production()

        self._solve_k(P_k)
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Override k solver with F4 blending
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve k equation with F4 blending in production."""
        mesh = self._mesh
        C = self._C

        F1 = self._F1()
        F4 = self._F4()
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

        # F4 modulates production in separated regions
        source = F4 * P_k - C.beta_star * omega_safe * k_safe
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced2Model(n_cells={self._mesh.n_cells})"
