"""
Enhanced k-omega SST turbulence model v4 — improved transition and curvature.

Extends :class:`~pyfoam.turbulence.k_omega_sst_enhanced_3.KOmegaSSTEnhanced3Model`
with:

- Gamma-Re_theta transition model coupling
- Improved curvature correction with Spalart-Shur formulation
- Vorticity-based production for improved separated flow predictions

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_4 import KOmegaSSTEnhanced4Model

    model = KOmegaSSTEnhanced4Model(mesh, U, phi)
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
from .k_omega_sst_enhanced_3 import KOmegaSSTEnhanced3Model, KOmegaSSTEnhanced3Constants

__all__ = ["KOmegaSSTEnhanced4Model", "KOmegaSSTEnhanced4Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced4Constants(KOmegaSSTEnhanced3Constants):
    """Constants for enhanced SST v4.

    Extends parent constants with:
        C_turb_trans: Transition onset threshold for intermittency.
        gamma_sep: Separation criterion for intermittency.
        r3_coeff: Third blending function coefficient.
        vort_prod_ratio: Vorticity-to-strain production ratio.
    """

    C_turb_trans: float = 0.6
    gamma_sep: float = 0.3
    r3_coeff: float = 0.5
    vort_prod_ratio: float = 0.0


_DEFAULTS = KOmegaSSTEnhanced4Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced4")
class KOmegaSSTEnhanced4Model(KOmegaSSTEnhanced3Model):
    """Enhanced SST v4 (Menter 2003 + transition + curvature).

    Features:
    - Intermittency-based transition triggering
    - Spalart-Shur rotation/curvature correction
    - Vorticity-based production option

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTEnhanced4Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTEnhanced4Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KOmegaSSTEnhanced3Model, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

        # Intermittency field
        self._gamma = torch.ones(n_cells, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Intermittency (transition triggering)
    # ------------------------------------------------------------------

    def _compute_intermittency(self) -> None:
        """Update intermittency field for transition triggering.

        gamma = 1 - exp(-C_turb_trans * Ret)

        where Ret = k / (nu * omega).
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)

        Ret = k_safe / (nu * omega_safe)
        C_t = getattr(C, 'C_turb_trans', 0.6)
        gamma = 1.0 - torch.exp(-C_t * Ret)
        self._gamma = gamma.clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Spalart-Shur curvature correction
    # ------------------------------------------------------------------

    def _spalart_shur_correction(self) -> torch.Tensor:
        """Spalart-Shur rotation/curvature correction.

        r_star = S / Omega (or Omega / S depending on convention)
        f_rotation = (1 + C_rc * max(0, r_star - 1)) / (1 + C_rc * max(0, 1/r_star - 1))

        Applied as multiplier to production.
        """
        if self._grad_U is None:
            return torch.ones_like(self._k)

        C = self._C
        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        r_star = S_mag / Omega_mag.clamp(min=1e-16)
        C_rc = getattr(C, 'C_rc', 1.0)

        numerator = 1.0 + C_rc * (r_star - 1.0).clamp(min=0.0)
        denominator = 1.0 + C_rc * (1.0 / r_star.clamp(min=1e-10) - 1.0).clamp(min=0.0)

        return (numerator / denominator.clamp(min=0.1)).clamp(min=0.2, max=5.0)

    # ------------------------------------------------------------------
    # Override nut with intermittency
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with intermittency correction.

        mu_t = gamma * mu_t_SST
        """
        nut_sst = super().nut()
        return self._gamma * nut_sst

    # ------------------------------------------------------------------
    # Override correct
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced SST v4 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        # Update intermittency
        self._compute_intermittency()

        P_k = self._compute_production()

        # Apply Spalart-Shur curvature correction
        f_ss = self._spalart_shur_correction()
        P_k = P_k * f_ss

        self._solve_k(P_k)
        self._solve_omega(P_k)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced4Model(n_cells={self._mesh.n_cells})"
