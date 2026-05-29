"""
Enhanced k-omega SST turbulence model v3 — SST 2003 with improved blending.

Extends :class:`~pyfoam.turbulence.k_omega_sst_enhanced_2.KOmegaSSTEnhanced2Model`
with:

- Improved F3 blending function for transition prediction
- Strain-rate adaptive sigma_k blending
- Rotation-curvature correction (DDES-style)

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_3 import KOmegaSSTEnhanced3Model

    model = KOmegaSSTEnhanced3Model(mesh, U, phi)
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
from .k_omega_sst_enhanced_2 import KOmegaSSTEnhanced2Model, KOmegaSSTEnhanced2Constants

__all__ = ["KOmegaSSTEnhanced3Model", "KOmegaSSTEnhanced3Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced3Constants(KOmegaSSTEnhanced2Constants):
    """Constants for enhanced SST v3.

    Extends parent constants with:
        C_rc: Rotation-curvature correction coefficient.
        F3_coeff: F3 transition blending coefficient.
        sigma_k_adaptive: Enable adaptive sigma_k from strain ratio.
    """

    C_rc: float = 1.0
    F3_coeff: float = 0.005
    sigma_k_adaptive: bool = True


_DEFAULTS = KOmegaSSTEnhanced3Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced3")
class KOmegaSSTEnhanced3Model(KOmegaSSTEnhanced2Model):
    """Enhanced SST v3 (Menter 2003 + improvements).

    Features:
    - F3 transition prediction blending function
    - Adaptive sigma_k based on strain-to-omega ratio
    - Rotation-curvature correction for improved rotating flow predictions

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTEnhanced3Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTEnhanced3Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KOmegaSSTEnhanced2Model, self).__init__(mesh, U, phi)

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
    # F3 transition blending function
    # ------------------------------------------------------------------

    def _F3(self) -> torch.Tensor:
        """Third blending function F3 for transition prediction.

        F3 = 1 - tanh((y * omega / nu)^2 * F3_coeff)

        Complementary to F4 (in parent): F3 suppresses production near
        transition onset while F4 activates in separated regions.
        """
        C = self._C
        omega_safe = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        nu = max(self._nu, 1e-30)

        arg = (y * omega_safe / nu).pow(2) * getattr(C, 'F3_coeff', 0.005)
        return 1.0 - torch.tanh(arg.clamp(max=10.0))

    # ------------------------------------------------------------------
    # Rotation-curvature correction
    # ------------------------------------------------------------------

    def _rotation_curvature_factor(self) -> torch.Tensor:
        """Rotation-curvature correction factor.

        f_rc = 1 + C_rc * max(0, (|Omega|/|S| - 1))

        In pure rotation regions (|Omega| > |S|), this enhances production.
        In irrotational strain regions, no modification.
        """
        if self._grad_U is None:
            return torch.ones_like(self._k)

        C = self._C
        S = self._strain_rate()
        Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))

        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))

        ratio = Omega_mag / S_mag.clamp(min=1e-16)
        f_rc = 1.0 + getattr(C, 'C_rc', 1.0) * (ratio - 1.0).clamp(min=0.0)

        return f_rc.clamp(min=0.5, max=3.0)

    # ------------------------------------------------------------------
    # Adaptive sigma_k
    # ------------------------------------------------------------------

    def _adaptive_sigma_k(self, F1: torch.Tensor) -> torch.Tensor:
        """Adaptive sigma_k based on strain-to-omega ratio.

        sigma_k = sigma_k_blend * (1 + 0.1 * (S/omega)^0.5)

        This provides extra diffusion in high-strain regions.
        """
        C = self._C
        if not getattr(C, 'sigma_k_adaptive', True):
            return F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2

        omega_safe = self._omega.clamp(min=1e-16)

        if self._grad_U is not None:
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            strain_omega_ratio = (S_mag / omega_safe).clamp(max=100.0)
            adaptive_factor = 1.0 + 0.1 * strain_omega_ratio.sqrt()
        else:
            adaptive_factor = torch.ones_like(self._k)

        sigma_k_base = F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2
        return sigma_k_base * adaptive_factor

    # ------------------------------------------------------------------
    # Override correct
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced SST v3 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        P_k = self._compute_production()
        # Apply rotation-curvature correction to production
        f_rc = self._rotation_curvature_factor()
        P_k = P_k * f_rc

        self._solve_k(P_k)
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Override k solver with F3 and adaptive sigma_k
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve k equation with F3 blending and adaptive sigma_k."""
        mesh = self._mesh
        C = self._C

        F1 = self._F1()
        F4 = self._F4()
        F3 = self._F3()

        # Adaptive sigma_k
        sigma_k = self._adaptive_sigma_k(F1)
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

        # Combine F3 and F4 for production modulation
        blend_factor = F3 * F4
        source = blend_factor * P_k - C.beta_star * omega_safe * k_safe
        eqn.source = eqn.source + source

        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced3Model(n_cells={self._mesh.n_cells})"
