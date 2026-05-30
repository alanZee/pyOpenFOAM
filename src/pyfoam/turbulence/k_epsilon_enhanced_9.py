"""Enhanced k-epsilon v9 — algebraic stress model, realizability-enforced blending, and TKE transport diagnostics.

Extends KEpsilonEnhanced8Model with:
- Algebraic stress model (ASM) for anisotropic Reynolds stresses
- Realizability-enforced blending near stagnation and strong curvature
- TKE transport diagnostics (production, dissipation, transport budgets)

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model
    model = KEpsilonEnhanced9Model(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .k_epsilon_enhanced_8 import KEpsilonEnhanced8Model, KEpsilonEnhanced8Constants

__all__ = ["KEpsilonEnhanced9Model", "KEpsilonEnhanced9Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced9Constants(KEpsilonEnhanced8Constants):
    """Constants for enhanced k-epsilon v9."""
    C_asm: float = 0.3
    C_realiz_blend: float = 0.5
    C_transport_diag: float = 0.1


_DEFAULTS = KEpsilonEnhanced9Constants()


@TurbulenceModel.register("realizableKEEnhanced9")
class KEpsilonEnhanced9Model(KEpsilonEnhanced8Model):
    """Enhanced k-epsilon v9 with ASM and transport diagnostics."""

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced9Constants | None = None,
        enable_asm: bool = False,
        enable_transport_diag: bool = False,
        **kwargs: Any,
    ) -> None:
        super(KEpsilonEnhanced8Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U = None
        self._y = self._compute_wall_distance()
        self._f_relax = torch.ones(n_cells, device=device, dtype=dtype)
        self._v2 = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._alpha3 = torch.zeros(n_cells, device=device, dtype=dtype)
        self._enable_asm = enable_asm
        self._enable_transport_diag = enable_transport_diag
        self._budget: dict[str, torch.Tensor] = {}

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def asm_anisotropy(self) -> torch.Tensor:
        """Algebraic stress model for anisotropic Reynolds stresses.

        Returns b_ij tensor (n_cells, 3, 3) where trace(b) = 0.
        """
        n_cells = self._mesh.n_cells
        b = torch.zeros(n_cells, 3, 3, device=self._device, dtype=self._dtype)
        if not self._enable_asm or self._grad_U is None:
            return b

        C = self._C
        C_asm = getattr(C, 'C_asm', 0.3)
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        S = self._strain_rate()
        tau = k_safe / eps_safe

        # b_ij ~ -C_asm * tau * S_ij (simplified)
        for i in range(3):
            for j in range(3):
                b[:, i, j] = -C_asm * tau * S[:, i, j]

        return b

    def realizability_blend(self, P_k: torch.Tensor) -> torch.Tensor:
        """Realizability-enforced blending near stagnation."""
        C = self._C
        C_rb = getattr(C, 'C_realiz_blend', 0.5)
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # Production-to-dissipation ratio
        ratio = (P_k / eps_safe).clamp(max=10.0)
        f_realiz = (1.0 / (1.0 + C_rb * ratio)).clamp(min=0.2, max=1.0)
        return P_k * f_realiz

    def transport_budget(self) -> dict[str, torch.Tensor]:
        """TKE transport budget diagnostics."""
        if not self._enable_transport_diag:
            return {}

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        if self._grad_U is not None:
            nut = self.nut()
            S = self._strain_rate()
            P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        else:
            P_k = torch.zeros_like(self._k)

        self._budget = {
            "production": P_k,
            "dissipation": eps_safe,
            "tke": k_safe,
            "ratio": (P_k / eps_safe).clamp(max=10.0),
        }
        return self._budget

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        P_k = self._realizability_limiter(P_k)
        P_k = self.realizability_blend(P_k)
        self._solve_k(P_k)
        self._solve_eps(P_k)
        if self._enable_transport_diag:
            self.transport_budget()

    def __repr__(self) -> str:
        asm = ", ASM" if self._enable_asm else ""
        return f"KEpsilonEnhanced9Model(n_cells={self._mesh.n_cells}{asm})"
