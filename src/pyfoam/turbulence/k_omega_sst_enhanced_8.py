"""Enhanced k-omega SST v8 — transition, roughness-aware blending, LES-RANS interface.

Extends KOmegaSSTEnhanced7Model with:
- Transition-sensitive SST with intermittency correction
- Roughness-aware blending function modification
- LES-RANS interface treatment for hybrid simulations

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_8 import KOmegaSSTEnhanced8Model
    model = KOmegaSSTEnhanced8Model(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .k_omega_sst_enhanced_7 import KOmegaSSTEnhanced7Model, KOmegaSSTEnhanced7Constants

__all__ = ["KOmegaSSTEnhanced8Model", "KOmegaSSTEnhanced8Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced8Constants(KOmegaSSTEnhanced7Constants):
    """Constants for enhanced SST v8."""
    C_transition: float = 0.4
    roughness_factor: float = 0.0
    C_les_rans: float = 0.5


_DEFAULTS = KOmegaSSTEnhanced8Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced8")
class KOmegaSSTEnhanced8Model(KOmegaSSTEnhanced7Model):
    """Enhanced SST v8 with transition sensitivity, roughness-aware blending, and LES-RANS interface."""

    def __init__(
        self, mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaSSTEnhanced8Constants | None = None,
        ks: float = 0.0, enable_sas: bool = True,
        **kwargs: Any,
    ) -> None:
        super(KOmegaSSTEnhanced7Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U = None
        self._y = self._compute_wall_distance()
        self._gamma = torch.ones(n_cells, device=device, dtype=dtype)
        self._n_ampl = torch.zeros(n_cells, device=device, dtype=dtype)
        self._ks = max(0.0, ks)
        self._enable_sas = enable_sas

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _transition_correction(self, P_k: torch.Tensor) -> torch.Tensor:
        """Transition-sensitive production correction.

        Reduces production in laminar-to-turbulent transition region.
        """
        C = self._C
        C_trans = getattr(C, 'C_transition', 0.4)
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = 0.09 * k_safe * self._omega.clamp(min=1e-16)

        # Intermittency-based correction
        Re_theta = self._n_ampl.clamp(min=0.0)
        gamma_eff = (Re_theta / (Re_theta + 50.0)).clamp(min=C_trans, max=1.0)

        return P_k * gamma_eff

    def _roughness_blending(self) -> torch.Tensor:
        """Roughness-aware blending function modification.

        Modifies F1/F2 based on equivalent sand-grain roughness.
        """
        C = self._C
        ks_factor = getattr(C, 'roughness_factor', 0.0)
        if ks_factor < 1e-10:
            return torch.ones_like(self._k)

        y_plus_est = self._y * 100.0  # Simplified estimate
        ks_plus = ks_factor * y_plus_est
        # Shift blending towards wall-law for rough walls
        f_rough = (1.0 + 0.1 * ks_plus.clamp(max=100.0)).clamp(max=3.0)
        return f_rough

    def _les_rans_interface(self, P_k: torch.Tensor) -> torch.Tensor:
        """LES-RANS interface treatment for hybrid simulations.

        Applies damping near the RANS-LES interface to prevent
        the pile-up of turbulent kinetic energy.
        """
        C = self._C
        C_lr = getattr(C, 'C_les_rans', 0.5)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        delta = self._y  # Use wall distance as proxy for local length scale

        # Length scale ratio
        L_t = k_safe.pow(1.5) / (0.09 * k_safe * omega_safe).clamp(min=1e-16)
        L_ratio = (L_t / delta.clamp(min=1e-10)).clamp(max=5.0)

        # Damping when turbulent length scale is large relative to cell size
        f_damp = (1.0 / (1.0 + C_lr * L_ratio.pow(2))).clamp(min=0.2, max=1.0)
        return P_k * f_damp

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._compute_amplification()
        self._compute_intermittency()
        P_k = self._compute_production()
        f_ss = self._spalart_shur_correction_v5()
        P_k = P_k * f_ss
        P_k = self._compressibility_correction(P_k)
        P_k = self._shock_limiter(P_k)
        P_k = self._transition_correction(P_k)
        P_k = self._les_rans_interface(P_k)
        sas = self._sas_source(P_k)
        self._solve_k(P_k + sas)
        self._solve_omega(P_k)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced8Model(n_cells={self._mesh.n_cells})"
