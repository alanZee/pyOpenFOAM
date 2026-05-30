"""
Enhanced k-omega SST turbulence model v5 — improved transition and rotation-curvature.

Extends :class:`~pyfoam.turbulence.k_omega_sst_enhanced_4.KOmegaSSTEnhanced4Model`
with:

- Second-mode transition coupling (amplification factor transport)
- Rotation-curvature correction with Smagorinsky-like strain coupling
- Adaptive sigma_k1/sigma_k2 based on local flow topology

Usage::

    from pyfoam.turbulence.k_omega_sst_enhanced_5 import KOmegaSSTEnhanced5Model

    model = KOmegaSSTEnhanced5Model(mesh, U, phi)
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
from .k_omega_sst_enhanced_4 import KOmegaSSTEnhanced4Model, KOmegaSSTEnhanced4Constants

__all__ = ["KOmegaSSTEnhanced5Model", "KOmegaSSTEnhanced5Constants"]


@dataclass(frozen=True)
class KOmegaSSTEnhanced5Constants(KOmegaSSTEnhanced4Constants):
    """Constants for enhanced SST v5.

    Extends parent constants with:
        C_amp: Amplification factor transport coefficient.
        C_rc2: Second rotation-curvature coefficient.
        sigma_adapt: Adaptive sigma blending width.
    """

    C_amp: float = 0.03
    C_rc2: float = 0.1
    sigma_adapt: float = 0.5


_DEFAULTS = KOmegaSSTEnhanced5Constants()


@TurbulenceModel.register("kOmegaSST2003Enhanced5")
class KOmegaSSTEnhanced5Model(KOmegaSSTEnhanced4Model):
    """Enhanced SST v5 with amplification factor and improved curvature.

    Features:
    - Amplification factor transport for second-mode transition
    - Improved rotation-curvature correction with dual parameters
    - Adaptive sigma_k based on local strain-to-vorticity ratio

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    constants : KOmegaSSTEnhanced5Constants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KOmegaSSTEnhanced5Constants | None = None,
        **kwargs: Any,
    ) -> None:
        # Call TurbulenceModel.__init__ (skip parent chain)
        super(KOmegaSSTEnhanced4Model, self).__init__(mesh, U, phi)

        self._C = constants or _DEFAULTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U: torch.Tensor | None = None
        self._y = self._compute_wall_distance()

        # Intermittency from v4
        self._gamma = torch.ones(n_cells, device=device, dtype=dtype)

        # Amplification factor
        self._n_ampl = torch.zeros(n_cells, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Wall distance
    # ------------------------------------------------------------------

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute simplified wall distance."""
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Amplification factor transport
    # ------------------------------------------------------------------

    def _compute_amplification(self) -> None:
        """Update amplification factor for second-mode transition.

        dn/dt = C_amp * (1 - gamma) * Ret * omega

        n_ampl tracks the amplification of disturbances in the
        transitional boundary layer.
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)

        Ret = k_safe / (nu * omega_safe)
        C_amp = getattr(C, 'C_amp', 0.03)

        dn = C_amp * (1.0 - self._gamma.clamp(max=1.0)) * Ret * omega_safe
        self._n_ampl = (self._n_ampl + dn * 0.01).clamp(min=0.0, max=20.0)

    # ------------------------------------------------------------------
    # Intermittency update (enhanced from v4)
    # ------------------------------------------------------------------

    def _compute_intermittency(self) -> None:
        """Update intermittency with amplification factor coupling.

        gamma = 1 - exp(-C_turb_trans * Ret - C_amp * n_ampl)
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)
        nu = max(self._nu, 1e-30)

        Ret = k_safe / (nu * omega_safe)
        C_t = getattr(C, 'C_turb_trans', 0.6)
        C_amp = getattr(C, 'C_amp', 0.03)

        gamma = 1.0 - torch.exp(-C_t * Ret - C_amp * self._n_ampl)
        self._gamma = gamma.clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Improved rotation-curvature correction
    # ------------------------------------------------------------------

    def _spalart_shur_correction_v5(self) -> torch.Tensor:
        """Improved Spalart-Shur with dual parameter rotation-curvature.

        f_rc = max(0, 1 + C_rc * (r_star - 1)) / max(0, 1 + C_rc2 * (1/r_star - 1))

        Parameters C_rc and C_rc2 allow independent control of rotation
        and curvature effects.
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
        C_rc2 = getattr(C, 'C_rc2', 0.1)

        numerator = 1.0 + C_rc * (r_star - 1.0).clamp(min=0.0)
        denominator = 1.0 + C_rc2 * (1.0 / r_star.clamp(min=1e-10) - 1.0).clamp(min=0.0)

        return (numerator / denominator.clamp(min=0.1)).clamp(min=0.2, max=5.0)

    # ------------------------------------------------------------------
    # Adaptive sigma_k
    # ------------------------------------------------------------------

    def _adaptive_sigma_k(self, F1: torch.Tensor | None = None) -> float:
        """Adaptive sigma_k based on local flow topology.

        sigma_k = sigma_k1 * F1 + sigma_k2 * (1 - F1)

        Parameters
        ----------
        F1 : torch.Tensor or None
            SST blending function (unused, accepted for API compatibility
            with parent class).
        """
        C = self._C
        sigma_k1 = getattr(C, 'sigma_k1', 0.85)
        sigma_k2 = getattr(C, 'sigma_k2', 1.0)

        # Simplified: use mean strain-to-vorticity ratio as proxy for F1
        if self._grad_U is not None:
            S = self._strain_rate()
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            Omega_mag = torch.sqrt(2.0 * (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30))
            ratio = (S_mag.mean() / Omega_mag.mean().clamp(min=1e-10)).clamp(max=2.0)
            F1_approx = 1.0 / (1.0 + ratio)
        else:
            F1_approx = 0.5

        return F1_approx * sigma_k1 + (1.0 - F1_approx) * sigma_k2

    # ------------------------------------------------------------------
    # Override nut with intermittency
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity with intermittency correction."""
        nut_sst = super(KOmegaSSTEnhanced4Model, self).nut()
        return self._gamma * nut_sst

    # ------------------------------------------------------------------
    # Override correct
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the enhanced SST v5 model."""
        grad_U = torch.zeros(
            self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(
                self._U[:, i], "Gauss linear", mesh=self._mesh
            )
        self._grad_U = grad_U

        # Update transition model
        self._compute_amplification()
        self._compute_intermittency()

        P_k = self._compute_production()

        # Improved curvature correction
        f_ss = self._spalart_shur_correction_v5()
        P_k = P_k * f_ss

        self._solve_k(P_k)
        self._solve_omega(P_k)

    def __repr__(self) -> str:
        return f"KOmegaSSTEnhanced5Model(n_cells={self._mesh.n_cells})"
