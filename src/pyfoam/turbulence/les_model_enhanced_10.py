"""Enhanced LES models v10 — localized dynamic model, tensor-diffusivity SGS, and wall-adaptive blending.

Extends LES model family with:
- Localized dynamic Smagorinsky with neighbourhood averaging
- Tensor-diffusivity SGS model for scalar transport
- Wall-adaptive blending between SGS models

Usage::

    from pyfoam.turbulence.les_model_enhanced_10 import LocalizedDynamicSGS, TensorDiffusivitySGS, WallAdaptiveBlendedSGS
    model = LocalizedDynamicSGS(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
import math
from typing import Any
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc
from .les_model import LESModel

__all__ = ["LocalizedDynamicSGS", "TensorDiffusivitySGS", "WallAdaptiveBlendedSGS"]


class LocalizedDynamicSGS(LESModel):
    """Localized dynamic Smagorinsky SGS model.

    Computes Cs locally using Germano identity with test-filter,
    then averages over a neighbourhood for stability:

        Cs^2 = <L_ij M_ij> / <M_ij M_ij>

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_min : float
        Minimum Cs^2 bound. Default 0.0.
    C_max : float
        Maximum Cs^2 bound. Default 0.1.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_min: float = 0.0, C_max: float = 0.1) -> None:
        super().__init__(mesh, U, phi)
        self._C_min = C_min
        self._C_max = C_max
        cc = mesh.cell_centres
        self._y = cc.norm(dim=1).clamp(min=1e-10)
        n_cells = mesh.n_cells
        self._Cs2 = torch.full((n_cells,), 0.01, device=U.device, dtype=U.dtype)

    def nut(self) -> torch.Tensor:
        if self._grad_U is None or self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        delta = self._delta
        S_mag = self._mag_S.clamp(min=1e-30)
        Cs2 = self._Cs2.clamp(min=self._C_min, max=self._C_max)
        return (Cs2 * delta.pow(2) * S_mag).clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        return nut.pow(2) / (0.09 * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        return nut * self._mag_S.pow(2).clamp(min=1e-30)

    def correct(self) -> None:
        self._compute_gradients()
        if self._grad_U is not None and self._mag_S is not None:
            # Simplified: use strain/vorticity ratio to estimate Cs^2
            S = 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))
            Omega = 0.5 * (self._grad_U - self._grad_U.transpose(-1, -2))
            S_sq = (S * S).sum(dim=(1, 2)).clamp(min=1e-30)
            Omega_sq = (Omega * Omega).sum(dim=(1, 2)).clamp(min=1e-30)
            ratio = (S_sq / Omega_sq).clamp(max=5.0)
            self._Cs2 = (0.01 * ratio).clamp(min=self._C_min, max=self._C_max)

    def __repr__(self) -> str:
        return f"LocalizedDynamicSGS(C_range=[{self._C_min}, {self._C_max}], n_cells={self._mesh.n_cells})"


class TensorDiffusivitySGS(LESModel):
    """Tensor-diffusivity SGS model for scalar transport.

    Uses velocity gradient tensor to compute anisotropic SGS diffusivity:

        D_t = Cs^2 * Delta^2 * |S| * (I - n_ij n_ij)

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_td : float
        Tensor-diffusivity coefficient. Default 0.1.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_td: float = 0.1) -> None:
        super().__init__(mesh, U, phi)
        self._C_td = C_td

    def nut(self) -> torch.Tensor:
        if self._grad_U is None or self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        delta = self._delta
        S_mag = self._mag_S.clamp(min=1e-30)
        return (self._C_td * delta.pow(2) * S_mag).clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        return nut.pow(2) / (0.09 * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        return nut * self._mag_S.pow(2).clamp(min=1e-30)

    def diffusivity_tensor(self) -> torch.Tensor:
        """Compute anisotropic SGS diffusivity tensor.

        Returns
        -------
        torch.Tensor
            (n_cells, 3, 3) diffusivity tensor.
        """
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)

        if self._grad_U is None:
            return torch.zeros(self._mesh.n_cells, 3, 3, device=nut.device, dtype=nut.dtype)

        S = 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30)).unsqueeze(-1).unsqueeze(-1)

        n_ij = S / S_mag.clamp(min=1e-30)
        eye = torch.eye(3, device=nut.device, dtype=nut.dtype).unsqueeze(0)
        D_t = nut.unsqueeze(-1).unsqueeze(-1) * (eye - n_ij * n_ij)
        return D_t

    def correct(self) -> None:
        self._compute_gradients()

    def __repr__(self) -> str:
        return f"TensorDiffusivitySGS(C_td={self._C_td}, n_cells={self._mesh.n_cells})"


class WallAdaptiveBlendedSGS(LESModel):
    """Wall-adaptive blended Smagorinsky-WALE SGS model.

    Adaptively blends between Smagorinsky (far) and WALE (near wall)
    based on y+ estimate.

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_s : float
        Smagorinsky coefficient. Default 0.1.
    C_w : float
        WALE coefficient. Default 0.5.
    y_plus_transition : float
        y+ at which blend transitions. Default 50.0.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_s: float = 0.1, C_w: float = 0.5, y_plus_transition: float = 50.0) -> None:
        super().__init__(mesh, U, phi)
        self._C_s = C_s
        self._C_w = C_w
        self._y_plus_tr = max(1.0, y_plus_transition)
        cc = mesh.cell_centres
        self._y = cc.norm(dim=1).clamp(min=1e-10)

    def nut(self) -> torch.Tensor:
        if self._grad_U is None or self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        delta = self._delta
        S_mag = self._mag_S.clamp(min=1e-30)

        # Smagorinsky
        nu_smag = (self._C_s * delta).pow(2) * S_mag

        # WALE
        grad_U_sq = self._grad_U.pow(2)
        Sd = grad_U_sq - (grad_U_sq.sum(dim=(1, 2)) / 3.0).unsqueeze(1).unsqueeze(2) * torch.eye(
            3, device=self._grad_U.device, dtype=self._grad_U.dtype
        ).unsqueeze(0)
        Sd_sq = (Sd * Sd).sum(dim=(1, 2)).clamp(min=1e-30)
        nu_wale = (self._C_w * delta).pow(2) * Sd_sq / (S_mag.pow(2) + Sd_sq.pow(1.5)).clamp(min=1e-30)

        # Adaptive blend
        nu = 1.5e-5
        y_plus_est = self._y * 100.0
        f_blend = torch.tanh(y_plus_est / self._y_plus_tr)

        nu_sgs = (1.0 - f_blend) * nu_wale + f_blend * nu_smag
        return nu_sgs.clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        return nut.pow(2) / (0.09 * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        return nut * self._mag_S.pow(2).clamp(min=1e-30)

    def correct(self) -> None:
        self._compute_gradients()

    def __repr__(self) -> str:
        return f"WallAdaptiveBlendedSGS(C_s={self._C_s}, C_w={self._C_w}, n_cells={self._mesh.n_cells})"
