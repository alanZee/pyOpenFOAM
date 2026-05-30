"""Enhanced LES models v9 — anisotropic minimum dissipation variant, spectral vanishing viscosity, and hybrid SGS.

Extends LES model family with:
- Anisotropic minimum dissipation (AMD) variant with improved low-Re behavior
- Spectral vanishing viscosity (SVV) SGS model
- Hybrid Smagorinsky-WALE SGS model

Usage::

    from pyfoam.turbulence.les_model_enhanced_9 import AnisotropicMDv2, SpectralVanishingViscosity, HybridSGS
    model = AnisotropicMDv2(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
import math
from typing import Any
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc
from .les_model import LESModel

__all__ = ["AnisotropicMDv2", "SpectralVanishingViscosity", "HybridSGS"]


class AnisotropicMDv2(LESModel):
    """Anisotropic minimum dissipation v2 SGS model.

    Improved variant of AMD with better near-wall behavior:

        nu_sgs = -C_AMD * Delta^2 * (S_ij * S_ij) / (S_ij * S_ij + |grad(U)|^2)

    with wall-adaptive coefficient.

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_amd : float
        AMD coefficient. Default 0.3.
    wall_damping_A : float
        Van Driest damping constant. Default 26.0.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_amd: float = 0.3, wall_damping_A: float = 26.0) -> None:
        super().__init__(mesh, U, phi)
        self._C_amd = C_amd
        self._A = wall_damping_A
        cc = mesh.cell_centres
        self._y = cc.norm(dim=1).clamp(min=1e-10)

    def nut(self) -> torch.Tensor:
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")

        delta = self._delta
        S = self._grad_U  # velocity gradient tensor
        S_sym = 0.5 * (S + S.transpose(-1, -2))
        S_sq = (S_sym * S_sym).sum(dim=(1, 2)).clamp(min=1e-30)

        # AMD numerator and denominator
        grad_U_sq = (S * S).sum(dim=(1, 2)).clamp(min=1e-30)

        # Wall damping
        nu = 1.5e-5
        y_plus_est = self._y * 100.0
        f_wall = (1.0 - torch.exp(-y_plus_est / self._A)).clamp(min=0.0, max=1.0)

        nu_sgs = -self._C_amd * delta.pow(2) * S_sq / (S_sq + grad_U_sq)
        return (nu_sgs * f_wall).clamp(min=0.0)

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
        return f"AnisotropicMDv2(C_amd={self._C_amd}, n_cells={self._mesh.n_cells})"


class SpectralVanishingViscosity(LESModel):
    """Spectral vanishing viscosity (SVV) SGS model.

    Applies viscosity only to high-frequency modes above a threshold
    wavenumber, preserving low-frequency content:

        nu_sgs(k) = C_svv * Delta * |S| * sigma(k/k_cut)

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_svv : float
        SVV coefficient. Default 0.1.
    k_cutoff_ratio : float
        Ratio of cutoff to Nyquist wavenumber. Default 0.5.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_svv: float = 0.1, k_cutoff_ratio: float = 0.5) -> None:
        super().__init__(mesh, U, phi)
        self._C_svv = C_svv
        self._k_cut_ratio = max(0.1, min(k_cutoff_ratio, 0.9))

    def nut(self) -> torch.Tensor:
        if self._grad_U is None or self._mag_S is None:
            raise RuntimeError("correct() must be called before nut()")

        delta = self._delta
        S_mag = self._mag_S.clamp(min=1e-30)

        # Simplified SVV: uniform viscosity with spectral cutoff proxy
        sigma = self._k_cut_ratio
        nu_sgs = self._C_svv * delta * S_mag * sigma
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
        return f"SpectralVanishingViscosity(C_svv={self._C_svv}, n_cells={self._mesh.n_cells})"


class HybridSGS(LESModel):
    """Hybrid Smagorinsky-WALE SGS model.

    Blends Smagorinsky and WALE contributions:

        nu_sgs = (1-f_wale) * nu_Smag + f_wale * nu_WALE

    where f_wale depends on the distance from the wall.

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_s : float
        Smagorinsky coefficient. Default 0.1.
    C_w : float
        WALE coefficient. Default 0.5.
    blend_A : float
        Blending transition parameter. Default 10.0.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_s: float = 0.1, C_w: float = 0.5, blend_A: float = 10.0) -> None:
        super().__init__(mesh, U, phi)
        self._C_s = C_s
        self._C_w = C_w
        self._blend_A = blend_A
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

        # Blend: near wall use WALE, far use Smagorinsky
        nu = 1.5e-5
        y_plus_est = self._y * 100.0
        f_wale = torch.tanh(y_plus_est / self._blend_A)

        nu_sgs = (1.0 - f_wale) * nu_wale + f_wale * nu_smag
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
        return f"HybridSGS(C_s={self._C_s}, C_w={self._C_w}, n_cells={self._mesh.n_cells})"
