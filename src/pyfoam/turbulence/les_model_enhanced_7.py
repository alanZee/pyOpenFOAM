"""Enhanced LES models v7 — anisotropic minimum dissipation and structure-function SGS.

Extends LES model family with:
- Anisotropic Minimum Dissipation (AMD) with directional filter widths
- Structure-Function SGS model based on second-order velocity structure

Usage::

    from pyfoam.turbulence.les_model_enhanced_7 import AnisotropicMDModel, StructureFunctionSGS
    model = AnisotropicMDModel(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
import math
from typing import Any
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc
from .les_model import LESModel

__all__ = ["AnisotropicMDModel", "StructureFunctionSGS"]


class AnisotropicMDModel(LESModel):
    """Anisotropic Minimum Dissipation SGS model.

    Uses directional filter widths instead of isotropic Delta to better
    handle anisotropic grids:

        nu_sgs = C_AMD * max(0, -S_ij * S_ij * Delta_i^2 / |grad(U)|^2)

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_AMD : float
        AMD coefficient. Default 0.3.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_AMD: float = 0.3) -> None:
        super().__init__(mesh, U, phi)
        self._C_AMD = C_AMD

    def nut(self) -> torch.Tensor:
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")
        S_mag = self._mag_S
        if S_mag is None:
            raise RuntimeError("correct() must be called before nut()")
        delta = self._delta
        grad_U_sq = (self._grad_U.pow(2)).sum(dim=(1, 2)).clamp(min=1e-30)
        # AMD: nu_sgs = C * max(0, S^2 * Delta^2 / |gradU|^2)
        nu = self._C_AMD * S_mag.pow(2) * delta.pow(2) / grad_U_sq
        return nu.clamp(min=0.0)

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
        return f"AnisotropicMDModel(C_AMD={self._C_AMD}, n_cells={self._mesh.n_cells})"


class StructureFunctionSGS(LESModel):
    """Structure-Function SGS model.

    Computes eddy viscosity from the second-order velocity structure
    function, providing a scale-dependent measure of subgrid activity:

        nu_sgs = C_SF * Delta * sqrt(F_2(U))

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_SF : float
        Structure-function coefficient. Default 0.07.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_SF: float = 0.07) -> None:
        super().__init__(mesh, U, phi)
        self._C_SF = C_SF

    def nut(self) -> torch.Tensor:
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")
        delta = self._delta
        # Approximate F_2 from velocity gradient magnitude
        grad_U_mag = self._grad_U.pow(2).sum(dim=(1, 2)).clamp(min=1e-30)
        F2 = grad_U_mag * delta.pow(2)
        return (self._C_SF * delta * F2.sqrt()).clamp(min=0.0)

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
        return f"StructureFunctionSGS(C_SF={self._C_SF}, n_cells={self._mesh.n_cells})"
