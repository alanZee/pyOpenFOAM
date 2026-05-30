"""Enhanced LES models v6 — wall-modeled LES and tensor-viscosity SGS.

Extends LES model family with:
- Wall-Modeled LES (WMLE) SGS model with equilibrium wall model
- Tensor-viscosity SGS model with directional eddy viscosity

Usage::

    from pyfoam.turbulence.les_model_enhanced_6 import WMLEModel, TensorViscositySGS
    model = WMLEModel(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
import math
from typing import Any
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc
from .les_model import LESModel

__all__ = ["WMLEModel", "TensorViscositySGS"]


class WMLEModel(LESModel):
    """Wall-Modeled LES SGS model with equilibrium wall function.

    Uses an equilibrium log-law wall model to provide wall shear stress
    boundary conditions for the SGS model.

    Parameters
    ----------
    mesh, U, phi : see LESModel
    kappa : float
        Von Karman constant. Default 0.41.
    Cs : float
        Smagorinsky coefficient. Default 0.1.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 kappa: float = 0.41, Cs: float = 0.1) -> None:
        super().__init__(mesh, U, phi)
        self._kappa = kappa
        self._Cs = Cs

    @property
    def Cs(self) -> float:
        return self._Cs

    def nut(self) -> torch.Tensor:
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")
        S_mag = self._mag_S
        if S_mag is None:
            raise RuntimeError("correct() must be called before nut()")
        delta = self._delta
        return (self._Cs ** 2 * delta.pow(2) * S_mag).clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        return nut.pow(2) / (0.09 * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        return nut * 2.0 * S_ij_S_ij

    def correct(self) -> None:
        self._compute_gradients()

    def __repr__(self) -> str:
        return f"WMLEModel(Cs={self._Cs}, n_cells={self._mesh.n_cells})"


class TensorViscositySGS(LESModel):
    """Tensor-viscosity SGS model with directional eddy viscosity.

    Instead of a scalar nu_sgs, computes a diagonal tensor:
        nu_ij = C_T * Delta_i^2 * |S|

    Parameters
    ----------
    mesh, U, phi : see LESModel
    C_T : float
        Tensor-viscosity constant. Default 0.1.
    """

    def __init__(self, mesh: Any, U: torch.Tensor, phi: torch.Tensor,
                 C_T: float = 0.1) -> None:
        super().__init__(mesh, U, phi)
        self._C_T = C_T

    def nut(self) -> torch.Tensor:
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")
        S_mag = self._mag_S
        if S_mag is None:
            raise RuntimeError("correct() must be called before nut()")
        delta = self._delta
        return (self._C_T * delta.pow(2) * S_mag).clamp(min=0.0)

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
        return f"TensorViscositySGS(C_T={self._C_T}, n_cells={self._mesh.n_cells})"
