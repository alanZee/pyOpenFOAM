"""
Enhanced LES models v5 — dynamic procedure and anisotropic minimum-dissipation.

Extends LES model family with:

- Dynamic Smagorinsky with Lagrangian averaging for stable coefficient estimation
- Anisotropic Minimum Dissipation (AMD) model (Rozema et al., 2015) that
  naturally handles anisotropy and wall-bounded flows without damping

Usage::

    from pyfoam.turbulence.les_model_enhanced_5 import DynamicLagrangianSGS, AMDModel

    model = AMDModel(mesh, U, phi)
    model.correct()
    nut = model.nut()
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc

from .les_model import LESModel

__all__ = ["DynamicLagrangianSGS", "AMDModel"]


class DynamicLagrangianSGS(LESModel):
    """Dynamic Smagorinsky with Lagrangian averaging (Meneveau et al., 1996).

    Uses Lagrangian path averaging of the dynamic coefficient to avoid
    the numerical instability of the standard dynamic procedure.

    The coefficient C_s is computed by following fluid particle paths
    and averaging the Germano identity numerator/denominator:

        C_s^2(x,t) = L_M_ij * M_ij (averaged) / M_ij * M_ij (averaged)

    where averaging is along Lagrangian trajectories via exponential
    weighting with a time scale T_L.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field ``(n_faces,)``.
    C_min : float
        Minimum Smagorinsky coefficient. Default 0.0.
    C_max : float
        Maximum Smagorinsky coefficient. Default 0.25.
    T_L : float
        Lagrangian averaging time scale (s). Default 1.0.
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        C_min: float = 0.0,
        C_max: float = 0.25,
        T_L: float = 1.0,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._C_min = C_min
        self._C_max = C_max
        self._T_L = max(T_L, 1e-10)

        n_cells = mesh.n_cells
        # Lagrangian averaging state
        self._L_avg = torch.zeros(n_cells, dtype=U.dtype)
        self._M_avg = torch.zeros(n_cells, dtype=U.dtype)
        self._Cs2 = torch.full((n_cells,), 0.01, dtype=U.dtype)

    @property
    def Cs(self) -> torch.Tensor:
        """Local Smagorinsky coefficient ``(n_cells,)``."""
        return self._Cs2.clamp(min=self._C_min ** 2, max=self._C_max ** 2).sqrt()

    def _compute_Cs2_dynamic(self) -> None:
        """Compute dynamic C_s^2 using Lagrangian averaging.

        1. Compute L_ij and M_ij at test filter level
        2. Update Lagrangian averages with exponential weighting
        3. Cs^2 = <L_ij M_ij> / <M_ij M_ij>
        """
        if self._grad_U is None:
            return

        n_cells = self._mesh.n_cells
        delta = self._delta
        alpha = 2.0  # Test-to-grid filter ratio

        # Strain rate magnitude at grid level
        S = self._S
        if S is None:
            return
        S_mag = self._mag_S
        if S_mag is None:
            return

        # Approximate test-filter strain using spatial averaging
        # (In a real implementation, this would use a proper test filter)
        S_test_mag = S_mag  # Simplified: assume test filter ~ grid filter

        # Germano identity approximate: L_ij M_ij ~ delta^2 * (S_test^2 - alpha^2 * S^2)
        L_M = delta.pow(2) * (S_test_mag.pow(2) - alpha.pow(2) * S_mag.pow(2))
        M_M = delta.pow(2) * alpha.pow(2) * S_mag.pow(2) * (alpha.pow(2) - 1.0)
        M_M = M_M.clamp(min=1e-30)

        # Lagrangian averaging (simplified: exponential moving average)
        dt_eff = 0.01  # Effective time step
        weight = dt_eff / self._T_L
        weight = weight.clamp(min=0.0, max=1.0)

        self._L_avg = (1.0 - weight) * self._L_avg + weight * L_M
        self._M_avg = (1.0 - weight) * self._M_avg + weight * M_M

        Cs2 = self._L_avg / self._M_avg.clamp(min=1e-30)
        self._Cs2 = Cs2.clamp(min=self._C_min ** 2, max=self._C_max ** 2)

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity using dynamic Lagrangian procedure.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")

        S_mag = self._mag_S
        if S_mag is None:
            raise RuntimeError("correct() must be called before nut()")
        delta = self._delta

        Cs2 = self._Cs2.clamp(min=0.0)
        return (Cs2 * delta.pow(2) * S_mag).clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        """SGS kinetic energy estimate."""
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        C_I = 0.09
        return nut.pow(2) / (C_I * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        """SGS dissipation rate."""
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        return nut * 2.0 * S_ij_S_ij

    def correct(self) -> None:
        """Update model: compute gradients, strain rate, and dynamic Cs."""
        self._compute_gradients()
        self._compute_Cs2_dynamic()

    def __repr__(self) -> str:
        Cs_mean = self._Cs2.mean().sqrt().item()
        return (
            f"DynamicLagrangianSGS(Cs_mean={Cs_mean:.4f}, "
            f"n_cells={self._mesh.n_cells})"
        )


class AMDModel(LESModel):
    """Anisotropic Minimum Dissipation model (Rozema et al., 2015).

    The AMD model computes SGS viscosity as:

        nu_sgs = -C_AMD * Delta^2 * (dUi/dxj * dUk/dxl * S_jk) / (dUm/dxn * dUm/dxn)

    where the numerator is a production-like term and the denominator
    is a dissipation-like term, both evaluated at the grid level.

    This is a simplified form that naturally handles:
    - Wall-bounded flows (no wall damping needed)
    - Anisotropic grids
    - Laminar and transitional flows (nu_sgs -> 0 in pure rotation)

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field ``(n_faces,)``.
    C_AMD : float
        AMD model constant. Default 0.3 (recommended for isotropic grids).
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        C_AMD: float = 0.3,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._C_AMD = C_AMD

    @property
    def C_AMD(self) -> float:
        """AMD model constant."""
        return self._C_AMD

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity using AMD model.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")

        g = self._grad_U  # (n_cells, 3, 3) -- dU_i/dx_j
        delta = self._delta  # (n_cells,)

        # Strain rate tensor
        S = 0.5 * (g + g.transpose(-1, -2))

        # Production-like numerator: g_ij * g_kl * S_jk * delta_il
        # Simplified: Tr(g^T @ S @ g) per cell
        # = sum_i sum_j (g_ik * S_kj * g_ij)
        gS = torch.matmul(g, S)  # (n_cells, 3, 3)
        numerator = (g * gS).sum(dim=(1, 2))  # (n_cells,)

        # Dissipation denominator: g_ij * g_ij (Frobenius norm squared)
        denominator = (g * g).sum(dim=(1, 2)).clamp(min=1e-30)  # (n_cells,)

        # nu_sgs = -C * Delta^2 * numerator / denominator (clamped >= 0)
        nut = -self._C_AMD * delta.pow(2) * numerator / denominator

        return nut.clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        """SGS kinetic energy estimate."""
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        C_I = 0.09
        return nut.pow(2) / (C_I * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        """SGS dissipation rate."""
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        return nut * 2.0 * S_ij_S_ij

    def correct(self) -> None:
        """Update model: compute gradients and strain rate."""
        self._compute_gradients()

    def __repr__(self) -> str:
        return f"AMDModel(C_AMD={self._C_AMD}, n_cells={self._mesh.n_cells})"
