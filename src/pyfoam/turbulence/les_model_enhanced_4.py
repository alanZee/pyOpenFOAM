"""
Enhanced LES models v4 — improved Vreman and Sigma SGS models.

Extends LES model family with:

- Vreman SGS model: eddy viscosity model that is stable in laminar and
  transitional flows (zero viscosity for uniform strain)
- Sigma SGS model: based on singular values of velocity gradient tensor
  (Nicoud et al., 2011), better in rotating flows than Smagorinsky/WALE

Usage::

    from pyfoam.turbulence.les_model_enhanced_4 import VremanModel, SigmaModel

    model = VremanModel(mesh, U, phi)
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

__all__ = ["VremanModel", "SigmaModel"]


class VremanModel(LESModel):
    """Vreman SGS model (Vreman, 2004).

    The Vreman model computes SGS viscosity as:

        nu_sgs = C_v * sqrt(B_beta / (alpha_ij * alpha_ij))

    where:
        alpha_ij = d(U_i)/d(x_j)
        B_beta = beta_11 * beta_22 - beta_12^2 + beta_11 * beta_33 - beta_13^2 + beta_22 * beta_33 - beta_23^2
        beta_ij = Delta_k^2 * alpha_ki * alpha_kj

    Advantages:
    - Naturally zero in laminar regions (no wall damping needed)
    - Galilean invariant
    - Smaller constant than Smagorinsky

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field ``(n_faces,)``.
    Cv : float
        Vreman constant. Default 0.07 (theoretical: 2.5*C_s^2).
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cv: float = 0.07,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cv = Cv
        self._alpha: torch.Tensor | None = None

    @property
    def Cv(self) -> float:
        """Vreman constant."""
        return self._Cv

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity using Vreman model.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._grad_U is None:
            raise RuntimeError("correct() must be called before nut()")

        alpha = self._grad_U  # (n_cells, 3, 3) -- alpha_ij = dU_i/dx_j
        delta = self._delta  # (n_cells,)

        # beta_ij = Delta^2 * alpha_ki * alpha_kj
        # beta_ij = Delta^2 * (alpha^T @ alpha)_ij
        alpha_T = alpha.transpose(-1, -2)
        beta = delta.pow(2).unsqueeze(-1).unsqueeze(-1) * torch.matmul(alpha_T, alpha)

        # B_beta = sum of 2x2 principal minors
        b11 = beta[:, 0, 0]
        b22 = beta[:, 1, 1]
        b33 = beta[:, 2, 2]
        b12 = beta[:, 0, 1]
        b13 = beta[:, 0, 2]
        b23 = beta[:, 1, 2]

        B_beta = (b11 * b22 - b12.pow(2)
                  + b11 * b33 - b13.pow(2)
                  + b22 * b33 - b23.pow(2))
        B_beta = B_beta.clamp(min=0.0)

        # alpha_ij * alpha_ij
        alpha_sq = (alpha * alpha).sum(dim=(1, 2)).clamp(min=1e-30)

        # nu_sgs = C_v * sqrt(B_beta / alpha_sq)
        nut = self._Cv * torch.sqrt(B_beta / alpha_sq)

        return nut.clamp(min=0.0)

    def k_sgs(self) -> torch.Tensor:
        """SGS kinetic energy estimate.

        k_sgs = nu_sgs^2 / (C_I * Delta^2)
        """
        nut = self.nut()
        delta_safe = self._delta.clamp(min=1e-10)
        C_I = 0.09
        return nut.pow(2) / (C_I * delta_safe.pow(2))

    def epsilon_sgs(self) -> torch.Tensor:
        """SGS dissipation rate.

        eps_sgs = nu_sgs * |S|^2
        """
        if self._mag_S is None:
            raise RuntimeError("correct() must be called before epsilon_sgs()")
        nut = self.nut()
        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        return nut * 2.0 * S_ij_S_ij

    def correct(self) -> None:
        """Update model: compute gradients and strain rate."""
        self._compute_gradients()

    def __repr__(self) -> str:
        return f"VremanModel(Cv={self._Cv}, n_cells={self._mesh.n_cells})"


class SigmaModel(LESModel):
    """Sigma SGS model (Nicoud et al., 2011).

    The Sigma model uses the singular values of the velocity gradient
    tensor to compute SGS viscosity:

        nu_sgs = (C_sigma * Delta)^2 * D_sigma

    where D_sigma = (sigma_3 * (sigma_1 - sigma_2) * (sigma_2 - sigma_3)) / sigma_1^2

    and sigma_1 >= sigma_2 >= sigma_3 >= 0 are the singular values of g_ij.

    Advantages:
    - Naturally zero in pure shear (2D) and solid rotation
    - Better in rotating flows than Smagorinsky/WALE
    - No wall damping needed

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field ``(n_faces,)``.
    C_sigma : float
        Sigma model constant. Default 1.5 (Nicoud et al. recommendation).
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        C_sigma: float = 1.5,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._C_sigma = C_sigma
        self._sigma_values: torch.Tensor | None = None

    @property
    def C_sigma(self) -> float:
        """Sigma model constant."""
        return self._C_sigma

    @property
    def singular_values(self) -> torch.Tensor | None:
        """Singular values of velocity gradient tensor ``(n_cells, 3)`` or None."""
        return self._sigma_values

    def _compute_sigma(self) -> None:
        """Compute singular values of velocity gradient tensor."""
        g = self._grad_U  # (n_cells, 3, 3)

        # SVD of g
        U_svd, S_svd, V_svd = torch.linalg.svd(g, full_matrices=False)
        # S_svd: (n_cells, 3) -- already sorted descending

        self._sigma_values = S_svd

    def nut(self) -> torch.Tensor:
        """Compute SGS viscosity using Sigma model.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor of SGS viscosity.
        """
        if self._sigma_values is None:
            raise RuntimeError("correct() must be called before nut()")

        sigma = self._sigma_values  # (n_cells, 3)
        s1 = sigma[:, 0].clamp(min=1e-30)
        s2 = sigma[:, 1]
        s3 = sigma[:, 2]

        # D_sigma = sigma_3 * (sigma_1 - sigma_2) * (sigma_2 - sigma_3) / sigma_1^2
        D_sigma = (
            s3 * (s1 - s2).clamp(min=0.0) * (s2 - s3).clamp(min=0.0)
            / s1.pow(2)
        )
        D_sigma = D_sigma.clamp(min=0.0)

        coeff = (self._C_sigma * self._delta).pow(2)
        return (coeff * D_sigma).clamp(min=0.0)

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
        """Update model: compute gradients, strain rate, and SVD."""
        self._compute_gradients()
        self._compute_sigma()

    def __repr__(self) -> str:
        return f"SigmaModel(C_sigma={self._C_sigma}, n_cells={self._mesh.n_cells})"
