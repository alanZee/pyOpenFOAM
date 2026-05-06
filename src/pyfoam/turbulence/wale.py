"""
WALE subgrid-scale model for LES.

The Wall-Adapting Local Eddy-viscosity (WALE) model (Nicoud & Ducros,
1999) computes the SGS turbulent viscosity as:

.. math::

    \\nu_{sgs} = (C_w \\Delta)^2
        \\frac{(S^d_{ij} S^d_{ij})^{3/2}}
             {(S_{ij} S_{ij})^{5/2} + (S^d_{ij} S^d_{ij})^{5/4}}

where:

- :math:`C_w` is the WALE constant (default ≈ 0.325)
- :math:`\\Delta` is the filter width
- :math:`S_{ij} = \\frac{1}{2}(g_{ij} + g_{ji})` is the strain rate tensor
- :math:`g_{ij} = \\partial u_i / \\partial x_j` is the velocity gradient
- :math:`S^d_{ij} = \\frac{1}{2}(g^2_{ij} + g^2_{ji})
  - \\frac{1}{3}\\delta_{ij} g^2_{kk}` is the traceless symmetric
  part of the squared velocity gradient
- :math:`g^2_{ij} = g_{ik} g_{kj}` is the matrix square of the
  velocity gradient

The WALE model naturally recovers the correct near-wall scaling
(:math:`\\nu_{sgs} \\propto y^3`) without requiring explicit wall-damping
functions, making it well suited for complex geometries.

References
----------
Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based
on the square of the velocity gradient tensor. *Flow, Turbulence and
Combustion*, 62(3), 183–200.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .les_model import LESModel

__all__ = ["WALEModel"]

# Default WALE constant
DEFAULT_CW = 0.325


class WALEModel(LESModel):
    """WALE (Wall-Adapting Local Eddy-viscosity) subgrid-scale model.

    The SGS viscosity is computed as:

        ν_sgs = (Cw * Δ)² * (Sd:Sd)^(3/2) /
                ((S:S)^(5/2) + (Sd:Sd)^(5/4))

    where Sd_ij is the traceless symmetric part of g²_ij, the matrix
    square of the velocity gradient.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cw : float, optional
        WALE constant.  Default is 0.325.

    Examples::

        >>> model = WALEModel(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.nut()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cw: float = DEFAULT_CW,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cw = Cw
        self._Sd: torch.Tensor | None = None
        self._mag_Sd_sq: torch.Tensor | None = None

    @property
    def Cw(self) -> float:
        """WALE constant."""
        return self._Cw

    @Cw.setter
    def Cw(self, value: float) -> None:
        self._Cw = value

    def nut(self) -> torch.Tensor:
        """Compute the SGS turbulent viscosity using the WALE formulation.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity.

        Raises:
            RuntimeError: If :meth:`correct` has not been called.
        """
        if self._mag_S is None or self._mag_Sd_sq is None:
            raise RuntimeError(
                "correct() must be called before nut() to compute "
                "the strain rate and Sd tensors"
            )

        Cw_delta = self._Cw * self._delta
        coeff = Cw_delta.pow(2)

        # Numerator: (Sd_ij * Sd_ij)^(3/2)
        # _mag_Sd_sq = Sd_ij * Sd_ij (scalar per cell)
        numerator = self._mag_Sd_sq.pow(1.5)

        # Denominator: (S_ij * S_ij)^(5/2) + (Sd_ij * Sd_ij)^(5/4)
        # mag_S = sqrt(2 * S_ij * S_ij), so S_ij * S_ij = mag_S^2 / 2
        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        denominator = S_ij_S_ij.pow(2.5) + self._mag_Sd_sq.pow(1.25) + 1e-30

        return coeff * numerator / denominator

    def correct(self) -> None:
        """Update the model with the current velocity field.

        Recomputes the velocity gradient, strain rate tensor, and the
        WALE-specific Sd tensor from the current velocity field.
        """
        # Compute velocity gradient and strain rate
        self._compute_gradients()

        # Compute the WALE Sd tensor
        self._compute_sd_tensor()

    def _compute_sd_tensor(self) -> None:
        """Compute the WALE Sd tensor and its squared magnitude.

        The Sd tensor is defined as:

            Sd_ij = 0.5 * (g²_ij + g²_ji) - (1/3) * δ_ij * g²_kk

        where g²_ij = g_ik * g_kj is the matrix square of the velocity
        gradient.

        Stores:
            - _Sd: ``(n_cells, 3, 3)`` the Sd tensor
            - _mag_Sd_sq: ``(n_cells,)`` the scalar Sd_ij * Sd_ij
        """
        g = self._grad_U  # (n_cells, 3, 3)

        # g²_ij = g_ik * g_kj  (matrix product per cell)
        g2 = torch.matmul(g, g)  # (n_cells, 3, 3)

        # Sd_ij = 0.5 * (g²_ij + g²_ji) - (1/3) * δ_ij * g²_kk
        g2_sym = 0.5 * (g2 + g2.transpose(-1, -2))

        # g²_kk = trace of g² per cell
        g2_trace = g2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (n_cells,)

        # Identity tensor
        I = torch.eye(3, dtype=self._dtype, device=self._device)
        I = I.unsqueeze(0)  # (1, 3, 3) — broadcast over cells

        self._Sd = g2_sym - (1.0 / 3.0) * g2_trace.unsqueeze(-1).unsqueeze(-1) * I

        # Sd_ij * Sd_ij (scalar per cell)
        self._mag_Sd_sq = (self._Sd * self._Sd).sum(dim=(-2, -1))

    @property
    def Sd(self) -> torch.Tensor | None:
        """WALE Sd tensor ``(n_cells, 3, 3)`` or ``None``."""
        return self._Sd

    @property
    def mag_Sd_sq(self) -> torch.Tensor | None:
        """Scalar Sd_ij * Sd_ij ``(n_cells,)`` or ``None``."""
        return self._mag_Sd_sq
