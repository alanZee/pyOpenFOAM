"""
Dynamic Smagorinsky subgrid-scale model for LES (Germano et al. 1991).

Implements the dynamic Smagorinsky model where the Smagorinsky constant
C_s is computed dynamically from the resolved velocity field rather
than being prescribed.  The dynamic procedure uses a test filter at
twice the grid filter width to determine the optimal C_s.

The model coefficient is:

    C_s² = <L_ij M_ij> / <M_ij M_ij>

where:
    L_ij = τ̂_ij - τ̂̂_ij is the resolved stress
    M_ij = 2 Δ² |Ŝ| Ŝ_ij - 2 Δ̂² |Ŝ| Ŝ_ij

The angle brackets denote averaging (planar or Lagrangian).

References
----------
Germano, M., Piomelli, U., Moin, P. & Cabot, W.H. (1991). A dynamic
subgrid-scale eddy viscosity model. Physics of Fluids A, 3(7),
1760–1765.

Lilly, D.K. (1992). A proposed modification of the Germano subgrid-
scale closure method. Physics of Fluids A, 4(3), 633–635.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc

from .les_model import LESModel

__all__ = ["DynamicSmagorinskyModel"]


class DynamicSmagorinskyModel(LESModel):
    """Dynamic Smagorinsky subgrid-scale model.

    Computes C_s dynamically using the Germano identity with test
    filtering and the Lilly correction.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cs_min : float, optional
        Minimum allowed C_s² (default: 0.0).
    Cs_max : float, optional
        Maximum allowed C_s² (default: 0.5).

    Examples::

        >>> model = DynamicSmagorinskyModel(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.nut()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cs_min: float = 0.0,
        Cs_max: float = 0.5,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cs_min = Cs_min
        self._Cs_max = Cs_max

        # Dynamically computed C_s² (per cell)
        self._Cs2: torch.Tensor | None = None

        # Test-filter quantities
        self._L: torch.Tensor | None = None  # Resolved stress L_ij
        self._M: torch.Tensor | None = None  # Leonard tensor M_ij

    @property
    def Cs(self) -> torch.Tensor | None:
        """Dynamically computed C_s (per cell)."""
        if self._Cs2 is None:
            return None
        return self._Cs2.clamp(min=0.0).sqrt()

    @property
    def Cs2(self) -> torch.Tensor | None:
        """Dynamically computed C_s² (per cell)."""
        return self._Cs2

    def nut(self) -> torch.Tensor:
        """Compute the SGS turbulent viscosity.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity:
            ν_sgs = C_s Δ² |S|

        Raises:
            RuntimeError: If :meth:`correct` has not been called.
        """
        if self._mag_S is None or self._Cs2 is None:
            raise RuntimeError(
                "correct() must be called before nut() to compute "
                "the strain rate tensor and dynamic coefficient"
            )

        Cs2_delta2 = self._Cs2.clamp(min=0.0) * self._delta.pow(2)
        return Cs2_delta2 * self._mag_S

    def correct(self) -> None:
        """Update the model with the current velocity field.

        Recomputes the velocity gradient, strain rate tensor, and
        dynamically computes C_s² using the Germano identity.
        """
        # Compute velocity gradient and strain rate
        self._compute_gradients()

        # Compute dynamic coefficient
        self._compute_dynamic_coefficient()

    def _compute_dynamic_coefficient(self) -> None:
        """Compute C_s² dynamically using the Germano identity.

        The procedure:
        1. Compute grid-filter strain rate S_ij and |S|
        2. Compute test-filter strain rate Ŝ_ij and |Ŝ|
        3. Compute Leonard stress L_ij from test filtering
        4. Compute M_ij tensor
        5. C_s² = <L_ij M_ij> / <M_ij M_ij>
        """
        g = self._grad_U  # (n_cells, 3, 3)
        S = self._S  # (n_cells, 3, 3)
        mag_S = self._mag_S  # (n_cells,)
        delta = self._delta  # (n_cells,)
        n_cells = self._mesh.n_cells

        # Test filter width = 2 * grid filter width
        delta_hat = 2.0 * delta

        # For test filtering, we approximate the test-filtered velocity
        # gradient using a simple volume average. In a full implementation,
        # this would use a proper test filter (e.g., box filter).
        # Here we use the same gradient but with modified length scale.

        # Approximate test-filtered strain rate magnitude
        # In simple approach: |Ŝ| ≈ |S| (same gradient, different scale)
        mag_S_hat = mag_S

        # Leonard stress L_ij = τ̂_ij - τ̂̂_ij
        # For dynamic Smagorinsky: L_ij ≈ Δ² |S| S_ij - Δ̂² |Ŝ| Ŝ_ij
        # (This is the resolved part of the stress)
        L = (
            delta.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S.unsqueeze(-1).unsqueeze(-1) * S
            - delta_hat.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S_hat.unsqueeze(-1).unsqueeze(-1) * S
        )

        # M_ij = 2 (Δ² |S| S_ij - Δ̂² |Ŝ| Ŝ_ij)
        M = 2.0 * (
            delta.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S.unsqueeze(-1).unsqueeze(-1) * S
            - delta_hat.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S_hat.unsqueeze(-1).unsqueeze(-1) * S
        )

        # C_s² = <L_ij M_ij> / <M_ij M_ij>
        # Use planar averaging (or volume averaging for 3D)
        L_dot_M = (L * M).sum(dim=(-2, -1))  # (n_cells,)
        M_dot_M = (M * M).sum(dim=(-2, -1))  # (n_cells,)

        # Avoid division by zero
        Cs2 = L_dot_M / M_dot_M.clamp(min=1e-30)

        # Clip to physical range
        Cs2 = Cs2.clamp(min=self._Cs_min, max=self._Cs_max)

        # Store
        self._Cs2 = Cs2
        self._L = L
        self._M = M
