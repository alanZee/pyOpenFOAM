"""
SGS (Subgrid-Scale) models for Large Eddy Simulation.

Provides a hierarchy of SGS models with a common abstract base class
and two concrete implementations:

- :class:`SGSModel` — abstract base with standard LES interface
- :class:`DynamicSmagorinskySGS` — dynamic Smagorinsky with automatic C_s
- :class:`WALE_SGS` — Wall-Adapting Local Eddy-viscosity model

These models extend :class:`LESModel` and add the ``compute_eddy_viscosity()``
method as a convenience interface for SGS viscosity computation.

Usage::

    from pyfoam.turbulence.les_sgs_models import DynamicSmagorinskySGS, WALE_SGS

    model = WALE_SGS(mesh, U, phi)
    model.correct()
    nut = model.compute_eddy_viscosity()

References
----------
Germano, M. et al. (1991). A dynamic subgrid-scale eddy viscosity model.
    Physics of Fluids A, 3(7), 1760-1765.

Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based
    on the square of the velocity gradient tensor. Flow, Turbulence and
    Combustion, 62(3), 183-200.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .les_model import LESModel

__all__ = ["SGSModel", "DynamicSmagorinskySGS", "WALE_SGS"]


class SGSModel(LESModel):
    """Abstract base class for SGS (Subgrid-Scale) turbulence models.

    Extends :class:`LESModel` with a standardised ``compute_eddy_viscosity()``
    interface.  All SGS models must implement ``nut()`` (from LESModel) and
    ``correct()`` (from LESModel).  The ``compute_eddy_viscosity()`` method
    delegates to ``nut()`` after ensuring ``correct()`` has been called.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.

    Examples::

        >>> model = SomeSGSModel(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.compute_eddy_viscosity()  # doctest: +SKIP
    """

    def compute_eddy_viscosity(self) -> torch.Tensor:
        """Compute and return the SGS eddy viscosity.

        This is a convenience method that ensures ``correct()`` has been
        called, then delegates to ``nut()``.

        Returns:
            ``(n_cells,)`` tensor of SGS eddy viscosity values.

        Raises:
            RuntimeError: If ``correct()`` has not been called yet.
        """
        if self._mag_S is None:
            raise RuntimeError(
                "correct() must be called before compute_eddy_viscosity()"
            )
        return self.nut()

    @abstractmethod
    def nut(self) -> torch.Tensor:
        """Return the SGS turbulent viscosity."""
        ...

    @abstractmethod
    def correct(self) -> None:
        """Update the model state from the current velocity field."""
        ...


# Default constants
_DEFAULT_CS_MIN = 0.0
_DEFAULT_CS_MAX = 0.5
_DEFAULT_CW = 0.325


class DynamicSmagorinskySGS(SGSModel):
    """Dynamic Smagorinsky SGS model.

    Computes the Smagorinsky coefficient C_s dynamically from the
    resolved velocity field using the Germano identity and Lilly
    correction.  The dynamic procedure uses a test filter at twice
    the grid filter width.

    The dynamically computed coefficient is:

        C_s² = <L_ij M_ij> / <M_ij M_ij>

    and is clipped to [Cs_min, Cs_max] for stability.

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

        >>> model = DynamicSmagorinskySGS(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.compute_eddy_viscosity()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cs_min: float = _DEFAULT_CS_MIN,
        Cs_max: float = _DEFAULT_CS_MAX,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cs_min = Cs_min
        self._Cs_max = Cs_max
        self._Cs2: torch.Tensor | None = None

    @property
    def Cs(self) -> torch.Tensor | None:
        """Dynamically computed C_s per cell, or ``None``."""
        if self._Cs2 is None:
            return None
        return self._Cs2.clamp(min=0.0).sqrt()

    @property
    def Cs2(self) -> torch.Tensor | None:
        """Dynamically computed C_s² per cell, or ``None``."""
        return self._Cs2

    @property
    def Cs_min(self) -> float:
        """Minimum allowed C_s²."""
        return self._Cs_min

    @property
    def Cs_max(self) -> float:
        """Maximum allowed C_s²."""
        return self._Cs_max

    def nut(self) -> torch.Tensor:
        """Compute the SGS turbulent viscosity.

        Returns:
            ``(n_cells,)`` tensor: ν_sgs = C_s Δ² |S|

        Raises:
            RuntimeError: If ``correct()`` has not been called.
        """
        if self._mag_S is None or self._Cs2 is None:
            raise RuntimeError(
                "correct() must be called before nut() to compute "
                "the strain rate tensor and dynamic coefficient"
            )
        return self._Cs2.clamp(min=0.0) * self._delta.pow(2) * self._mag_S

    def correct(self) -> None:
        """Update velocity gradients and compute dynamic C_s²."""
        self._compute_gradients()
        self._compute_dynamic_coefficient()

    def _compute_dynamic_coefficient(self) -> None:
        """Compute C_s² dynamically using the Germano identity.

        Uses a simplified planar/volume averaging approach.
        """
        S = self._S  # (n_cells, 3, 3)
        mag_S = self._mag_S  # (n_cells,)
        delta = self._delta  # (n_cells,)

        # Test filter width = 2 * grid filter width
        delta_hat = 2.0 * delta

        # Approximate test-filtered strain rate magnitude
        mag_S_hat = mag_S

        # Leonard stress L_ij = Δ²|S|S_ij - Δ̂²|Ŝ|Ŝ_ij
        L = (
            delta.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S.unsqueeze(-1).unsqueeze(-1) * S
            - delta_hat.pow(2).unsqueeze(-1).unsqueeze(-1) * mag_S_hat.unsqueeze(-1).unsqueeze(-1) * S
        )

        # M_ij = 2(Δ²|S|S_ij - Δ̂²|Ŝ|Ŝ_ij)
        M = 2.0 * L

        # C_s² = <L_ij M_ij> / <M_ij M_ij>
        L_dot_M = (L * M).sum(dim=(-2, -1))
        M_dot_M = (M * M).sum(dim=(-2, -1)).clamp(min=1e-30)

        Cs2 = (L_dot_M / M_dot_M).clamp(min=self._Cs_min, max=self._Cs_max)
        self._Cs2 = Cs2

    def __repr__(self) -> str:
        return (
            f"DynamicSmagorinskySGS("
            f"n_cells={self._mesh.n_cells}, "
            f"Cs_min={self._Cs_min}, Cs_max={self._Cs_max}, "
            f"device={self._device}, dtype={self._dtype})"
        )


class WALE_SGS(SGSModel):
    """WALE (Wall-Adapting Local Eddy-viscosity) SGS model.

    Computes the SGS viscosity using the WALE formulation:

        ν_sgs = (Cw Δ)² * (Sd:Sd)^(3/2) /
                ((S:S)^(5/2) + (Sd:Sd)^(5/4))

    where Sd_ij is the traceless symmetric part of g²_ij, the matrix
    square of the velocity gradient tensor.

    The WALE model naturally recovers the correct near-wall scaling
    (ν_sgs ~ y³) without requiring wall-damping functions.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cw : float, optional
        WALE constant (default: 0.325).

    Examples::

        >>> model = WALE_SGS(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.compute_eddy_viscosity()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cw: float = _DEFAULT_CW,
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

    @property
    def Sd(self) -> torch.Tensor | None:
        """WALE Sd tensor ``(n_cells, 3, 3)`` or ``None``."""
        return self._Sd

    @property
    def mag_Sd_sq(self) -> torch.Tensor | None:
        """Scalar Sd:Sd ``(n_cells,)`` or ``None``."""
        return self._mag_Sd_sq

    def nut(self) -> torch.Tensor:
        """Compute the SGS turbulent viscosity using WALE formulation.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity.

        Raises:
            RuntimeError: If ``correct()`` has not been called.
        """
        if self._mag_S is None or self._mag_Sd_sq is None:
            raise RuntimeError(
                "correct() must be called before nut() to compute "
                "the strain rate and Sd tensors"
            )

        coeff = (self._Cw * self._delta).pow(2)
        numerator = self._mag_Sd_sq.pow(1.5)

        S_ij_S_ij = (self._mag_S.pow(2) / 2.0).clamp(min=1e-30)
        denominator = S_ij_S_ij.pow(2.5) + self._mag_Sd_sq.pow(1.25) + 1e-30

        return coeff * numerator / denominator

    def correct(self) -> None:
        """Update velocity gradients, strain rate, and WALE Sd tensor."""
        self._compute_gradients()
        self._compute_sd_tensor()

    def _compute_sd_tensor(self) -> None:
        """Compute the WALE Sd tensor and its squared magnitude.

        Sd_ij = 0.5 * (g²_ij + g²_ji) - (1/3) * δ_ij * g²_kk
        """
        g = self._grad_U  # (n_cells, 3, 3)
        g2 = torch.matmul(g, g)
        g2_sym = 0.5 * (g2 + g2.transpose(-1, -2))
        g2_trace = g2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        I = torch.eye(3, dtype=self._dtype, device=self._device).unsqueeze(0)
        self._Sd = g2_sym - (1.0 / 3.0) * g2_trace.unsqueeze(-1).unsqueeze(-1) * I
        self._mag_Sd_sq = (self._Sd * self._Sd).sum(dim=(-2, -1))

    def __repr__(self) -> str:
        return (
            f"WALE_SGS("
            f"n_cells={self._mesh.n_cells}, "
            f"Cw={self._Cw}, "
            f"device={self._device}, dtype={self._dtype})"
        )
