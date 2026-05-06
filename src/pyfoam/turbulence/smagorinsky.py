"""
Smagorinsky subgrid-scale model for LES.

The Smagorinsky model (Smagorinsky, 1963) is the classical SGS model
for Large Eddy Simulation.  The SGS turbulent viscosity is:

.. math::

    \\nu_{sgs} = (C_s \\Delta)^2 |S|

where:

- :math:`C_s` is the Smagorinsky constant (default ≈ 0.17)
- :math:`\\Delta` is the filter width (cube root of cell volume)
- :math:`|S|` is the magnitude of the resolved strain rate tensor

The strain rate magnitude is:

.. math::

    |S| = \\sqrt{2 S_{ij} S_{ij}}

where :math:`S_{ij} = \\frac{1}{2}\\left(\\frac{\\partial u_i}{\\partial x_j}
+ \\frac{\\partial u_j}{\\partial x_i}\\right)`.

The Smagorinsky model is simple and robust but requires wall-damping
functions (e.g. van Driest damping) for wall-bounded flows.

References
----------
Smagorinsky, J. (1963). General circulation experiments with the
primitive equations. *Monthly Weather Review*, 91(3), 99–164.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .les_model import LESModel

__all__ = ["SmagorinskyModel"]

# Default Smagorinsky constant
DEFAULT_CS = 0.17


class SmagorinskyModel(LESModel):
    """Smagorinsky subgrid-scale model.

    The SGS viscosity is computed as:

        ν_sgs = (Cs * Δ)² * |S|

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.
    Cs : float, optional
        Smagorinsky constant.  Default is 0.17.

    Examples::

        >>> model = SmagorinskyModel(mesh, U, phi)  # doctest: +SKIP
        >>> model.correct()  # doctest: +SKIP
        >>> nut = model.nut()  # doctest: +SKIP
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
        Cs: float = DEFAULT_CS,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._Cs = Cs

    @property
    def Cs(self) -> float:
        """Smagorinsky constant."""
        return self._Cs

    @Cs.setter
    def Cs(self, value: float) -> None:
        self._Cs = value

    def nut(self) -> torch.Tensor:
        """Compute the SGS turbulent viscosity.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity:
            ν_sgs = (Cs * Δ)² * |S|

        Raises:
            RuntimeError: If :meth:`correct` has not been called.
        """
        if self._mag_S is None:
            raise RuntimeError(
                "correct() must be called before nut() to compute "
                "the strain rate tensor"
            )
        Cs_delta = self._Cs * self._delta
        return Cs_delta.pow(2) * self._mag_S

    def correct(self) -> None:
        """Update the model with the current velocity field.

        Recomputes the velocity gradient tensor and strain rate tensor
        from the current velocity field stored in :attr:`_U`.
        """
        self._compute_gradients()
