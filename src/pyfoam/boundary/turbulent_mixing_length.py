"""
Turbulent mixing length boundary condition.

Implements ``turbulentMixingLength`` — a general-purpose mixing-length BC
that sets either epsilon or omega depending on the turbulence model being
used.  Complements the existing
``turbulentMixingLengthDissipationRateInlet`` and
``turbulentMixingLengthFrequencyInlet`` with a unified interface.

In OpenFOAM syntax::

    type            turbulentMixingLength;
    mixingLength    0.01;           // mixing length (m)
    Cmu             0.09;           // model constant
    intensity       0.05;           // turbulence intensity (for k estimate)
    value           uniform 0.01;

"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentMixingLengthBC"]


@BoundaryCondition.register("turbulentMixingLength")
class TurbulentMixingLengthBC(BoundaryCondition):
    """Turbulent mixing length boundary condition.

    Computes turbulence quantities from a specified mixing length.
    Operates in two modes selected by the ``mode`` coefficient:

    - ``"epsilon"`` (default): ``epsilon = C_mu^0.75 * k^1.5 / l``
    - ``"omega"``: ``omega = sqrt(k) / (C_mu^0.25 * l)``

    Where k is estimated from velocity and turbulence intensity::

        k = 1.5 * (I * |U|)^2

    Coefficients:
        - ``mixingLength``: Mixing length in m (default: 0.01).
        - ``Cmu``: Model constant (default: 0.09).
        - ``intensity``: Turbulence intensity (default: 0.05).
        - ``mode``: ``"epsilon"`` or ``"omega"`` (default: ``"epsilon"``).
        - ``value``: Initial value (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._mode = str(self._coeffs.get("mode", "epsilon"))

    @property
    def mixing_length(self) -> float:
        """Return the mixing length."""
        return self._mixing_length

    @property
    def C_mu(self) -> float:
        """Return the C_mu constant."""
        return self._C_mu

    @property
    def intensity(self) -> float:
        """Return the turbulence intensity."""
        return self._intensity

    @property
    def mode(self) -> str:
        """Return the operating mode ('epsilon' or 'omega')."""
        return self._mode

    def _estimate_k(
        self,
        velocity: torch.Tensor | None,
        k: torch.Tensor | None,
    ) -> torch.Tensor:
        """Estimate turbulent kinetic energy from available data.

        Priority: explicit *k* > velocity-based estimate > default.
        """
        if k is not None:
            return k
        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            return 1.5 * (self._intensity * u_mag) ** 2
        # Fallback default
        return torch.full(
            (self._patch.n_faces,),
            0.01,
            dtype=get_default_dtype(),
            device=get_device(),
        )

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face turbulence quantity from mixing length.

        Args:
            field: Turbulent quantity field (epsilon or omega).
            patch_idx: Optional start index into *field*.
            k: ``(n_faces,)`` turbulent kinetic energy at boundary.
            velocity: ``(n_faces, 3)`` velocity at boundary.
        """
        device = field.device
        dtype = field.dtype

        k_est = self._estimate_k(velocity, k).to(device=device, dtype=dtype)

        if self._mode == "omega":
            values = torch.sqrt(k_est) / (self._C_mu ** 0.25 * self._mixing_length)
        else:  # epsilon
            values = (self._C_mu ** 0.75) * (k_est ** 1.5) / self._mixing_length

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = values
        else:
            field[self._patch.face_indices] = values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for turbulence mixing length BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        default_val = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * default_val)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
