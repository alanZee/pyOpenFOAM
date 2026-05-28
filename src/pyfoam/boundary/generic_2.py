"""
Generic2 boundary condition — enhanced generic BC.

An enhanced version of the generic mixed BC that adds:
- **Time-varying coefficients**: ``value``, ``gradient``, and ``valueFraction``
  can be specified as table-interpolated functions of time.
- **Blending functions**: configurable blending between fixed-value and
  zero-gradient treatment with selectable blend modes (linear, harmonic,
  exponential).

In OpenFOAM syntax::

    type            generic2;
    value           uniform 10;
    gradient        uniform 0;
    valueFraction   uniform 0.8;
    blendMode       linear;       // linear | harmonic | exponential
    timeVarying     true;         // enable time-dependent coefficients
    valueTable      ((0 5) (1 10) (2 15));  // (time value) pairs
    fractionTable   ((0 1) (1 0.5) (2 0));  // (time fraction) pairs

When ``timeVarying = false`` (default), the BC behaves like an enhanced
generic BC with selectable blending modes.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["Generic2BC"]

# Blending function registry
_BLEND_MODES = {}


def _register_blend_mode(name: str):
    def decorator(func):
        _BLEND_MODES[name] = func
        return func
    return decorator


@_register_blend_mode("linear")
def _blend_linear(
    f: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    owner: torch.Tensor,
) -> torch.Tensor:
    """Standard linear blending: face = f * v + (1-f) * owner."""
    return f * v + (1.0 - f) * owner


@_register_blend_mode("harmonic")
def _blend_harmonic(
    f: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    owner: torch.Tensor,
) -> torch.Tensor:
    """Harmonic blending: face = 1 / (f/v + (1-f)/owner).

    Falls back to linear when owner or v is near zero.
    """
    _EPS = 1e-30
    v_safe = v.abs().clamp(min=_EPS) * v.sign().clamp(min=0).add(
        (v.abs() < _EPS).to(v.dtype) * _EPS
    )
    o_safe = owner.abs().clamp(min=_EPS) * owner.sign().clamp(min=0).add(
        (owner.abs() < _EPS).to(owner.dtype) * _EPS
    )
    denom = f / v_safe + (1.0 - f) / o_safe
    denom = denom.abs().clamp(min=_EPS) * denom.sign().clamp(min=0).add(
        (denom.abs() < _EPS).to(denom.dtype) * _EPS
    )
    return 1.0 / denom


@_register_blend_mode("exponential")
def _blend_exponential(
    f: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    owner: torch.Tensor,
) -> torch.Tensor:
    """Exponential blending: face = owner + (v - owner) * (1 - exp(-f*k)).

    k=3 gives a smooth S-curve transition.
    """
    k = 3.0
    weight = 1.0 - torch.exp(-f * k)
    return owner + (v - owner) * weight


def _interpolate_table(table: list[tuple[float, float]], t: float) -> float:
    """Linearly interpolate a (time, value) table at time t."""
    if not table:
        return 0.0
    if len(table) == 1:
        return table[0][1]
    if t <= table[0][0]:
        return table[0][1]
    if t >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        t0, v0 = table[i]
        t1, v1 = table[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            return v0 + frac * (v1 - v0)
    return table[-1][1]


@BoundaryCondition.register("generic2")
class Generic2BC(BoundaryCondition):
    """Enhanced generic boundary condition.

    Extends :class:`GenericBC` with time-varying coefficients and
    configurable blending functions between Dirichlet and Neumann
    treatments.

    Parameters
    ----------
    patch : Patch
        Boundary patch.
    coeffs : dict
        BC coefficients. Supported keys:

        - ``"value"``: prescribed value (scalar or per-face tensor)
        - ``"gradient"``: prescribed gradient
        - ``"valueFraction"``: blending fraction (0=Neumann, 1=Dirichlet)
        - ``"blendMode"``: ``"linear"`` (default), ``"harmonic"``, ``"exponential"``
        - ``"timeVarying"``: bool (default False)
        - ``"valueTable"``: list of (time, value) pairs
        - ``"fractionTable"``: list of (time, fraction) pairs
        - ``"currentTime"``: float, initial time for interpolation (default 0)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)

        # Static values
        self._value = self._resolve_tensor("value", 0.0)
        self._gradient = self._resolve_tensor("gradient", 0.0)
        self._value_fraction = self._resolve_tensor("valueFraction", 1.0)

        # Blending mode
        mode_name = str(self._coeffs.get("blendMode", "linear"))
        if mode_name not in _BLEND_MODES:
            available = sorted(_BLEND_MODES.keys())
            raise ValueError(
                f"Unknown blend mode '{mode_name}'. Available: {available}"
            )
        self._blend_mode = mode_name

        # Time-varying settings
        self._time_varying = bool(self._coeffs.get("timeVarying", False))
        self._current_time = float(self._coeffs.get("currentTime", 0.0))

        # Parse tables
        self._value_table = self._parse_table(self._coeffs.get("valueTable"))
        self._fraction_table = self._parse_table(self._coeffs.get("fractionTable"))

    def _resolve_tensor(self, key: str, default: float) -> torch.Tensor:
        """Parse a coefficient into a per-face tensor."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @staticmethod
    def _parse_table(raw: Any) -> list[tuple[float, float]]:
        """Parse a table specification into (time, value) pairs."""
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            result = []
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    result.append((float(item[0]), float(item[1])))
            return result
        return []

    # ------------------------------------------------------------------
    # Time update
    # ------------------------------------------------------------------

    def set_time(self, t: float) -> None:
        """Update the current time for time-varying coefficients.

        Args:
            t: Current simulation time.
        """
        self._current_time = t
        if self._time_varying:
            if self._value_table:
                new_val = _interpolate_table(self._value_table, t)
                self._value = torch.full_like(self._value, new_val)
            if self._fraction_table:
                new_frac = _interpolate_table(self._fraction_table, t)
                self._value_fraction = torch.full_like(self._value_fraction, new_frac)

    @property
    def current_time(self) -> float:
        """Current simulation time."""
        return self._current_time

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def value(self) -> torch.Tensor:
        """Prescribed fixed-value part."""
        return self._value

    @property
    def gradient(self) -> torch.Tensor:
        """Prescribed gradient part."""
        return self._gradient

    @property
    def value_fraction(self) -> torch.Tensor:
        """Value fraction (0 = pure gradient, 1 = pure value)."""
        return self._value_fraction

    @property
    def blend_mode(self) -> str:
        """Blending mode name."""
        return self._blend_mode

    @property
    def is_time_varying(self) -> bool:
        """Whether this BC has time-varying coefficients."""
        return self._time_varying

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply the BC using the configured blending mode.

        ``face_value = blend(fraction, value, gradient, owner_value)``

        where ``blend`` is the selected blending function.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        f = self._value_fraction.to(device=field.device, dtype=field.dtype)
        v = self._value.to(device=field.device, dtype=field.dtype)
        g = self._gradient.to(device=field.device, dtype=field.dtype)

        blend_fn = _BLEND_MODES[self._blend_mode]
        face_values = blend_fn(f, v, g, owner_values)

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    # ------------------------------------------------------------------
    # Matrix contributions
    # ------------------------------------------------------------------

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted penalty method based on valueFraction.

        Same as GenericBC but with the current (possibly time-varying)
        valueFraction and value.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        values = self._value.to(device=device, dtype=dtype)
        f = self._value_fraction.to(device=device, dtype=dtype)

        coeff = deltas * areas * f
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source

    def __repr__(self) -> str:
        return (
            f"Generic2BC(patch='{self._patch.name}', "
            f"blend={self._blend_mode}, "
            f"time_varying={self._time_varying})"
        )
