"""
temporalInterpolateEnhanced — enhanced temporal interpolation for OpenFOAM fields.

Extends the basic temporal interpolation with:

- **Multiple interpolation schemes**: linear, cubic spline, Lagrange
  polynomial.
- **Field selection**: Interpolate only specific fields.
- **Extrapolation control**: Clamp, linear-extrapolate, or raise error
  for out-of-range times.
- **Multi-field batch processing**: Interpolate all fields in one pass.
- **Time-derivative estimation**: Compute dF/dt at the requested times.

Mirrors and extends OpenFOAM's ``temporalInterpolate`` utility.

Usage::

    from pyfoam.tools.temporal_interpolate_enhanced import temporal_interpolate

    result = temporal_interpolate(
        fields_data={
            "p": {"0": p_0, "1": p_1},
            "U": {"0": U_0, "1": U_1},
        },
        target_times=[0.25, 0.5, 0.75],
        scheme="cubic",
    )
    interpolated = result.fields  # {"p": {0.25: ..., 0.5: ..., ...}, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

__all__ = ["TemporalInterpolateResult", "temporal_interpolate"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TemporalInterpolateResult:
    """Result from :func:`temporal_interpolate`.

    Attributes
    ----------
    fields : dict[str, dict[float, np.ndarray]]
        Interpolated field values keyed by field name, then time value.
    derivatives : dict[str, dict[float, np.ndarray]]
        Estimated time derivatives ``dF/dt`` at each target time.
    target_times : list[float]
        Requested interpolation times.
    scheme : str
        Interpolation scheme used.
    n_fields : int
        Number of fields interpolated.
    """

    fields: dict = field(default_factory=dict)
    derivatives: dict = field(default_factory=dict)
    target_times: list = field(default_factory=list)
    scheme: str = ""
    n_fields: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def temporal_interpolate(
    fields_data: Dict[str, Dict[Union[str, float], np.ndarray]],
    target_times: Sequence[float],
    scheme: str = "linear",
    extrapolation: str = "clamp",
) -> TemporalInterpolateResult:
    """Interpolate fields at specified target times.

    Parameters
    ----------
    fields_data : dict
        Nested dict ``{field_name: {time_key: array}}``.  Time keys can
        be strings or floats.  Each array can be any shape; all arrays
        for a given field must share the same shape.
    target_times : sequence of float
        Times at which to interpolate.
    scheme : str
        Interpolation scheme: ``"linear"``, ``"cubic"``, or ``"lagrange"``.
    extrapolation : str
        Out-of-range behaviour: ``"clamp"`` (use nearest boundary value),
        ``"linear"`` (linear extrapolation), or ``"error"`` (raise).

    Returns
    -------
    TemporalInterpolateResult
        Interpolated fields and time derivatives.

    Raises
    ------
    ValueError
        If inputs are invalid or extrapolation is ``"error"`` for
        out-of-range target times.
    """
    valid_schemes = {"linear", "cubic", "lagrange"}
    if scheme not in valid_schemes:
        raise ValueError(f"Unknown scheme {scheme!r}. Valid: {sorted(valid_schemes)}")

    valid_extrap = {"clamp", "linear", "error"}
    if extrapolation not in valid_extrap:
        raise ValueError(
            f"Unknown extrapolation {extrapolation!r}. Valid: {sorted(valid_extrap)}"
        )

    target_times = sorted(float(t) for t in target_times)

    # Convert time keys and sort
    parsed_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fname, time_dict in fields_data.items():
        times_list = []
        arrays_list = []
        for tk, arr in time_dict.items():
            times_list.append(float(tk))
            arrays_list.append(np.asarray(arr, dtype=np.float64))
        order = np.argsort(times_list)
        times_sorted = np.array(times_list)[order]
        arrays_sorted = [arrays_list[i] for i in order]
        # Stack into array: (n_times, *field_shape)
        stacked = np.stack(arrays_sorted, axis=0)
        parsed_fields[fname] = (times_sorted, stacked)

    # Validate target times
    for fname, (src_times, _) in parsed_fields.items():
        t_min, t_max = src_times[0], src_times[-1]
        for tt in target_times:
            if (tt < t_min - 1e-12 or tt > t_max + 1e-12) and extrapolation == "error":
                raise ValueError(
                    f"Target time {tt} is outside source range "
                    f"[{t_min}, {t_max}] for field {fname!r} "
                    f"and extrapolation='error'."
                )

    # Interpolate each field
    interp_fields: dict[str, dict[float, np.ndarray]] = {}
    deriv_fields: dict[str, dict[float, np.ndarray]] = {}

    for fname, (src_times, stacked) in parsed_fields.items():
        interp_times: dict[float, np.ndarray] = {}
        deriv_times: dict[float, np.ndarray] = {}

        for tt in target_times:
            if scheme == "linear":
                val, deriv = _linear_interp(src_times, stacked, tt, extrapolation)
            elif scheme == "cubic":
                val, deriv = _cubic_interp(src_times, stacked, tt, extrapolation)
            else:
                val, deriv = _lagrange_interp(src_times, stacked, tt, extrapolation)

            interp_times[tt] = val
            deriv_times[tt] = deriv

        interp_fields[fname] = interp_times
        deriv_fields[fname] = deriv_times

    return TemporalInterpolateResult(
        fields=interp_fields,
        derivatives=deriv_fields,
        target_times=target_times,
        scheme=scheme,
        n_fields=len(parsed_fields),
    )


# ---------------------------------------------------------------------------
# Interpolation kernels
# ---------------------------------------------------------------------------


def _clamp_time(
    t: float,
    src_times: np.ndarray,
) -> float:
    """Clamp time to source range."""
    return max(src_times[0], min(src_times[-1], t))


def _find_bracket(
    src_times: np.ndarray,
    t: float,
) -> tuple[int, int]:
    """Find the indices of the two bracketing source times."""
    idx = int(np.searchsorted(src_times, t, side="right")) - 1
    idx = max(0, min(idx, len(src_times) - 2))
    return idx, idx + 1


def _linear_interp(
    src_times: np.ndarray,
    stacked: np.ndarray,
    t: float,
    extrapolation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation with derivative estimation."""
    if extrapolation == "clamp":
        t_eff = _clamp_time(t, src_times)
    elif extrapolation == "linear":
        t_eff = t
    else:
        t_eff = t

    i0, i1 = _find_bracket(src_times, t_eff)
    t0, t1 = src_times[i0], src_times[i1]
    dt = t1 - t0

    if dt < 1e-30:
        return stacked[i0].copy(), np.zeros_like(stacked[i0])

    alpha = (t_eff - t0) / dt
    val = stacked[i0] * (1.0 - alpha) + stacked[i1] * alpha
    deriv = (stacked[i1] - stacked[i0]) / dt

    return val, deriv


def _cubic_interp(
    src_times: np.ndarray,
    stacked: np.ndarray,
    t: float,
    extrapolation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation (Catmull-Rom spline)."""
    if extrapolation == "clamp":
        t_eff = _clamp_time(t, src_times)
    else:
        t_eff = t

    n = len(src_times)
    if n < 4:
        # Fall back to linear for small datasets
        return _linear_interp(src_times, stacked, t, extrapolation)

    i0, i1 = _find_bracket(src_times, t_eff)
    # Get 4 points for Catmull-Rom: i0-1, i0, i1, i1+1
    im1 = max(0, i0 - 1)
    ip2 = min(n - 1, i1 + 1)

    t0, t1 = src_times[i0], src_times[i1]
    dt = t1 - t0
    if dt < 1e-30:
        return stacked[i0].copy(), np.zeros_like(stacked[i0])

    s = (t_eff - t0) / dt

    # Catmull-Rom basis functions
    s2 = s * s
    s3 = s2 * s

    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2

    # Tangents (scaled by dt for Catmull-Rom)
    m0 = 0.5 * (stacked[i1] - stacked[im1]) if i0 > 0 else (stacked[i1] - stacked[i0])
    m1 = 0.5 * (stacked[ip2] - stacked[i0]) if i1 < n - 1 else (stacked[i1] - stacked[i0])

    val = h00 * stacked[i0] + h10 * dt * m0 + h01 * stacked[i1] + h11 * dt * m1

    # Derivative
    dh00 = 6 * s2 - 6 * s
    dh10 = 3 * s2 - 4 * s + 1
    dh01 = -6 * s2 + 6 * s
    dh11 = 3 * s2 - 2 * s

    deriv = (dh00 * stacked[i0] + dh10 * dt * m0 + dh01 * stacked[i1] + dh11 * dt * m1) / dt

    return val, deriv


def _lagrange_interp(
    src_times: np.ndarray,
    stacked: np.ndarray,
    t: float,
    extrapolation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange polynomial interpolation (up to 4th order)."""
    if extrapolation == "clamp":
        t_eff = _clamp_time(t, src_times)
    else:
        t_eff = t

    n = len(src_times)
    if n < 3:
        return _linear_interp(src_times, stacked, t, extrapolation)

    # Select up to 4 nearest points around t_eff
    idx = int(np.searchsorted(src_times, t_eff, side="right")) - 1
    idx = max(0, min(idx, n - 1))

    # Window of up to 4 points
    order = min(4, n)
    half = order // 2
    start = max(0, min(idx - half, n - order))
    end = start + order

    ts = src_times[start:end]
    vs = stacked[start:end]

    # Lagrange interpolation
    val = np.zeros_like(stacked[0])
    deriv = np.zeros_like(stacked[0])

    for i in range(order):
        # Compute Lagrange basis
        Li = 1.0
        dLi = 0.0
        for j in range(order):
            if i == j:
                continue
            denom = ts[i] - ts[j]
            if abs(denom) < 1e-30:
                continue
            Li *= (t_eff - ts[j]) / denom
            # Derivative of Li: sum of products with one factor differentiated
            dLi_contrib = 1.0 / denom
            for k in range(order):
                if k == i or k == j:
                    continue
                denom_k = ts[i] - ts[k]
                if abs(denom_k) < 1e-30:
                    continue
                dLi_contrib *= (t_eff - ts[k]) / denom_k
            dLi += dLi_contrib

        val += Li * vs[i]
        deriv += dLi * vs[i]

    return val, deriv
