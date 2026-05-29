"""
transformPoints2 — enhanced coordinate transformation with explicit rotation matrix.

Mirrors an enhanced version of OpenFOAM's ``transformPoints`` utility.
Extends the base ``transform_points`` with explicit rotation matrix support
and composable transformation chains.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch

if TYPE_CHECKING:
    from pyfoam.mesh.poly_mesh import PolyMesh

__all__ = ["transform_points_enhanced"]


def transform_points_enhanced(
    points: torch.Tensor,
    translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
    rotation_axis: Optional[Union[Sequence[float], torch.Tensor]] = None,
    rotation_angle: Optional[float] = None,
    scale: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    rotation_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform points with enhanced rotation matrix support.

    Supports all transformations of the base ``transform_points`` plus an
    explicit 3x3 rotation matrix.  If both ``rotation_matrix`` and
    ``rotation_axis``/``rotation_angle`` are provided, the axis-angle
    rotation is applied first, then the explicit matrix.

    Transformations are applied in the order:
    scale -> axis-angle rotation -> explicit rotation matrix -> translation.

    Parameters
    ----------
    points : Tensor
        ``(n_points, 3)`` vertex positions.
    translation : sequence of 3 floats or Tensor, optional
        Translation vector ``(tx, ty, tz)``.
    rotation_axis : sequence of 3 floats or Tensor, optional
        Unit vector defining the rotation axis.
    rotation_angle : float, optional
        Rotation angle in **degrees**.
    scale : float, sequence of 3 floats, or Tensor, optional
        Uniform scalar or per-axis ``(sx, sy, sz)`` scale factors.
    rotation_matrix : Tensor, optional
        Explicit 3x3 rotation matrix.  Applied after axis-angle rotation.

    Returns
    -------
    torch.Tensor
        Transformed ``(n_points, 3)`` points tensor.  The input tensor is
        **not** modified.

    Raises
    ------
    ValueError
        If ``rotation_matrix`` is not 3x3, or if axis/angle are
        inconsistently specified.
    """
    result = points.clone()
    device = result.device
    dtype = result.dtype

    # --- 1. Scaling ---
    if scale is not None:
        if isinstance(scale, (int, float)):
            s = torch.tensor([scale, scale, scale], device=device, dtype=dtype)
        else:
            s = torch.as_tensor(scale, device=device, dtype=dtype).flatten()
            if s.numel() == 1:
                s = s.expand(3)
            if s.numel() != 3:
                raise ValueError(
                    f"scale must be a scalar or 3-element vector, got {s.numel()} elements"
                )
        result = result * s.unsqueeze(0)

    # --- 2. Axis-angle rotation ---
    if rotation_axis is not None or rotation_angle is not None:
        if rotation_axis is None or rotation_angle is None:
            raise ValueError(
                "rotation_axis and rotation_angle must both be provided"
            )
        R = _build_rotation_matrix(rotation_axis, rotation_angle, device, dtype)
        result = result @ R.T

    # --- 3. Explicit rotation matrix ---
    if rotation_matrix is not None:
        R_explicit = torch.as_tensor(rotation_matrix, device=device, dtype=dtype)
        if R_explicit.shape != (3, 3):
            raise ValueError(
                f"rotation_matrix must be 3x3, got shape {tuple(R_explicit.shape)}"
            )
        result = result @ R_explicit.T

    # --- 4. Translation ---
    if translation is not None:
        t = torch.as_tensor(translation, device=device, dtype=dtype).flatten()
        if t.numel() != 3:
            raise ValueError("translation must have exactly 3 elements")
        result = result + t.unsqueeze(0)

    return result


def _build_rotation_matrix(
    axis: Union[Sequence[float], torch.Tensor],
    angle_deg: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """从轴角构建 3x3 旋转矩阵（Rodrigues 公式）。"""
    ax = torch.as_tensor(axis, device=device, dtype=dtype).flatten()
    if ax.numel() != 3:
        raise ValueError("rotation_axis must have exactly 3 elements")

    axis_norm = ax.norm()
    if axis_norm < 1e-30:
        raise ValueError("rotation_axis must be a non-zero vector")
    ax = ax / axis_norm

    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    one_minus_cos = 1.0 - cos_t

    kx, ky, kz = ax[0].item(), ax[1].item(), ax[2].item()

    r00 = cos_t + kx * kx * one_minus_cos
    r01 = kx * ky * one_minus_cos - kz * sin_t
    r02 = kx * kz * one_minus_cos + ky * sin_t
    r10 = ky * kx * one_minus_cos + kz * sin_t
    r11 = cos_t + ky * ky * one_minus_cos
    r12 = ky * kz * one_minus_cos - kx * sin_t
    r20 = kz * kx * one_minus_cos - ky * sin_t
    r21 = kz * ky * one_minus_cos + kx * sin_t
    r22 = cos_t + kz * kz * one_minus_cos

    return torch.tensor(
        [[r00, r01, r02],
         [r10, r11, r12],
         [r20, r21, r22]],
        device=device, dtype=dtype,
    )
