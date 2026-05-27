"""
transformPoints — transform mesh vertex coordinates.

Mirrors OpenFOAM's ``transformPoints`` utility.  Supports:

- **Translation** — shift all points by a vector.
- **Rotation** — rotate about an arbitrary axis through the origin.
- **Scaling** — uniform or anisotropic scaling.

The function returns a **new** points tensor; the original mesh is not
modified.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch

if TYPE_CHECKING:
    from pyfoam.mesh.poly_mesh import PolyMesh

__all__ = ["transform_points"]


def transform_points(
    mesh: "PolyMesh",
    translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
    rotation_axis: Optional[Union[Sequence[float], torch.Tensor]] = None,
    rotation_angle: Optional[float] = None,
    scale: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
) -> torch.Tensor:
    """Transform mesh vertex coordinates.

    Transformations are applied in the order: scale → rotate → translate,
    matching the typical OpenFOAM behaviour.

    Parameters
    ----------
    mesh : PolyMesh
        Source mesh whose ``points`` property provides the ``(n_points, 3)``
        vertex tensor.
    translation : sequence of 3 floats or Tensor, optional
        Translation vector ``(tx, ty, tz)``.
    rotation_axis : sequence of 3 floats or Tensor, optional
        Unit vector defining the rotation axis.  Required together with
        *rotation_angle*.
    rotation_angle : float, optional
        Rotation angle in **degrees**.  Required together with
        *rotation_axis*.
    scale : float, sequence of 3 floats, or Tensor, optional
        Uniform scalar or per-axis ``(sx, sy, sz)`` scale factors.

    Returns
    -------
    torch.Tensor
        Transformed ``(n_points, 3)`` points tensor.  The original mesh
        points are **not** modified.

    Raises
    ------
    ValueError
        If *rotation_axis* is given without *rotation_angle* (or vice versa),
        or if *rotation_axis* is a zero vector.
    """
    points = mesh.points.clone()
    device = points.device
    dtype = points.dtype

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
        points = points * s.unsqueeze(0)

    # --- 2. Rotation ---
    if rotation_axis is not None or rotation_angle is not None:
        if rotation_axis is None or rotation_angle is None:
            raise ValueError(
                "rotation_axis and rotation_angle must both be provided"
            )

        axis = torch.as_tensor(rotation_axis, device=device, dtype=dtype).flatten()
        if axis.numel() != 3:
            raise ValueError("rotation_axis must have exactly 3 elements")

        axis_norm = axis.norm()
        if axis_norm < 1e-30:
            raise ValueError("rotation_axis must be a non-zero vector")
        axis = axis / axis_norm

        theta = math.radians(rotation_angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        one_minus_cos = 1.0 - cos_t

        # Rodrigues' rotation matrix: R = cos*I + sin*[k]_x + (1-cos)*k⊗k
        kx, ky, kz = axis[0], axis[1], axis[2]
        r00 = cos_t + kx * kx * one_minus_cos
        r01 = kx * ky * one_minus_cos - kz * sin_t
        r02 = kx * kz * one_minus_cos + ky * sin_t
        r10 = ky * kx * one_minus_cos + kz * sin_t
        r11 = cos_t + ky * ky * one_minus_cos
        r12 = ky * kz * one_minus_cos - kx * sin_t
        r20 = kz * kx * one_minus_cos - ky * sin_t
        r21 = kz * ky * one_minus_cos + kx * sin_t
        r22 = cos_t + kz * kz * one_minus_cos

        R = torch.tensor(
            [[r00, r01, r02],
             [r10, r11, r12],
             [r20, r21, r22]],
            device=device, dtype=dtype,
        )
        points = points @ R.T

    # --- 3. Translation ---
    if translation is not None:
        t = torch.as_tensor(translation, device=device, dtype=dtype).flatten()
        if t.numel() != 3:
            raise ValueError("translation must have exactly 3 elements")
        points = points + t.unsqueeze(0)

    return points
