"""
Wave generation inlet boundary condition.

Generates waves at an inlet using Airy (linear) wave theory.
The velocity and pressure are prescribed according to the linearised
surface gravity wave solution:

    u = A * omega * cosh(k(z+d)) / sinh(kd) * cos(kx - omega*t)
    w = A * omega * sinh(k(z+d)) / sinh(kd) * sin(kx - omega*t)

where:
    A = wave amplitude (m)
    k = 2*pi / wavelength = wave number (1/m)
    omega = 2*pi / T = angular frequency (rad/s)
    d = water depth (m)
    T = wave period (s)
    z = vertical coordinate measured from still water level

The dispersion relation relates omega and k:

    omega^2 = g * k * tanh(k * d)

Usage::

    @BoundaryCondition.register("waveGeneration")
    class WaveGenerationBC(BoundaryCondition):
        ...
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["WaveGenerationBC"]

# Gravitational acceleration (m/s^2)
_G: float = 9.81


@BoundaryCondition.register("waveGeneration")
class WaveGenerationBC(BoundaryCondition):
    """Wave generation inlet boundary condition using Airy wave theory.

    Prescribes velocity at the inlet based on linear surface gravity
    waves.  The wave parameters (amplitude, period, depth) fully
    specify the velocity field.

    Coefficients:
        - ``amplitude``: wave amplitude A in m (default 0.01).
        - ``period``: wave period T in s (default 1.0).
        - ``depth``: water depth d in m (default 1.0).
        - ``phase``: phase offset phi in rad (default 0.0).
        - ``g``: gravitational acceleration (default 9.81).
        - ``zRef``: z-coordinate of still water level (default 0.0).
        - ``rampTime``: time over which amplitude ramps from 0 to 1 (default 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._amplitude: float = float(self._coeffs.get("amplitude", 0.01))
        self._period: float = float(self._coeffs.get("period", 1.0))
        self._depth: float = float(self._coeffs.get("depth", 1.0))
        self._phase: float = float(self._coeffs.get("phase", 0.0))
        self._g: float = float(self._coeffs.get("g", _G))
        self._z_ref: float = float(self._coeffs.get("zRef", 0.0))
        self._ramp_time: float = float(self._coeffs.get("rampTime", 0.0))

        # Derived quantities
        self._omega = 2.0 * math.pi / max(self._period, 1e-30)
        self._k = self._solve_dispersion_relation()

    def _solve_dispersion_relation(self) -> float:
        """Solve the dispersion relation for the wave number.

        omega^2 = g * k * tanh(k * d)

        Uses Newton-Raphson iteration.

        Returns
        -------
        float
            Wave number k (1/m).
        """
        omega2 = self._omega ** 2
        d = max(self._depth, 1e-10)
        g = self._g

        # Deep-water initial guess: k_0 = omega^2 / g
        k = omega2 / g

        for _ in range(50):
            tanh_kd = math.tanh(k * d)
            f = g * k * tanh_kd - omega2
            # Derivative: g * (tanh(kd) + kd * sech^2(kd))
            sech_kd = 1.0 / math.cosh(k * d)
            df = g * (tanh_kd + k * d * sech_kd ** 2)
            if abs(df) < 1e-30:
                break
            dk = f / df
            k = k - dk
            if abs(dk) < 1e-12 * max(abs(k), 1e-30):
                break
            k = max(k, 1e-10)  # Keep positive

        return k

    def _ramp_factor(self, time: float) -> float:
        """Compute amplitude ramp factor.

        Returns 1.0 after ramp time, linearly increasing before that.
        """
        if self._ramp_time <= 0.0:
            return 1.0
        return min(1.0, time / self._ramp_time)

    def compute_velocity(
        self,
        face_centres: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """Compute wave-induced velocity at face centres.

        Parameters
        ----------
        face_centres : torch.Tensor
            ``(n_faces, 3)`` face centre coordinates.
        time : float
            Current simulation time.

        Returns
        -------
        torch.Tensor
            ``(n_faces, 3)`` velocity vectors (u, v, w).
        """
        device = get_device()
        dtype = get_default_dtype()
        face_centres = face_centres.to(device=device, dtype=dtype)

        n_faces = face_centres.shape[0]
        velocity = torch.zeros(n_faces, 3, dtype=dtype, device=device)

        A = self._amplitude * self._ramp_factor(time)
        k = self._k
        omega = self._omega
        d = self._depth

        z = face_centres[:, 2] - self._z_ref  # Depth below SWL

        # Hyperbolic functions of k*(z + d)
        kd = k * d
        sinh_kd = math.sinh(kd)
        cosh_kd = math.cosh(kd)

        # Avoid division by zero
        sinh_kd = max(sinh_kd, 1e-30)
        cosh_kd = max(cosh_kd, 1e-30)

        kz_plus_kd = k * (z + d)
        cosh_kz = torch.cosh(kz_plus_kd)
        sinh_kz = torch.sinh(kz_plus_kd)

        phase_arg = k * face_centres[:, 0] - omega * time + self._phase

        # Horizontal velocity (u-component, along x)
        velocity[:, 0] = A * omega * cosh_kz / sinh_kd * torch.cos(phase_arg)

        # Vertical velocity (w-component, along z)
        velocity[:, 2] = A * omega * sinh_kz / sinh_kd * torch.sin(phase_arg)

        return velocity

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set face values from coefficients if available.

        For full velocity computation, use :meth:`compute_velocity` with
        face centre coordinates and time.
        """
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx:patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wave generation: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
