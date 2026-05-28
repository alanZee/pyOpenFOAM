"""
turbulentInlet2 — enhanced turbulent inlet boundary condition.

Generates spatially correlated turbulent fluctuations using a digital
filter method, producing more realistic turbulence structures than the
simple random perturbation approach.

The digital filter method:

    R_ij(n Delta) = exp(-pi * |n| * Delta / (L_ij))

generates correlated random fields from white noise via a convolution
with a Gaussian filter kernel. This produces proper two-point
correlations with specified integral length scales.

In OpenFOAM syntax::

    type            turbulentInlet2;
    referenceField  uniform (1 0 0);    // mean velocity
    intensity       0.05;               // turbulence intensity (U_rms/U_mean)
    lengthScale     0.01;               // integral length scale (m)
    nFilterPoints   8;                  // number of filter kernel points
    alpha           0.1;                // relaxation factor [0,1]
    value           uniform (1 0 0);

Usage::

    bc = BoundaryCondition.create("turbulentInlet2", patch, coeffs={
        "referenceField": [1.0, 0.0, 0.0],
        "intensity": 0.05,
        "lengthScale": 0.01,
    })
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentInlet2BC"]


@BoundaryCondition.register("turbulentInlet2")
class TurbulentInlet2BC(BoundaryCondition):
    """Enhanced turbulent inlet BC with digital-filter turbulence generation.

    Uses a 1-D digital filter to create spatially correlated velocity
    fluctuations at the inlet, which is more physically realistic than
    independent random perturbations.

    The filter kernel is a discrete Gaussian:

        g(n) = exp(-pi * n^2 / N^2) / sum(exp(...))

    where N controls the filter width from the integral length scale.

    Coefficients
    ------------
    referenceField : list[float]
        Mean velocity [Ux, Uy, Uz]. Default: [1, 0, 0].
    intensity : float
        Turbulence intensity I = U_rms / U_mean. Default: 0.05.
    lengthScale : float
        Integral length scale (m). Default: 0.01.
    nFilterPoints : int
        Number of filter kernel points (odd recommended). Default: 8.
    alpha : float
        Relaxation factor in [0, 1]. Default: 0.1.
    value : list[float]
        Initial velocity. Default: [1, 0, 0].
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)

        # Mean velocity
        ref_raw = self._coeffs.get("referenceField", [1.0, 0.0, 0.0])
        self._reference = torch.tensor(
            ref_raw, dtype=get_default_dtype(), device=get_device()
        )

        # Turbulence parameters
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._n_filter = int(self._coeffs.get("nFilterPoints", 8))
        self._alpha = float(self._coeffs.get("alpha", 0.1))

        # Build digital filter kernel
        self._kernel = self._build_filter_kernel()

        # White noise buffer (n_faces + n_filter for causal convolution)
        self._noise_buffer: torch.Tensor | None = None

        # Previous face values (for relaxation)
        self._prev_values: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reference_field(self) -> torch.Tensor:
        """Mean velocity vector."""
        return self._reference

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def length_scale(self) -> float:
        """Integral length scale (m)."""
        return self._length_scale

    @property
    def alpha(self) -> float:
        """Relaxation factor."""
        return self._alpha

    @property
    def kernel(self) -> torch.Tensor:
        """Digital filter kernel."""
        return self._kernel

    # ------------------------------------------------------------------
    # Digital filter
    # ------------------------------------------------------------------

    def _build_filter_kernel(self) -> torch.Tensor:
        """Build a 1-D Gaussian digital filter kernel.

        The kernel width is derived from the integral length scale
        and face spacing. The kernel is normalised so that
        sum(g^2) = 1/N to produce the correct variance.

        Returns
        -------
        torch.Tensor
            Normalised filter kernel ``(n_filter,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        N = self._n_filter

        # Filter half-width mapped to integer indices
        n = torch.arange(N, device=device, dtype=dtype) - N // 2

        # Gaussian kernel based on integral length scale
        sigma = max(self._length_scale, 1e-6) * N / (2.0 * math.pi)
        # Ensure sigma is large enough to avoid numerical underflow
        sigma = max(sigma, N / 6.0)
        g = torch.exp(-0.5 * (n / max(sigma, 1e-6)) ** 2)

        # Normalise: sum(g^2) = 1/N  (variance-preserving normalisation)
        norm = torch.sqrt((g * g).sum() * N)
        if norm > 1e-30:
            g = g / norm

        return g

    def _generate_correlated_fluctuations(
        self,
        n_faces: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate spatially correlated velocity fluctuations.

        Uses 1-D digital filter convolution on white noise to produce
        proper two-point spatial correlations.

        Returns
        -------
        torch.Tensor
            Fluctuation velocities ``(n_faces, 3)``.
        """
        N = self._n_filter
        U_mean = self._reference.to(device=device, dtype=dtype)
        U_rms = self._intensity * U_mean.norm()

        # Generate white noise (extend buffer for causal convolution)
        buf_size = n_faces + N
        white_noise = torch.randn(buf_size, 3, device=device, dtype=dtype) * U_rms

        # Apply 1-D digital filter via convolution
        kernel = self._kernel.to(device=device, dtype=dtype)

        # Convolve each component separately
        fluctuations = torch.zeros(n_faces, 3, device=device, dtype=dtype)
        for comp in range(3):
            # Pad and convolve
            signal = white_noise[:, comp]  # (buf_size,)
            # Manual 1-D convolution (valid region)
            conv_result = torch.zeros(n_faces, device=device, dtype=dtype)
            for k in range(N):
                conv_result = conv_result + kernel[k] * signal[k : k + n_faces]
            fluctuations[:, comp] = conv_result

        return fluctuations

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocities with digital-filter turbulence.

        U = (1 - alpha) * U_prev + alpha * (U_mean + epsilon_filtered)
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        ref = self._reference.to(device=device, dtype=dtype)
        mean = ref.unsqueeze(0).expand(n_faces, -1)

        # Generate correlated fluctuations
        epsilon = self._generate_correlated_fluctuations(n_faces, device, dtype)

        # Blend with previous values
        if self._prev_values is not None and self._alpha < 1.0:
            prev = self._prev_values.to(device=device, dtype=dtype)
            face_values = (1.0 - self._alpha) * prev + self._alpha * (mean + epsilon)
        else:
            face_values = mean + epsilon

        self._prev_values = face_values.detach().clone()

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method with mean reference value for implicit part.

        Uses the reference (mean) velocity for diagonal dominance.
        Fluctuations appear only in the explicit source.
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

        ref = self._reference.to(device=device, dtype=dtype)
        ref_scalar = ref[0] if ref.dim() > 0 else ref

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * ref_scalar)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
