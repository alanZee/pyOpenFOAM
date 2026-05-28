"""
Turbulence inlet models for prescribing turbulence quantities at
boundary inlets.

Provides models for setting turbulence boundary conditions at inlets
where the turbulence state is known or can be estimated.

Models:

- :class:`TurbulenceInletModel` — abstract base with RTS registry
- :class:`FixedTurbulenceInlet` — fixed turbulence quantities at inlet
- :class:`MappedTurbulenceInlet` — mapped turbulence from a reference

Usage::

    from pyfoam.turbulence.turbulence_inlet_models import (
        FixedTurbulenceInlet,
        MappedTurbulenceInlet,
    )

    # Fixed turbulence at inlet: k=0.01, epsilon=0.001
    model = FixedTurbulenceInlet(k=0.01, epsilon=0.001)
    k_bc = model.compute_k(n_faces=100)
    eps_bc = model.compute_epsilon(n_faces=100)

    # Mapped turbulence from reference data
    model = MappedTurbulenceInlet()
    model.set_reference(k_ref, epsilon_ref)
    k_bc = model.compute_k(n_faces=100)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceInletModel",
    "FixedTurbulenceInlet",
    "MappedTurbulenceInlet",
]

logger = logging.getLogger(__name__)

# Turbulence model constants
_C_MU: float = 0.09


class TurbulenceInletModel(ABC):
    """Abstract base class for turbulence inlet models.

    Subclasses implement :meth:`compute_k`, :meth:`compute_epsilon`, and
    optionally :meth:`compute_omega` to prescribe turbulence quantities
    at inlet boundaries.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceInletModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register an inlet model under *name*."""

        def decorator(model_cls: Type[TurbulenceInletModel]) -> Type[TurbulenceInletModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence inlet model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceInletModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence inlet model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_k(self, n_faces: int) -> torch.Tensor:
        """Compute turbulent kinetic energy at inlet faces.

        Parameters
        ----------
        n_faces : int
            Number of inlet faces.

        Returns
        -------
        torch.Tensor
            k values at inlet faces ``(n_faces,)``.
        """

    @abstractmethod
    def compute_epsilon(self, n_faces: int) -> torch.Tensor:
        """Compute turbulent dissipation rate at inlet faces.

        Parameters
        ----------
        n_faces : int
            Number of inlet faces.

        Returns
        -------
        torch.Tensor
            epsilon values at inlet faces ``(n_faces,)``.
        """

    def compute_omega(self, n_faces: int) -> torch.Tensor:
        """Compute specific dissipation rate at inlet faces.

        Default implementation converts from epsilon and k:
            omega = epsilon / (C_mu * k)

        Parameters
        ----------
        n_faces : int
            Number of inlet faces.

        Returns
        -------
        torch.Tensor
            omega values at inlet faces ``(n_faces,)``.
        """
        k = self.compute_k(n_faces).clamp(min=1e-16)
        eps = self.compute_epsilon(n_faces)
        return (eps / (_C_MU * k)).clamp(min=1e-10)


@TurbulenceInletModel.register("fixedTurbulenceInlet")
class FixedTurbulenceInlet(TurbulenceInletModel):
    """Fixed turbulence quantities at inlet.

    Prescribes uniform turbulence quantities (k, epsilon, and optionally
    omega) at all inlet faces.  This is the simplest inlet model,
    suitable for cases where the inlet turbulence state is known.

    Parameters
    ----------
    k : float
        Turbulent kinetic energy. Default: 0.01.
    epsilon : float
        Turbulent dissipation rate. Default: 0.001.
    omega : float or None
        Specific dissipation rate. If None, computed from k and epsilon.
    intensity : float or None
        Turbulent intensity (alternative specification). If provided
        along with U_ref, k is computed as:
            k = 1.5 * (intensity * U_ref)^2
    U_ref : float
        Reference velocity for intensity-based specification. Default: 1.0.
    length_scale : float or None
        Turbulent length scale. If provided, epsilon is computed as:
            epsilon = C_mu^{3/4} * k^{3/2} / length_scale
    C_mu : float
        Model constant. Default: 0.09.
    """

    def __init__(
        self,
        k: float | None = 0.01,
        epsilon: float | None = 0.001,
        omega: float | None = None,
        intensity: float | None = None,
        U_ref: float = 1.0,
        length_scale: float | None = None,
        C_mu: float = _C_MU,
    ) -> None:
        self.C_mu = C_mu

        # Intensity-based k specification
        if intensity is not None:
            self._k = 1.5 * (intensity * U_ref) ** 2
        elif k is not None:
            self._k = k
        else:
            self._k = 0.01

        # Length-scale-based epsilon specification
        if length_scale is not None:
            k_val = max(self._k, 1e-16)
            self._epsilon = C_mu ** 0.75 * k_val ** 1.5 / max(length_scale, 1e-10)
        elif epsilon is not None:
            self._epsilon = epsilon
        else:
            self._epsilon = 0.001

        self._omega = omega  # None means compute from k, epsilon

    @property
    def k_value(self) -> float:
        """Prescribed turbulent kinetic energy."""
        return self._k

    @property
    def epsilon_value(self) -> float:
        """Prescribed dissipation rate."""
        return self._epsilon

    @property
    def omega_value(self) -> float | None:
        """Prescribed specific dissipation rate (None = auto-computed)."""
        return self._omega

    def compute_k(self, n_faces: int) -> torch.Tensor:
        """Return uniform k at all inlet faces."""
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_faces,), self._k, dtype=dtype, device=device)

    def compute_epsilon(self, n_faces: int) -> torch.Tensor:
        """Return uniform epsilon at all inlet faces."""
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_faces,), self._epsilon, dtype=dtype, device=device)

    def compute_omega(self, n_faces: int) -> torch.Tensor:
        """Return omega at all inlet faces.

        If omega was explicitly provided, use it; otherwise compute
        from k and epsilon.
        """
        device = get_device()
        dtype = get_default_dtype()
        if self._omega is not None:
            return torch.full((n_faces,), self._omega, dtype=dtype, device=device)
        return super().compute_omega(n_faces)


@TurbulenceInletModel.register("mappedTurbulenceInlet")
class MappedTurbulenceInlet(TurbulenceInletModel):
    """Mapped turbulence from a reference location.

    Maps turbulence quantities from a reference dataset (e.g., a
    recycling plane or experimental data) to the inlet faces.

    Supports two mapping modes:
    1. **Uniform**: A single reference value is broadcast to all faces.
    2. **Spatially varying**: Face-by-face reference values are
       interpolated or resampled to the inlet.

    If the reference data has a different number of points than the
    inlet faces, linear resampling is applied.

    Parameters
    ----------
    scale_k : float
        Scaling factor for mapped k. Default: 1.0.
    scale_epsilon : float
        Scaling factor for mapped epsilon. Default: 1.0.
    C_mu : float
        Model constant. Default: 0.09.
    """

    def __init__(
        self,
        scale_k: float = 1.0,
        scale_epsilon: float = 1.0,
        C_mu: float = _C_MU,
    ) -> None:
        self.scale_k = scale_k
        self.scale_epsilon = scale_epsilon
        self.C_mu = C_mu
        self._k_ref: torch.Tensor | None = None
        self._epsilon_ref: torch.Tensor | None = None
        self._omega_ref: torch.Tensor | None = None

    def set_reference(
        self,
        k_ref: torch.Tensor,
        epsilon_ref: torch.Tensor,
        omega_ref: torch.Tensor | None = None,
    ) -> None:
        """Set reference turbulence data.

        Parameters
        ----------
        k_ref : torch.Tensor
            Reference k values ``(n_ref,)`` or scalar.
        epsilon_ref : torch.Tensor
            Reference epsilon values ``(n_ref,)`` or scalar.
        omega_ref : torch.Tensor, optional
            Reference omega values ``(n_ref,)`` or scalar.
        """
        device = get_device()
        dtype = get_default_dtype()
        self._k_ref = k_ref.to(device=device, dtype=dtype)
        self._epsilon_ref = epsilon_ref.to(device=device, dtype=dtype)
        if omega_ref is not None:
            self._omega_ref = omega_ref.to(device=device, dtype=dtype)
        else:
            self._omega_ref = None

    def _resample(self, ref: torch.Tensor, n_faces: int) -> torch.Tensor:
        """Resample reference data to n_faces using linear interpolation.

        Handles both scalar and 1-D reference tensors.
        """
        device = get_device()
        dtype = get_default_dtype()

        if ref.dim() == 0 or ref.numel() == 1:
            # Scalar: broadcast
            val = ref.flatten()[0].to(device=device, dtype=dtype)
            return torch.full((n_faces,), val.item(), dtype=dtype, device=device)

        if ref.numel() == n_faces:
            return ref.to(device=device, dtype=dtype)

        # Linear resampling
        n_ref = ref.numel()
        indices = torch.linspace(0, n_ref - 1, n_faces, device=device, dtype=dtype)
        idx_low = indices.long().clamp(max=n_ref - 2)
        idx_high = idx_low + 1
        frac = indices - idx_low.to(dtype)

        ref_d = ref.to(device=device, dtype=dtype).flatten()
        return ref_d[idx_low] * (1.0 - frac) + ref_d[idx_high] * frac

    def compute_k(self, n_faces: int) -> torch.Tensor:
        """Compute mapped k at inlet faces."""
        if self._k_ref is None:
            logger.warning("No k reference data set; returning zeros")
            device = get_device()
            dtype = get_default_dtype()
            return torch.zeros(n_faces, dtype=dtype, device=device)

        k = self._resample(self._k_ref, n_faces)
        return (k * self.scale_k).clamp(min=0.0)

    def compute_epsilon(self, n_faces: int) -> torch.Tensor:
        """Compute mapped epsilon at inlet faces."""
        if self._epsilon_ref is None:
            logger.warning("No epsilon reference data set; returning zeros")
            device = get_device()
            dtype = get_default_dtype()
            return torch.zeros(n_faces, dtype=dtype, device=device)

        eps = self._resample(self._epsilon_ref, n_faces)
        return (eps * self.scale_epsilon).clamp(min=1e-30)

    def compute_omega(self, n_faces: int) -> torch.Tensor:
        """Compute mapped omega at inlet faces.

        If omega reference data is set, uses it directly; otherwise
        computes from k and epsilon.
        """
        if self._omega_ref is not None:
            omega = self._resample(self._omega_ref, n_faces)
            return omega.clamp(min=1e-10)

        return super().compute_omega(n_faces)
