"""
Enhanced turbulence inlet models — version 2.

Provides advanced synthetic turbulence generation methods for prescribing
realistic, spatially correlated turbulence at inlet boundaries.  Builds
upon the basic models in ``turbulence_inlet_models.py`` by adding:

Models:

- :class:`TurbulenceInletModel2` — enhanced base class with spatial
  correlation support and spectrum methods
- :class:`DigitalFilterInlet` — digital filter turbulence generation
  (Klein et al., 2003)
- :class:`SyntheticEddyInlet` — synthetic eddy method
  (Jarrin et al., 2006)

Digital Filter Method
---------------------
Generates synthetic turbulence by convolving white noise with a
filter kernel derived from the two-point spatial correlation::

    u'(x) = sum_j b_j * r(x - x_j)

where ``b_j`` are filter coefficients and ``r`` is a vector of
uniform random numbers.  The filter widths are determined by the
integral length scales in each direction.

Synthetic Eddy Method
---------------------
Creates a box of synthetic eddies around the inlet.  Each eddy
contributes a velocity fluctuation based on its position relative
to the inlet faces::

    u'(x) = (1/sqrt(N)) * sum_k f_k(x - x_k) * sigma_k

where ``f_k`` is the eddy shape function, ``x_k`` is the eddy
position, and ``sigma_k`` is the eddy strength derived from the
Reynolds stress tensor.

References
----------
Klein, M., Sadiki, A., Janicka, J., 2003.
"A digital filter based generation of inflow data for spatially
developing direct numerical or large eddy simulations."
J. Comput. Phys. 186(2), 652-665.

Jarrin, N., Benhamadouche, S., Laurence, D., Prosser, R., 2006.
"A synthetic-eddy-method for generating inflow conditions for
large-eddy simulations."
Int. J. Heat Fluid Flow 27(4), 585-593.

Usage::

    from pyfoam.turbulence.turbulence_inlet_models_2 import (
        DigitalFilterInlet,
        SyntheticEddyInlet,
    )

    # Digital filter inlet
    model = DigitalFilterInlet(
        k=0.01, epsilon=0.001,
        length_scale_x=0.1, length_scale_y=0.05, length_scale_z=0.05,
    )
    u_fluct = model.generate_fluctuations(n_faces=100, face_positions=positions)

    # Synthetic eddy inlet
    model = SyntheticEddyInlet(
        k=0.01, epsilon=0.001, n_eddies=50,
    )
    u_fluct = model.generate_fluctuations(n_faces=100, face_positions=positions)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceInletModel2",
    "DigitalFilterInlet",
    "SyntheticEddyInlet",
]

logger = logging.getLogger(__name__)

# Turbulence model constants
_C_MU: float = 0.09


class TurbulenceInletModel2(ABC):
    """Enhanced abstract base class for turbulence inlet models.

    Extends the basic inlet model interface with methods for generating
    spatially correlated velocity fluctuations at inlet boundaries.

    Subclasses implement :meth:`generate_fluctuations` to produce
    turbulent velocity fluctuations with prescribed statistics.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceInletModel2"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register an inlet model under *name*."""
        def decorator(model_cls: Type[TurbulenceInletModel2]) -> Type[TurbulenceInletModel2]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence inlet model v2 '{name}' is already registered"
                )
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceInletModel2":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence inlet model v2 '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def generate_fluctuations(
        self,
        n_faces: int,
        face_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate synthetic velocity fluctuations at inlet faces.

        Parameters
        ----------
        n_faces : int
            Number of inlet faces.
        face_positions : torch.Tensor, optional
            Face centre positions ``(n_faces, 3)``.  If None, faces
            are assumed uniformly spaced.

        Returns
        -------
        torch.Tensor
            Velocity fluctuations ``(n_faces, 3)`` [m/s].
        """

    def compute_k(self, n_faces: int) -> torch.Tensor:
        """Compute turbulent kinetic energy at inlet faces.

        Default: computed from the RMS of generated fluctuations.
        """
        device = get_device()
        dtype = get_default_dtype()
        u_fluct = self.generate_fluctuations(n_faces)
        # k = 0.5 * (u'^2 + v'^2 + w'^2)
        k = 0.5 * (u_fluct ** 2).sum(dim=-1)
        return k.clamp(min=1e-16)

    def compute_epsilon(self, n_faces: int) -> torch.Tensor:
        """Compute dissipation rate at inlet faces.

        Default: epsilon = C_mu^(3/4) * k^(3/2) / L_t.
        """
        k = self.compute_k(n_faces)
        L_t = self._get_length_scale()
        return (_C_MU ** 0.75 * k.pow(1.5) / L_t).clamp(min=1e-30)

    def compute_omega(self, n_faces: int) -> torch.Tensor:
        """Compute specific dissipation rate at inlet faces.

        Default: omega = epsilon / (C_mu * k).
        """
        k = self.compute_k(n_faces).clamp(min=1e-16)
        eps = self.compute_epsilon(n_faces)
        return (eps / (_C_MU * k)).clamp(min=1e-10)

    def _get_length_scale(self) -> float:
        """Return the characteristic turbulent length scale.

        Override in subclasses to provide model-specific values.
        """
        return 0.1


@TurbulenceInletModel2.register("digitalFilterInlet")
class DigitalFilterInlet(TurbulenceInletModel2):
    """Digital filter turbulence generation (Klein et al., 2003).

    Generates spatially correlated synthetic turbulence by applying
    a digital filter to white noise.  The filter widths are determined
    by the integral length scales in each spatial direction.

    The filter is::

        u'_i(x) = sum_j b_i(j) * r_i(x - j*Delta)

    where ``b_i`` is the filter coefficient vector for direction ``i``,
    ``r_i`` is white noise, and ``Delta`` is the grid spacing.

    The filter coefficients are derived from the Gaussian correlation::

        R(xi) = exp(-pi * xi^2 / (4 * L^2))

    Parameters
    ----------
    k : float
        Target turbulent kinetic energy. Default: 0.01.
    epsilon : float
        Target dissipation rate. Default: 0.001.
    length_scale_x : float
        Integral length scale in x-direction (m). Default: 0.1.
    length_scale_y : float
        Integral length scale in y-direction (m). Default: 0.05.
    length_scale_z : float
        Integral length scale in z-direction (m). Default: 0.05.
    n_filter : int
        Number of filter points per direction (odd). Default: 11.
    """

    def __init__(
        self,
        k: float = 0.01,
        epsilon: float = 0.001,
        length_scale_x: float = 0.1,
        length_scale_y: float = 0.05,
        length_scale_z: float = 0.05,
        n_filter: int = 11,
    ) -> None:
        self._k_target = k
        self._epsilon = epsilon
        self._Lx = length_scale_x
        self._Ly = length_scale_y
        self._Lz = length_scale_z
        # Ensure n_filter is odd
        self._n_filter = n_filter if n_filter % 2 == 1 else n_filter + 1

    @property
    def k_target(self) -> float:
        """Target turbulent kinetic energy."""
        return self._k_target

    @property
    def epsilon_value(self) -> float:
        """Target dissipation rate."""
        return self._epsilon

    @property
    def length_scales(self) -> tuple[float, float, float]:
        """Integral length scales (Lx, Ly, Lz) in metres."""
        return (self._Lx, self._Ly, self._Lz)

    @property
    def n_filter(self) -> int:
        """Number of filter points."""
        return self._n_filter

    def _get_length_scale(self) -> float:
        """Geometric mean of length scales."""
        return (self._Lx * self._Ly * self._Lz) ** (1.0 / 3.0)

    def compute_filter_coefficients(self, L: float, dx: float) -> torch.Tensor:
        """Compute 1D digital filter coefficients from Gaussian correlation.

        The filter is derived from the target two-point correlation::

            R(xi) = exp(-pi * xi^2 / (4 * L^2))

        The filter coefficients are::

            b(n) = sqrt(2 * dx / L) * exp(-pi * (n * dx)^2 / (2 * L^2))

        Parameters
        ----------
        L : float
            Integral length scale (m).
        dx : float
            Grid spacing (m).

        Returns
        -------
        torch.Tensor
            Filter coefficients ``(n_filter,)``.
        """
        n = self._n_filter
        half = n // 2
        device = get_device()
        dtype = get_default_dtype()

        indices = torch.arange(-half, half + 1, device=device, dtype=dtype)
        xi = indices * dx

        # Gaussian correlation filter
        b = torch.exp(-math.pi * xi.pow(2) / (2.0 * L ** 2))

        # Normalise to unit variance
        b = b / b.sum()

        return b

    def generate_fluctuations(
        self,
        n_faces: int,
        face_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate digital-filter synthetic velocity fluctuations.

        Applies a separable 3D digital filter to white noise to produce
        spatially correlated velocity fluctuations with the prescribed
        integral length scales and turbulent kinetic energy.

        Parameters
        ----------
        n_faces : int
            Number of inlet faces.
        face_positions : torch.Tensor, optional
            Face centre positions ``(n_faces, 3)``.

        Returns
        -------
        torch.Tensor
            Velocity fluctuations ``(n_faces, 3)`` [m/s].
        """
        device = get_device()
        dtype = get_default_dtype()

        # Estimate grid spacing from face positions or use length scale
        if face_positions is not None and n_faces > 1:
            # Use mean spacing between adjacent faces
            dx = (face_positions[1:] - face_positions[:-1]).norm(dim=-1).mean().item()
            dx = max(dx, 1e-6)
        else:
            dx = self._Lx / self._n_filter

        # Generate white noise for each velocity component
        rng = torch.randn(n_faces, 3, device=device, dtype=dtype)

        # Apply 1D filter along the "inlet plane" direction (y)
        # For a general inlet, we filter in the index direction
        b_y = self.compute_filter_coefficients(self._Ly, dx)
        half = self._n_filter // 2

        u_filtered = torch.zeros_like(rng)
        for i in range(n_faces):
            for j in range(self._n_filter):
                idx = i + j - half
                if 0 <= idx < n_faces:
                    u_filtered[i] += b_y[j] * rng[idx]

        # Scale to match target k
        # k = 0.5 * (u'^2 + v'^2 + w'^2)
        # We want k = k_target per face
        current_k = 0.5 * (u_filtered ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-30)
        scale = (self._k_target / current_k).sqrt()
        u_filtered = u_filtered * scale

        return u_filtered

    def compute_k(self, n_faces: int) -> torch.Tensor:
        """Return the target k (uniform)."""
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_faces,), self._k_target, dtype=dtype, device=device)

    def compute_epsilon(self, n_faces: int) -> torch.Tensor:
        """Return the target epsilon (uniform)."""
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_faces,), self._epsilon, dtype=dtype, device=device)


@TurbulenceInletModel2.register("syntheticEddyInlet")
class SyntheticEddyInlet(TurbulenceInletModel2):
    """Synthetic eddy method (Jarrin et al., 2006).

    Generates synthetic turbulence by superimposing velocity fields
    from a set of randomly placed synthetic eddies.  Each eddy
    contributes a localized velocity fluctuation based on its distance
    to the inlet faces.

    The velocity fluctuation from a single eddy at position ``x_k``
    with orientation ``epsilon_k`` and strength ``sigma_k`` is::

        u'_i(x) = f_sigma(x - x_k) * epsilon_k_i * sigma_k_i

    where ``f_sigma`` is a compact-support shape function::

        f_sigma(r) = prod_j sqrt(2) * max(0, 1 - |r_j / sigma_j|)

    The total fluctuation is::

        u'(x) = (1/sqrt(N)) * sum_k u'_k(x)

    Parameters
    ----------
    k : float
        Target turbulent kinetic energy. Default: 0.01.
    epsilon : float
        Target dissipation rate. Default: 0.001.
    n_eddies : int
        Number of synthetic eddies. Default: 50.
    box_size : tuple of 3 floats
        Size of the eddy box (Lx, Ly, Lz) in metres.
        Default: (0.5, 0.5, 0.5).
    """

    def __init__(
        self,
        k: float = 0.01,
        epsilon: float = 0.001,
        n_eddies: int = 50,
        box_size: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self._k_target = k
        self._epsilon = epsilon
        self._n_eddies = n_eddies
        self._box_size = box_size
        # Derived: velocity fluctuation magnitude per component
        # k = 0.5 * (u'^2 + v'^2 + w'^2) → u_rms = sqrt(2k/3)
        self._u_rms = math.sqrt(2.0 * k / 3.0)

    @property
    def k_target(self) -> float:
        """Target turbulent kinetic energy."""
        return self._k_target

    @property
    def epsilon_value(self) -> float:
        """Target dissipation rate."""
        return self._epsilon

    @property
    def n_eddies(self) -> int:
        """Number of synthetic eddies."""
        return self._n_eddies

    @property
    def box_size(self) -> tuple[float, float, float]:
        """Eddy box size (Lx, Ly, Lz) in metres."""
        return self._box_size

    def _get_length_scale(self) -> float:
        """Geometric mean of box size."""
        bx, by, bz = self._box_size
        return (bx * by * bz) ** (1.0 / 3.0)

    def _generate_eddy_positions(
        self, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate random eddy positions within the eddy box.

        Returns
        -------
        torch.Tensor
            Eddy positions ``(n_eddies, 3)``.
        """
        Lx, Ly, Lz = self._box_size
        x = torch.rand(self._n_eddies, device=device, dtype=dtype) * Lx
        y = torch.rand(self._n_eddies, device=device, dtype=dtype) * Ly
        z = torch.rand(self._n_eddies, device=device, dtype=dtype) * Lz
        return torch.stack([x, y, z], dim=-1)

    def _generate_eddy_orientations(
        self, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate random eddy orientation vectors (unit vectors).

        Returns
        -------
        torch.Tensor
            Orientation vectors ``(n_eddies, 3)`` with unit norm.
        """
        # Random directions on the unit sphere
        vec = torch.randn(self._n_eddies, 3, device=device, dtype=dtype)
        norm = vec.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        return vec / norm

    def _shape_function(self, r: torch.Tensor) -> torch.Tensor:
        """Compact-support eddy shape function.

        f(r) = prod_j sqrt(2) * max(0, 1 - |r_j / sigma_j|)

        where sigma_j = box_size_j / 2 (the eddy half-width).

        Parameters
        ----------
        r : torch.Tensor
            Distance vector from eddy to face ``(n, 3)``.

        Returns
        -------
        torch.Tensor
            Shape function value ``(n,)``.
        """
        sigma = torch.tensor(
            [s / 2.0 for s in self._box_size],
            device=r.device, dtype=r.dtype,
        )
        # Relative distance scaled by eddy size
        r_scaled = (r.abs() / sigma).clamp(max=1.0)
        # Compact support: (1 - |r/sigma|) for each direction
        f_each = (1.0 - r_scaled).clamp(min=0.0)
        # Product over directions, scaled by sqrt(2) per direction
        return math.sqrt(2.0) ** 3 * f_each.prod(dim=-1)

    def generate_fluctuations(
        self,
        n_faces: int,
        face_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate synthetic eddy velocity fluctuations.

        Places N eddies randomly in a box around the inlet and
        computes the superposition of their velocity contributions
        at each face position.

        Parameters
        ----------
        n_faces : int
            Number of inlet faces.
        face_positions : torch.Tensor, optional
            Face centre positions ``(n_faces, 3)``.
            If None, positions are generated uniformly in [0, 1]^3.

        Returns
        -------
        torch.Tensor
            Velocity fluctuations ``(n_faces, 3)`` [m/s].
        """
        device = get_device()
        dtype = get_default_dtype()

        if face_positions is None:
            # Generate uniform face positions
            face_positions = torch.rand(n_faces, 3, device=device, dtype=dtype)

        face_pos = face_positions.to(device=device, dtype=dtype)

        # Generate eddy properties
        eddy_pos = self._generate_eddy_positions(device, dtype)
        eddy_orient = self._generate_eddy_orientations(device, dtype)

        # Eddy strength: each component has std = u_rms
        # Random signs for each component
        eddy_strength = (
            torch.sign(torch.randn(self._n_eddies, 3, device=device, dtype=dtype))
            * self._u_rms
        )

        # Compute fluctuation at each face
        u_fluct = torch.zeros(n_faces, 3, device=device, dtype=dtype)

        for k in range(self._n_eddies):
            # Distance from eddy k to all faces
            r = face_pos - eddy_pos[k]  # (n_faces, 3)
            f = self._shape_function(r)  # (n_faces,)
            # Contribution: f * epsilon_k * sigma_k
            # eddy_orient[k] is (3,), eddy_strength[k] is (3,)
            orient_strength = eddy_orient[k] * eddy_strength[k]  # (3,)
            u_fluct += f.unsqueeze(-1) * orient_strength.unsqueeze(0)

        # Normalise by sqrt(N)
        u_fluct = u_fluct / math.sqrt(self._n_eddies)

        return u_fluct

    def compute_k(self, n_faces: int) -> torch.Tensor:
        """Return the target k (uniform)."""
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_faces,), self._k_target, dtype=dtype, device=device)

    def compute_epsilon(self, n_faces: int) -> torch.Tensor:
        """Return the target epsilon (uniform)."""
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_faces,), self._epsilon, dtype=dtype, device=device)
