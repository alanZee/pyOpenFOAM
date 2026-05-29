"""
Contact angle models for multiphase wall boundary conditions.

The contact angle determines how a fluid interface meets a solid wall
and is critical for capillary-driven flows, droplet spreading, and
meniscus behaviour.

Models:

- :class:`ContactAngleModel` — abstract base with RTS registry
- :class:`ConstantContactAngle` — fixed contact angle
- :class:`DynamicContactAngle` — velocity-dependent contact angle
- :class:`KistlerContactAngle` — Kistler (1993) dynamic contact angle

In OpenFOAM, contact angle models are selected in the boundary
condition for the alpha field::

    wall
    {
        type            alphaContactAngle;
        theta0          90;
        dynamicContactAngle
        {
            type        Kistler;
            U_max       10;
        }
    }

Reference:
    Kistler, S.F. (1993). "Hydrodynamics of wetting." In: Berg, J.C.
    (ed.) Wettability. Marcel Dekker, New York, pp. 311-429.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "ContactAngleModel",
    "ConstantContactAngle",
    "DynamicContactAngle",
    "KistlerContactAngle",
]

logger = logging.getLogger(__name__)


class ContactAngleModel(ABC):
    """Abstract base class for contact angle models.

    Subclasses implement :meth:`compute` which returns the local
    contact angle at each wall face.

    RTS (Run-Time Selection) registry allows string-based lookup::

        model = ContactAngleModel.create("constant", theta0=90.0)
    """

    _registry: ClassVar[dict[str, Type["ContactAngleModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a contact angle model under *name*."""

        def decorator(model_cls: Type[ContactAngleModel]) -> Type[ContactAngleModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Contact angle model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "ContactAngleModel":
        """Factory: create a contact angle model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown contact angle model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute(
        self,
        U_wall: torch.Tensor,
        n_wall: torch.Tensor,
        grad_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the contact angle at each wall face.

        Parameters
        ----------
        U_wall : torch.Tensor
            ``(n_faces, 3)`` wall-adjacent velocity.
        n_wall : torch.Tensor
            ``(n_faces, 3)`` wall unit normal (pointing into fluid).
        grad_alpha : torch.Tensor
            ``(n_faces, 3)`` volume fraction gradient at the wall.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` contact angle in radians.
        """
        ...


# ======================================================================
# Constant contact angle
# ======================================================================


@ContactAngleModel.register("constant")
class ConstantContactAngle(ContactAngleModel):
    """Fixed (static) contact angle model.

    Returns a uniform contact angle regardless of flow conditions.
    This is the simplest contact angle model and is appropriate for
    equilibrium or low-Ca flows where the contact line velocity is
    negligible.

    Parameters
    ----------
    theta0 : float
        Static contact angle in degrees. Default: 90.0 (neutral wetting).
    """

    def __init__(self, theta0: float = 90.0) -> None:
        if theta0 < 0 or theta0 > 180:
            raise ValueError(f"Contact angle must be in [0, 180], got {theta0}")
        self._theta0 = theta0

    @property
    def theta0(self) -> float:
        """Static contact angle in degrees."""
        return self._theta0

    @property
    def theta0_rad(self) -> float:
        """Static contact angle in radians."""
        return math.radians(self._theta0)

    def compute(
        self,
        U_wall: torch.Tensor,
        n_wall: torch.Tensor,
        grad_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Return the constant contact angle for all faces.

        Parameters
        ----------
        U_wall : torch.Tensor
            ``(n_faces, 3)`` wall velocity (unused).
        n_wall : torch.Tensor
            ``(n_faces, 3)`` wall normal (unused).
        grad_alpha : torch.Tensor
            ``(n_faces, 3)`` alpha gradient (unused).

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` constant contact angle in radians.
        """
        n_faces = U_wall.shape[0]
        return torch.full(
            (n_faces,), self.theta0_rad, dtype=U_wall.dtype, device=U_wall.device
        )

    def __repr__(self) -> str:
        return f"ConstantContactAngle(theta0={self._theta0}°)"


# ======================================================================
# Dynamic contact angle (simple velocity-dependent)
# ======================================================================


@ContactAngleModel.register("dynamic")
class DynamicContactAngle(ContactAngleModel):
    """Simple velocity-dependent dynamic contact angle.

    Linearly interpolates between the advancing and receding contact
    angles based on the contact line velocity:

        theta_d = theta_rec + (theta_adv - theta_rec) * (V_cl / V_max + 1) / 2

    where V_cl is the component of velocity along the wall tangent
    in the direction of the interface normal projection.

    Parameters
    ----------
    theta_adv : float
        Advancing contact angle (degrees). Default: 120.0.
    theta_rec : float
        Receding contact angle (degrees). Default: 60.0.
    U_max : float
        Reference velocity for full dynamic range (m/s). Default: 1.0.
    """

    def __init__(
        self,
        theta_adv: float = 120.0,
        theta_rec: float = 60.0,
        U_max: float = 1.0,
    ) -> None:
        if theta_adv < theta_rec:
            raise ValueError(
                f"Advancing angle ({theta_adv}) must be >= receding ({theta_rec})"
            )
        self._theta_adv = theta_adv
        self._theta_rec = theta_rec
        self._U_max = max(U_max, 1e-10)

    @property
    def theta_adv(self) -> float:
        """Advancing contact angle (degrees)."""
        return self._theta_adv

    @property
    def theta_rec(self) -> float:
        """Receding contact angle (degrees)."""
        return self._theta_rec

    @property
    def U_max(self) -> float:
        """Reference velocity (m/s)."""
        return self._U_max

    def compute(
        self,
        U_wall: torch.Tensor,
        n_wall: torch.Tensor,
        grad_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity-dependent dynamic contact angle.

        Parameters
        ----------
        U_wall : torch.Tensor
            ``(n_faces, 3)`` wall velocity.
        n_wall : torch.Tensor
            ``(n_faces, 3)`` wall unit normal.
        grad_alpha : torch.Tensor
            ``(n_faces, 3)`` volume fraction gradient.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` dynamic contact angle in radians.
        """
        dtype = U_wall.dtype
        device = U_wall.device

        # Contact line velocity: projection of U_wall onto wall tangent
        # in the direction of grad_alpha projected onto the wall
        grad_tang = grad_alpha - (grad_alpha * n_wall).sum(dim=1, keepdim=True) * n_wall
        grad_tang_mag = grad_tang.norm(dim=1, keepdim=True).clamp(min=1e-30)
        grad_tang_hat = grad_tang / grad_tang_mag

        V_cl = (U_wall * grad_tang_hat).sum(dim=1)

        # Normalised velocity: [-1, 1]
        V_norm = (V_cl / self._U_max).clamp(-1.0, 1.0)

        # Linear interpolation
        t = (V_norm + 1.0) * 0.5  # maps [-1,1] to [0,1]
        theta_adv_rad = math.radians(self._theta_adv)
        theta_rec_rad = math.radians(self._theta_rec)

        theta = theta_rec_rad + (theta_adv_rad - theta_rec_rad) * t

        return theta.clamp(min=0.0, max=math.pi)

    def __repr__(self) -> str:
        return (
            f"DynamicContactAngle(theta_adv={self._theta_adv}°, "
            f"theta_rec={self._theta_rec}°, U_max={self._U_max})"
        )


# ======================================================================
# Kistler dynamic contact angle
# ======================================================================


@ContactAngleModel.register("Kistler")
class KistlerContactAngle(ContactAngleModel):
    """Kistler (1993) dynamic contact angle model.

    The Kistler model relates the dynamic contact angle to the
    capillary number Ca = mu * V_cl / sigma through the Hoffman
    function:

        theta_d = f_H(Ca + f_H^-1(theta_0))

    where f_H is the Hoffman-Voinov-Tanner law:

        f_H(x) = arccos(1 - tanh(5.16 * (x / (1 + 1.31 * x^0.99))^0.706))

    and f_H^-1 is its inverse.

    This model is widely used for dynamic wetting problems and is
    the default in many OpenFOAM simulations.

    Parameters
    ----------
    theta0 : float
        Equilibrium (static) contact angle (degrees). Default: 90.0.
    mu : float
        Dynamic viscosity of the advancing fluid (Pa·s). Default: 1e-3.
    sigma : float
        Surface tension coefficient (N/m). Default: 0.072.
    """

    def __init__(
        self,
        theta0: float = 90.0,
        mu: float = 1e-3,
        sigma: float = 0.072,
    ) -> None:
        if theta0 <= 0 or theta0 >= 180:
            raise ValueError(f"Equilibrium angle must be in (0, 180), got {theta0}")
        self._theta0 = theta0
        self.mu = mu
        self.sigma = sigma

    @property
    def theta0(self) -> float:
        """Equilibrium contact angle (degrees)."""
        return self._theta0

    @staticmethod
    def _hoffman_function(x: torch.Tensor) -> torch.Tensor:
        """Hoffman-Voinov-Tanner function f_H(x).

        f_H(x) = arccos(1 - tanh(5.16 * (x / (1 + 1.31 * x^0.99))^0.706))

        Parameters
        ----------
        x : torch.Tensor
            Input (related to capillary number).

        Returns
        -------
        torch.Tensor
            Contact angle in radians.
        """
        x_safe = x.abs()
        inner = 5.16 * (x_safe / (1.0 + 1.31 * x_safe.pow(0.99))).pow(0.706)
        cos_arg = 1.0 - inner.tanh()
        # Clamp to [-1, 1] for arccos
        cos_arg = cos_arg.clamp(-1.0, 1.0)
        return cos_arg.acos()

    @staticmethod
    def _hoffman_inverse(theta: torch.Tensor) -> torch.Tensor:
        """Approximate inverse of the Hoffman function.

        Uses a Newton-Raphson iteration to find x such that
        f_H(x) = theta.

        Parameters
        ----------
        theta : torch.Tensor
            Contact angle in radians.

        Returns
        -------
        torch.Tensor
            x value (related to capillary number).
        """
        # Initial guess using the Tanner law approximation
        cos_theta = theta.cos()
        x = torch.zeros_like(theta)

        # Newton iterations (5 iterations is sufficient)
        for _ in range(10):
            f_h = KistlerContactAngle._hoffman_function(x)
            residual = f_h - theta.abs()
            # Numerical derivative
            dx = 1e-6 * (x.abs() + 1e-8)
            f_h_plus = KistlerContactAngle._hoffman_function(x + dx)
            df = (f_h_plus - f_h) / dx
            df = df.clamp(min=1e-30)
            x = x - residual / df
            x = x.clamp(min=0.0)

        return x

    def compute(
        self,
        U_wall: torch.Tensor,
        n_wall: torch.Tensor,
        grad_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Kistler dynamic contact angle.

        Parameters
        ----------
        U_wall : torch.Tensor
            ``(n_faces, 3)`` wall velocity.
        n_wall : torch.Tensor
            ``(n_faces, 3)`` wall unit normal.
        grad_alpha : torch.Tensor
            ``(n_faces, 3)`` volume fraction gradient.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` dynamic contact angle in radians.
        """
        dtype = U_wall.dtype
        device = U_wall.device

        # Contact line velocity
        grad_tang = grad_alpha - (grad_alpha * n_wall).sum(dim=1, keepdim=True) * n_wall
        grad_tang_mag = grad_tang.norm(dim=1, keepdim=True).clamp(min=1e-30)
        grad_tang_hat = grad_tang / grad_tang_mag
        V_cl = (U_wall * grad_tang_hat).sum(dim=1)

        # Capillary number
        Ca = self.mu * V_cl.abs() / max(self.sigma, 1e-30)

        # Hoffman inverse of equilibrium angle
        theta0_tensor = torch.tensor(
            math.radians(self._theta0), dtype=dtype, device=device
        )
        x0 = self._hoffman_inverse(theta0_tensor.expand(V_cl.shape[0]))

        # Dynamic angle: f_H(Ca + x0) for advancing, -f_H(Ca + x0) for receding
        x_dyn = Ca + x0
        theta_dyn = self._hoffman_function(x_dyn)

        # Receding: angle decreases; advancing: angle increases
        theta = torch.where(
            V_cl >= 0,
            theta_dyn,
            2.0 * theta0_tensor - theta_dyn,
        )

        return theta.clamp(min=0.01, max=math.pi - 0.01)

    def __repr__(self) -> str:
        return (
            f"KistlerContactAngle(theta0={self._theta0}°, "
            f"mu={self.mu}, sigma={self.sigma})"
        )
