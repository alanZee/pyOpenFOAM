"""
Virtual mass force models for multiphase Euler-Euler flows.

Provides an abstract base class and two standard virtual mass
(force) correlations used in gas-liquid and gas-solid multiphase
simulations.  The virtual mass force accounts for the inertia of
the continuous phase displaced by accelerating dispersed-phase
particles/bubbles.

Models:

- :class:`VirtualMassModel` — abstract base with RTS registry
- :class:`ConstantVirtualMass` — constant virtual mass coefficient C_vm
- :class:`LambVirtualMass` — Lamb's inviscid virtual mass (C_vm = 0.5)

The virtual mass force per unit volume is::

    F_vm = C_vm * rho_c * alpha * (D(U_c)/Dt - D(U_d)/Dt)

where:
    - ``C_vm`` is the virtual mass coefficient
    - ``rho_c`` is the continuous-phase density
    - ``alpha`` is the dispersed-phase volume fraction
    - ``D(U)/Dt`` is the material derivative

All models register themselves via ``@VirtualMassModel.register(name)``
and can be instantiated at run-time via ``VirtualMassModel.create(name, ...)``.

Usage::

    from pyfoam.multiphase.virtual_mass_models import VirtualMassModel

    vm = VirtualMassModel.create("constant", C_vm=0.5, rho_c=1000.0)
    F = vm.compute(alpha, DUDt_c, DUDt_d)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "VirtualMassModel",
    "ConstantVirtualMass",
    "LambVirtualMass",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class VirtualMassModel(ABC):
    """Abstract base class for virtual mass force models.

    Subclasses must implement :meth:`compute` which returns the virtual
    mass force per unit volume.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @VirtualMassModel.register("constant")
        class ConstantVirtualMass(VirtualMassModel):
            ...

        vm = VirtualMassModel.create("constant", C_vm=0.5, rho_c=1000.0)
    """

    _registry: ClassVar[dict[str, Type[VirtualMassModel]]] = {}

    def __init__(self, rho_c: float) -> None:
        """
        Parameters
        ----------
        rho_c : float
            Continuous phase density (kg/m^3).
        """
        self.rho_c = rho_c

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a virtual mass model class under *name*."""

        def decorator(model_cls: Type[VirtualMassModel]) -> Type[VirtualMassModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Virtual mass model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> VirtualMassModel:
        """Factory: create a virtual mass model instance by registered *name*.

        Parameters
        ----------
        name : str
            Registered model type name.
        **kwargs
            Constructor arguments forwarded to the model class.

        Returns
        -------
        VirtualMassModel
            Instantiated virtual mass model.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown virtual mass model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered virtual mass model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        alpha: torch.Tensor,
        DUDt_c: torch.Tensor,
        DUDt_d: torch.Tensor,
    ) -> torch.Tensor:
        """Compute virtual mass force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        DUDt_c : torch.Tensor
            Material derivative of continuous-phase velocity
            ``(n_cells, 3)`` [m/s^2].
        DUDt_d : torch.Tensor
            Material derivative of dispersed-phase velocity
            ``(n_cells, 3)`` [m/s^2].

        Returns
        -------
        torch.Tensor
            Virtual mass force per unit volume ``(n_cells, 3)`` [N/m^3].
        """

    @property
    @abstractmethod
    def C_vm(self) -> float:
        """Virtual mass coefficient."""


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------


@VirtualMassModel.register("constant")
class ConstantVirtualMass(VirtualMassModel):
    """Constant virtual mass coefficient model.

    Uses a user-specified constant virtual mass coefficient ``C_vm``.
    Common values:
        - C_vm = 0.5 for spheres (standard)
        - C_vm = 1.0 for bubbles in dense packing

    The virtual mass force is::

        F_vm = C_vm * rho_c * alpha * (DU_c/Dt - DU_d/Dt)

    Parameters
    ----------
    C_vm : float
        Virtual mass coefficient (default: 0.5).
    rho_c : float
        Continuous phase density (kg/m^3).
    """

    def __init__(self, rho_c: float, C_vm: float = 0.5) -> None:
        super().__init__(rho_c)
        self._C_vm = C_vm

    @property
    def C_vm(self) -> float:
        """Virtual mass coefficient."""
        return self._C_vm

    def compute(
        self,
        alpha: torch.Tensor,
        DUDt_c: torch.Tensor,
        DUDt_d: torch.Tensor,
    ) -> torch.Tensor:
        """Compute constant-coefficient virtual mass force.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        DUDt_c : torch.Tensor
            Material derivative of continuous-phase velocity
            ``(n_cells, 3)``.
        DUDt_d : torch.Tensor
            Material derivative of dispersed-phase velocity
            ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Virtual mass force per unit volume ``(n_cells, 3)``.
        """
        # F_vm = C_vm * rho_c * alpha * (DU_c/Dt - DU_d/Dt)
        accel_diff = DUDt_c - DUDt_d
        return self._C_vm * self.rho_c * alpha.unsqueeze(-1) * accel_diff


@VirtualMassModel.register("lamb")
class LambVirtualMass(VirtualMassModel):
    """Lamb's inviscid virtual mass model.

    Based on Lamb's (1932) analytical result for the added mass of a
    sphere accelerating in an unbounded inviscid fluid::

        C_vm = 0.5

    This is the theoretical value for a single sphere and is the default
    in most OpenFOAM multiphase solvers.  The force is::

        F_vm = 0.5 * rho_c * alpha * (DU_c/Dt - DU_d/Dt)

    Parameters
    ----------
    rho_c : float
        Continuous phase density (kg/m^3).
    """

    def __init__(self, rho_c: float) -> None:
        super().__init__(rho_c)

    @property
    def C_vm(self) -> float:
        """Lamb virtual mass coefficient (always 0.5)."""
        return 0.5

    def compute(
        self,
        alpha: torch.Tensor,
        DUDt_c: torch.Tensor,
        DUDt_d: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Lamb inviscid virtual mass force.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        DUDt_c : torch.Tensor
            Material derivative of continuous-phase velocity
            ``(n_cells, 3)``.
        DUDt_d : torch.Tensor
            Material derivative of dispersed-phase velocity
            ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Virtual mass force per unit volume ``(n_cells, 3)``.
        """
        accel_diff = DUDt_c - DUDt_d
        return 0.5 * self.rho_c * alpha.unsqueeze(-1) * accel_diff
