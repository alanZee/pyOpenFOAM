"""
Base turbulence model with RTS (Run-Time Selection) registry.

Mirrors OpenFOAM's ``turbulenceModel`` hierarchy.  All RANS models
register themselves via the ``@TurbulenceModel.register(name)`` decorator
and can be instantiated at run-time via ``TurbulenceModel.create(name, ...)``.

Usage::

    @TurbulenceModel.register("kEpsilon")
    class KEpsilonModel(TurbulenceModel):
        ...

    model = TurbulenceModel.create("kEpsilon", mesh, U, phi)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.fields.vol_fields import volScalarField, volVectorField

__all__ = ["TurbulenceModel"]


class TurbulenceModel(ABC):
    """Abstract base class for RANS turbulence models.

    Subclasses must implement :meth:`nut`, :meth:`correct`, and
    :meth:`k` (turbulent kinetic energy).

    RTS (Run-Time Selection) registry allows string-based lookup::

        model = TurbulenceModel.create("kEpsilon", mesh, U, phi)

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    """

    # Class-level RTS registry: name -> class
    _registry: ClassVar[dict[str, Type[TurbulenceModel]]] = {}

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
    ) -> None:
        self._mesh = mesh
        self._device = get_device()
        self._dtype = get_default_dtype()

        # Store velocity and flux references
        if isinstance(U, volVectorField):
            self._U = U.internal_field.to(device=self._device, dtype=self._dtype)
        else:
            self._U = U.to(device=self._device, dtype=self._dtype)

        if hasattr(phi, "internal_field"):
            self._phi = phi.internal_field.to(device=self._device, dtype=self._dtype)
        else:
            self._phi = phi.to(device=self._device, dtype=self._dtype)

        # Molecular viscosity (default for air at STP)
        self._nu: float = 1.5e-5

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a turbulence model under *name*.

        Usage::

            @TurbulenceModel.register("kEpsilon")
            class KEpsilonModel(TurbulenceModel):
                ...
        """

        def decorator(model_cls: Type[TurbulenceModel]) -> Type[TurbulenceModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        **kwargs: Any,
    ) -> TurbulenceModel:
        """Factory: create a turbulence model by registered *name*.

        Args:
            name: Registered model type name (e.g. ``"kEpsilon"``).
            mesh: The finite volume mesh.
            U: Velocity field.
            phi: Face flux.
            **kwargs: Additional model-specific arguments.

        Returns:
            Instantiated turbulence model.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](mesh, U, phi, **kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> Any:
        """The mesh."""
        return self._mesh

    @property
    def nu(self) -> float:
        """Molecular kinematic viscosity."""
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:
        self._nu = value

    @property
    def device(self) -> torch.device:
        """Device of model tensors."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of model tensors."""
        return self._dtype

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def nut(self) -> torch.Tensor:
        """Return turbulent viscosity field ``(n_cells,)``."""

    @abstractmethod
    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy field ``(n_cells,)``."""

    @abstractmethod
    def correct(self) -> None:
        """Update the turbulence model (solve transport equations)."""

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def epsilon(self) -> torch.Tensor:
        """Return turbulent dissipation rate ``(n_cells,)``.

        Default implementation returns ``C_mu * k^1.5 / delta``.
        Override in k-epsilon models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement epsilon()"
        )

    def omega(self) -> torch.Tensor:
        """Return specific dissipation rate ``(n_cells,)``.

        Default implementation returns ``epsilon / (C_mu * k)``.
        Override in k-omega models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement omega()"
        )

    def devReff(self) -> torch.Tensor:
        """Return effective deviatoric Reynolds stress ``(n_cells, 3, 3)``.

        τ_eff = ν_eff * (∇U + ∇U^T) - (2/3) k I
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement devReff()"
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"n_cells={self._mesh.n_cells})"
        )
