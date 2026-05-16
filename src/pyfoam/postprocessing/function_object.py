"""
FunctionObject — base class for OpenFOAM-style function objects.

Function objects are post-processing utilities that execute at specified
intervals during a simulation.  They read fields from the solver and
produce derived quantities (forces, wall shear stress, probes, etc.).

This module provides:

- :class:`FunctionObject` — abstract base with lifecycle hooks
- :class:`FunctionObjectRegistry` — registry for discovering and creating
  function objects by name

Lifecycle
---------
1. ``__init__`` — parse dictionary configuration
2. ``initialise(mesh, fields)`` — called once at start
3. ``execute(time)`` — called each time step (or write interval)
4. ``write()`` — called to flush results to disk
5. ``finalise()`` — called at end of simulation

Usage::

    class MyFunctionObject(FunctionObject):
        def initialise(self, mesh, fields):
            ...

        def execute(self, time):
            ...

    # Register
    FunctionObjectRegistry.register("myFunctionObject", MyFunctionObject)

    # Create from dict
    fo = FunctionObjectRegistry.create("myFunctionObject", {"name": "myFO", ...})
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch

__all__ = ["FunctionObject", "FunctionObjectRegistry"]

logger = logging.getLogger(__name__)


class FunctionObject(ABC):
    """Abstract base class for function objects.

    Subclasses must implement :meth:`initialise`, :meth:`execute`, and
    optionally :meth:`write` and :meth:`finalise`.

    Parameters
    ----------
    name : str
        Name of this function object instance (for output directories).
    config : dict
        Dictionary configuration (from ``controlDict`` or ``functionObjectDict``).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self._mesh = None
        self._fields: Dict[str, Any] = {}
        self._output_path: Optional[Path] = None
        self._enabled = self.config.get("enabled", True)
        self._log = logging.getLogger(f"{__name__}.{name}")

    @property
    def mesh(self):
        """The FvMesh (available after :meth:`initialise`)."""
        return self._mesh

    @property
    def fields(self) -> Dict[str, Any]:
        """Dictionary of available fields."""
        return self._fields

    @property
    def output_path(self) -> Optional[Path]:
        """Output directory for this function object."""
        return self._output_path

    def set_output_path(self, path: Path) -> None:
        """Set the output directory and create it if needed."""
        self._output_path = path
        self._output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Called once at the start of the simulation.

        Args:
            mesh: The :class:`~pyfoam.mesh.fv_mesh.FvMesh`.
            fields: Dictionary mapping field names to field objects.
        """
        ...

    @abstractmethod
    def execute(self, time: float) -> None:
        """Called at each output time step.

        Args:
            time: Current simulation time.
        """
        ...

    def write(self) -> None:
        """Flush results to disk.  Default is no-op."""
        pass

    def finalise(self) -> None:
        """Called at the end of the simulation.  Default is no-op."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self._enabled})"


class FunctionObjectRegistry:
    """Registry for discovering and creating function objects by type name.

    Usage::

        FunctionObjectRegistry.register("forces", Forces)
        fo = FunctionObjectRegistry.create("forces", {"name": "forces1", ...})
    """

    _registry: Dict[str, Type[FunctionObject]] = {}

    @classmethod
    def register(cls, type_name: str, fo_class: Type[FunctionObject]) -> None:
        """Register a function object class under *type_name*.

        Args:
            type_name: OpenFOAM-style name (e.g. ``"forces"``, ``"yPlus"``).
            fo_class: The class to register.

        Raises:
            TypeError: If *fo_class* is not a subclass of :class:`FunctionObject`.
        """
        if not (isinstance(fo_class, type) and issubclass(fo_class, FunctionObject)):
            raise TypeError(
                f"fo_class must be a subclass of FunctionObject, got {fo_class}"
            )
        cls._registry[type_name] = fo_class
        logger.debug("Registered function object '%s' -> %s", type_name, fo_class.__name__)

    @classmethod
    def create(cls, type_name: str, config: Dict[str, Any]) -> FunctionObject:
        """Create a function object instance from *type_name* and *config*.

        Args:
            type_name: Registered type name.
            config: Dictionary configuration (must include ``"name"`` key).

        Returns:
            A new :class:`FunctionObject` instance.

        Raises:
            KeyError: If *type_name* is not registered.
        """
        if type_name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Unknown function object type '{type_name}'. "
                f"Available: {available}"
            )
        fo_class = cls._registry[type_name]
        name = config.get("name", type_name)
        return fo_class(name=name, config=config)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return list of registered type names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (for testing)."""
        cls._registry.clear()
