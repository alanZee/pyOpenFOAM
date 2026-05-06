"""
RAS (Reynolds-Averaged Simulation) model wrapper.

Provides a high-level interface for creating and managing RANS turbulence
models.  The ``RASModel`` class wraps a ``TurbulenceModel`` and provides
additional functionality for integration with solvers:

- Automatic model creation from dictionary-like configuration
- Effective viscosity computation for momentum equations
- Source term computation for turbulence transport equations
- Wall function integration

Usage::

    from pyfoam.turbulence.ras_model import RASModel, RASConfig

    config = RASConfig(model_name="kEpsilon", nu=1.5e-5)
    ras = RASModel(mesh, U, phi, config)

    # Update turbulence
    ras.correct()

    # Get effective viscosity for momentum equation
    mu_eff = ras.mu_eff()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .turbulence_model import TurbulenceModel

__all__ = ["RASModel", "RASConfig"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RASConfig:
    """Configuration for RAS turbulence model.

    Attributes:
        model_name: Name of the turbulence model (e.g. ``"kEpsilon"``).
        enabled: Whether turbulence modelling is active.
        nu: Molecular kinematic viscosity.
        model_kwargs: Additional keyword arguments for the model constructor.
    """

    model_name: str = "kEpsilon"
    enabled: bool = True
    nu: float = 1.5e-5
    model_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RAS model wrapper
# ---------------------------------------------------------------------------


class RASModel:
    """RAS (Reynolds-Averaged Simulation) turbulence model wrapper.

    Wraps a ``TurbulenceModel`` instance and provides solver-friendly
    interface for momentum and turbulence equation coupling.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    U : volVectorField or torch.Tensor
        Velocity field.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    config : RASConfig, optional
        RAS configuration.  Defaults to standard k-ε.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        config: RASConfig | None = None,
    ) -> None:
        self._config = config or RASConfig()
        self._mesh = mesh
        self._device = get_device()
        self._dtype = get_default_dtype()

        # Create the underlying turbulence model
        if self._config.enabled:
            self._model = TurbulenceModel.create(
                self._config.model_name,
                mesh,
                U,
                phi,
                **self._config.model_kwargs,
            )
            self._model.nu = self._config.nu
        else:
            self._model = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> RASConfig:
        """RAS configuration."""
        return self._config

    @property
    def model(self) -> TurbulenceModel | None:
        """The underlying turbulence model (None if disabled)."""
        return self._model

    @property
    def enabled(self) -> bool:
        """Whether turbulence modelling is active."""
        return self._config.enabled and self._model is not None

    # ------------------------------------------------------------------
    # Turbulence quantities
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Turbulent viscosity ``(n_cells,)``.

        Returns zeros if turbulence is disabled.
        """
        if not self.enabled:
            return torch.zeros(
                self._mesh.n_cells,
                device=self._device,
                dtype=self._dtype,
            )
        return self._model.nut()

    def k(self) -> torch.Tensor:
        """Turbulent kinetic energy ``(n_cells,)``.

        Returns zeros if turbulence is disabled.
        """
        if not self.enabled:
            return torch.zeros(
                self._mesh.n_cells,
                device=self._device,
                dtype=self._dtype,
            )
        return self._model.k()

    def epsilon(self) -> torch.Tensor:
        """Dissipation rate ``(n_cells,)``.

        Returns zeros if turbulence is disabled.
        """
        if not self.enabled:
            return torch.zeros(
                self._mesh.n_cells,
                device=self._device,
                dtype=self._dtype,
            )
        return self._model.epsilon()

    def omega(self) -> torch.Tensor:
        """Specific dissipation rate ``(n_cells,)``.

        Returns zeros if turbulence is disabled.
        """
        if not self.enabled:
            return torch.zeros(
                self._mesh.n_cells,
                device=self._device,
                dtype=self._dtype,
            )
        return self._model.omega()

    # ------------------------------------------------------------------
    # Effective properties for momentum equation
    # ------------------------------------------------------------------

    def mu_eff(self) -> torch.Tensor:
        """Effective viscosity (molecular + turbulent).

        μ_eff = ν + ν_t

        Returns:
            ``(n_cells,)`` effective kinematic viscosity.
        """
        return self._config.nu + self.nut()

    def mu_eff_field(self) -> torch.Tensor:
        """Effective viscosity as a field (same as mu_eff).

        Alias for compatibility with OpenFOAM-style interfaces.
        """
        return self.mu_eff()

    # ------------------------------------------------------------------
    # Model update
    # ------------------------------------------------------------------

    def correct(self) -> None:
        """Update the turbulence model.

        Solves transport equations and updates turbulent viscosity.
        Does nothing if turbulence is disabled.
        """
        if self.enabled:
            self._model.correct()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.enabled:
            return (
                f"RASModel(model='{self._config.model_name}', "
                f"n_cells={self._mesh.n_cells}, enabled=True)"
            )
        return f"RASModel(enabled=False)"
