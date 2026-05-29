"""
QCriterion — Q-criterion for vortex identification.

Computes the Q-criterion, a scalar field used to identify vortical
structures in turbulent flows.

Physics
-------
The velocity gradient tensor is decomposed into symmetric (strain-rate)
and antisymmetric (rotation-rate) parts:

    S = 0.5 * (∇U + (∇U)^T)   (strain-rate tensor)
    Ω = 0.5 * (∇U - (∇U)^T)   (rotation-rate tensor)

The Q-criterion is defined as:

    Q = 0.5 * (|Ω|² - |S|²)

Regions where Q > 0 correspond to zones where rotation dominates
strain, indicating vortex cores.

References
----------
- Hunt, Wray & Moin, "Eddies, streams, and convergence zones in
  turbulent flows", 1988
- OpenFOAM ``Q`` function object source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["QCriterion"]

logger = logging.getLogger(__name__)


class QCriterion(FunctionObject):
    """Compute the Q-criterion for vortex identification.

    Configuration keys:

    - ``field``: velocity field name (default: ``"U"``)
    - ``threshold``: Q threshold for vortex detection (default: 0.0)
    - ``writeField``: if True, write Q to time directories (default: False)

    Example controlDict entry::

        Q1
        {
            type            Q;
            libs            ("libfieldFunctionObjects.so");
            field           U;
            threshold       0.0;
            writeField      true;
        }

    Output shape: ``(n_cells,)`` — one Q value per cell.
    """

    def __init__(self, name: str = "Q", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_name: str = self.config.get("field", "U")
        self._threshold: float = float(self.config.get("threshold", 0.0))
        self._write_field: bool = self.config.get("writeField", False)

        # Results
        self._Q: Optional[torch.Tensor] = None
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info("QCriterion '%s' initialised: field=%s threshold=%g",
                     self.name, self._field_name, self._threshold)

    def execute(self, time: float) -> None:
        """Compute Q-criterion at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get(self._field_name)
        if U is None:
            logger.warning("Field '%s' required for QCriterion. Skipping.", self._field_name)
            return

        Q = self._compute(U)
        self._Q = Q.detach().cpu()
        self._times.append(time)

        n_vortex = (Q > self._threshold).sum().item()
        self._log.info(
            "t=%g  Q computed  Q_min=%.6e  Q_max=%.6e  vortex_cells=%d",
            time, Q.min().item(), Q.max().item(), n_vortex
        )

    def _compute(self, U_field) -> torch.Tensor:
        """Compute Q = 0.5 * (|Ω|² - |S|²).

        Args:
            U_field: Velocity field (tensor or GeometricField).

        Returns:
            ``(n_cells,)`` Q-criterion scalar field.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh
        n_cells = mesh.n_cells

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        # Compute velocity gradient tensor ∇U
        from pyfoam.discretisation.operators import fvc

        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for j in range(3):
            grad_U[:, j, :] = fvc.grad(U_data[:, j], mesh=mesh)

        # Strain-rate tensor: S = 0.5 * (grad_U + grad_U^T)
        # grad_U[i, j, k] = ∂U_j/∂x_k
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))

        # Rotation-rate tensor: Ω = 0.5 * (grad_U - grad_U^T)
        Omega = 0.5 * (grad_U - grad_U.transpose(-1, -2))

        # |S|² = S:S = sum of squares of all components
        S_mag_sqr = (S * S).sum(dim=(-2, -1))

        # |Ω|² = Ω:Ω
        Omega_mag_sqr = (Omega * Omega).sum(dim=(-2, -1))

        # Q = 0.5 * (|Ω|² - |S|²)
        Q = 0.5 * (Omega_mag_sqr - S_mag_sqr)

        return Q

    @property
    def Q_field(self) -> Optional[torch.Tensor]:
        """Computed Q-criterion field ``(n_cells,)``."""
        return self._Q

    @property
    def threshold(self) -> float:
        """Q threshold for vortex detection."""
        return self._threshold

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times

    def vortex_cells(self) -> Optional[torch.Tensor]:
        """Indices of cells where Q > threshold."""
        if self._Q is None:
            return None
        return (self._Q > self._threshold).nonzero(as_tuple=True)[0]

    def write(self) -> None:
        """Write Q-criterion data to output files."""
        if self._output_path is None or self._Q is None:
            return

        info_file = self._output_path / "Q.info"
        with open(info_file, "w") as f:
            f.write("# Q-criterion for vortex identification\n")
            f.write(f"# Field: {self._field_name}\n")
            f.write(f"# Threshold: {self._threshold}\n")
            f.write(f"# Shape: {self._Q.shape}\n")
            f.write(f"# Q_min: {self._Q.min().item():.6e}\n")
            f.write(f"# Q_max: {self._Q.max().item():.6e}\n")
            f.write(f"# Times computed: {len(self._times)}\n")
        logger.info("Wrote Q-criterion info to %s", info_file)


# Register
FunctionObjectRegistry.register("Q", QCriterion)
