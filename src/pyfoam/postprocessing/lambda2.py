"""
Lambda2 — λ₂ criterion for vortex identification.

Computes the λ₂ eigenvalue criterion, a robust method for identifying
vortex cores based on the eigenvalues of a derived tensor.

Physics
-------
The velocity gradient tensor ∇U is decomposed into:

    S = 0.5 * (∇U + (∇U)^T)   (symmetric: strain-rate)
    Ω = 0.5 * (∇U - (∇U)^T)   (antisymmetric: rotation-rate)

A tensor S² + Ω² is formed. The eigenvalues λ₁ ≥ λ₂ ≥ λ₃ of this
tensor are computed at each cell. Vortex cores are identified where
λ₂ < 0 (i.e., the second eigenvalue is negative).

The λ₂ criterion is preferred over Q-criterion because it is Galilean
invariant and does not require a threshold value.

References
----------
- Jeong & Hussain, "On the identification of a vortex", J. Fluid Mech.
  285, 69-94, 1995
- OpenFOAM ``Lambda2`` function object source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["Lambda2"]

logger = logging.getLogger(__name__)


class Lambda2(FunctionObject):
    """Compute the λ₂ criterion for vortex identification.

    Configuration keys:

    - ``field``: velocity field name (default: ``"U"``)
    - ``writeField``: if True, write λ₂ to time directories (default: False)

    Example controlDict entry::

        lambda2_1
        {
            type            Lambda2;
            libs            ("libfieldFunctionObjects.so");
            field           U;
            writeField      true;
        }

    Output: ``(n_cells,)`` scalar field of λ₂ values. Cells with λ₂ < 0
    are vortex cores.
    """

    def __init__(self, name: str = "lambda2", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_name: str = self.config.get("field", "U")
        self._write_field: bool = self.config.get("writeField", False)

        # Results
        self._lambda2: Optional[torch.Tensor] = None
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info("Lambda2 '%s' initialised: field=%s", self.name, self._field_name)

    def execute(self, time: float) -> None:
        """Compute λ₂ at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get(self._field_name)
        if U is None:
            logger.warning("Field '%s' required for Lambda2. Skipping.", self._field_name)
            return

        l2 = self._compute(U)
        self._lambda2 = l2.detach().cpu()
        self._times.append(time)

        n_vortex = (l2 < 0).sum().item()
        self._log.info(
            "t=%g  lambda2 computed  l2_min=%.6e  l2_max=%.6e  vortex_cells=%d",
            time, l2.min().item(), l2.max().item(), n_vortex
        )

    def _compute(self, U_field) -> torch.Tensor:
        """Compute λ₂ eigenvalue criterion.

        Args:
            U_field: Velocity field (tensor or GeometricField).

        Returns:
            ``(n_cells,)`` tensor of λ₂ values.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh
        n_cells = mesh.n_cells

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        # Compute velocity gradient
        from pyfoam.discretisation.operators import fvc

        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for j in range(3):
            grad_U[:, j, :] = fvc.grad(U_data[:, j], mesh=mesh)

        # Symmetric and antisymmetric parts
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))
        Omega = 0.5 * (grad_U - grad_U.transpose(-1, -2))

        # S² + Ω²
        A = torch.bmm(S, S) + torch.bmm(Omega, Omega)

        # Eigenvalues (sorted descending)
        eigenvalues = torch.linalg.eigvalsh(A)  # (n_cells, 3), ascending order
        # λ₁ ≥ λ₂ ≥ λ₃ → λ₂ is index 1 when descending
        # eigvalsh returns ascending, so index 1 is the middle eigenvalue
        l2 = eigenvalues[:, 1]

        return l2

    @property
    def lambda2_field(self) -> Optional[torch.Tensor]:
        """Computed λ₂ field ``(n_cells,)``."""
        return self._lambda2

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times

    def vortex_cells(self) -> Optional[torch.Tensor]:
        """Indices of cells identified as vortex cores (λ₂ < 0)."""
        if self._lambda2 is None:
            return None
        return (self._lambda2 < 0).nonzero(as_tuple=True)[0]

    def write(self) -> None:
        """Write λ₂ data to output files."""
        if self._output_path is None or self._lambda2 is None:
            return

        info_file = self._output_path / "lambda2.info"
        with open(info_file, "w") as f:
            f.write("# Lambda2 criterion for vortex identification\n")
            f.write(f"# Field: {self._field_name}\n")
            f.write(f"# Shape: {self._lambda2.shape}\n")
            f.write(f"# lambda2_min: {self._lambda2.min().item():.6e}\n")
            f.write(f"# lambda2_max: {self._lambda2.max().item():.6e}\n")
            f.write(f"# Vortex cells (lambda2<0): {(self._lambda2 < 0).sum().item()}\n")
            f.write(f"# Times computed: {len(self._times)}\n")
        logger.info("Wrote lambda2 info to %s", info_file)


# Register
FunctionObjectRegistry.register("Lambda2", Lambda2)
