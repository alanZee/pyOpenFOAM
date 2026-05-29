"""
Vorticity — standalone vorticity computation function object.

Computes the vorticity field ω = ∇ × U from the velocity field.

Physics
-------
Vorticity is defined as the curl of the velocity field:

    ω = ∇ × U = (∂Uz/∂y - ∂Uy/∂z,
                  ∂Ux/∂z - ∂Uz/∂x,
                  ∂Uy/∂x - ∂Ux/∂y)

The gradient is computed via the Gauss (divergence) theorem using face
integrals, consistent with OpenFOAM's finite-volume discretisation.

References
----------
- OpenFOAM ``vorticity`` function object source
- Panton, "Incompressible Flow", 4th ed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["Vorticity"]

logger = logging.getLogger(__name__)


class Vorticity(FunctionObject):
    """Compute vorticity (curl of velocity) on the mesh.

    Configuration keys:

    - ``field``: velocity field name (default: ``"U"``)
    - ``writeField``: if True, write ω to time directories (default: False)

    Example controlDict entry::

        vorticity1
        {
            type            vorticity;
            libs            ("libfieldFunctionObjects.so");
            field           U;
            writeField      true;
        }

    Output shape: ``(n_cells, 3)`` — one vorticity vector per cell.
    """

    def __init__(self, name: str = "vorticity", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_name: str = self.config.get("field", "U")
        self._write_field: bool = self.config.get("writeField", False)

        # Results
        self._omega: Optional[torch.Tensor] = None
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info("Vorticity '%s' initialised: field=%s", self.name, self._field_name)

    def execute(self, time: float) -> None:
        """Compute vorticity at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get(self._field_name)
        if U is None:
            logger.warning("Field '%s' required for Vorticity. Skipping.", self._field_name)
            return

        omega = self._compute(U)
        self._omega = omega.detach().cpu()
        self._times.append(time)

        omega_mag = omega.norm(dim=1)
        self._log.info(
            "t=%g  vorticity computed  |omega|_avg=%.6e  |omega|_max=%.6e",
            time, omega_mag.mean().item(), omega_mag.max().item()
        )

    def _compute(self, U_field) -> torch.Tensor:
        """Compute vorticity ω = ∇ × U using finite-volume gradient.

        Args:
            U_field: Velocity field (tensor or GeometricField).

        Returns:
            ``(n_cells, 3)`` vorticity tensor.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh
        n_cells = mesh.n_cells

        # Extract data
        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        # Compute gradient of each velocity component via fvc::grad
        from pyfoam.discretisation.operators import fvc

        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for j in range(3):
            grad_U[:, j, :] = fvc.grad(U_data[:, j], mesh=mesh)

        # Curl from gradient tensor: curl_i = ε_ijk ∂U_k/∂x_j
        omega = torch.zeros_like(U_data)
        omega[:, 0] = grad_U[:, 2, 1] - grad_U[:, 1, 2]
        omega[:, 1] = grad_U[:, 0, 2] - grad_U[:, 2, 0]
        omega[:, 2] = grad_U[:, 1, 0] - grad_U[:, 0, 1]

        return omega

    @property
    def omega(self) -> Optional[torch.Tensor]:
        """Computed vorticity field ``(n_cells, 3)``."""
        return self._omega

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times

    def write(self) -> None:
        """Write vorticity data to output files."""
        if self._output_path is None or self._omega is None:
            return

        info_file = self._output_path / "vorticity.info"
        with open(info_file, "w") as f:
            f.write("# Vorticity (curl of velocity)\n")
            f.write(f"# Field: {self._field_name}\n")
            f.write(f"# Shape: {self._omega.shape}\n")
            f.write(f"# Times computed: {len(self._times)}\n")
        logger.info("Wrote vorticity info to %s", info_file)


# Register
FunctionObjectRegistry.register("vorticity", Vorticity)
