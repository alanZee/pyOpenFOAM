"""
Enstrophy — standalone enstrophy computation function object.

Computes the enstrophy field ε = 0.5 * |ω|², where ω = ∇ × U is the
vorticity.

Physics
-------
Enstrophy is a scalar measure of the intensity of vorticity:

    ε = 0.5 * |ω|² = 0.5 * (ωx² + ωy² + ωz²)

It represents the rotational kinetic energy per unit volume and is
important in turbulence research. In 3D turbulence, enstrophy is
dissipated by viscosity at the Kolmogorov scale.

References
----------
- Tennekes & Lumley, "A First Course in Turbulence", MIT Press
- OpenFOAM ``enstrophy`` function object source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["Enstrophy"]

logger = logging.getLogger(__name__)


class Enstrophy(FunctionObject):
    """Compute enstrophy (0.5 * |vorticity|²) on the mesh.

    Configuration keys:

    - ``field``: velocity field name (default: ``"U"``)
    - ``writeField``: if True, write ε to time directories (default: False)

    Example controlDict entry::

        enstrophy1
        {
            type            enstrophy;
            libs            ("libfieldFunctionObjects.so");
            field           U;
            writeField      true;
        }

    Output shape: ``(n_cells,)`` — one scalar enstrophy value per cell.
    """

    def __init__(self, name: str = "enstrophy", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_name: str = self.config.get("field", "U")
        self._write_field: bool = self.config.get("writeField", False)

        # Results
        self._enstrophy: Optional[torch.Tensor] = None
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info("Enstrophy '%s' initialised: field=%s", self.name, self._field_name)

    def execute(self, time: float) -> None:
        """Compute enstrophy at current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get(self._field_name)
        if U is None:
            logger.warning("Field '%s' required for Enstrophy. Skipping.", self._field_name)
            return

        eps = self._compute(U)
        self._enstrophy = eps.detach().cpu()
        self._times.append(time)

        self._log.info(
            "t=%g  enstrophy computed  eps_avg=%.6e  eps_max=%.6e",
            time, eps.mean().item(), eps.max().item()
        )

    def _compute(self, U_field) -> torch.Tensor:
        """Compute enstrophy ε = 0.5 * |∇ × U|².

        Args:
            U_field: Velocity field (tensor or GeometricField).

        Returns:
            ``(n_cells,)`` scalar enstrophy field.
        """
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh
        n_cells = mesh.n_cells

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        # Compute vorticity via curl
        from pyfoam.discretisation.operators import fvc

        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for j in range(3):
            grad_U[:, j, :] = fvc.grad(U_data[:, j], mesh=mesh)

        omega = torch.zeros_like(U_data)
        omega[:, 0] = grad_U[:, 2, 1] - grad_U[:, 1, 2]
        omega[:, 1] = grad_U[:, 0, 2] - grad_U[:, 2, 0]
        omega[:, 2] = grad_U[:, 1, 0] - grad_U[:, 0, 1]

        # Enstrophy
        enstrophy = 0.5 * (omega * omega).sum(dim=1)

        return enstrophy

    @property
    def enstrophy_field(self) -> Optional[torch.Tensor]:
        """Computed enstrophy field ``(n_cells,)``."""
        return self._enstrophy

    @property
    def times(self) -> List[float]:
        """Time values."""
        return self._times

    def write(self) -> None:
        """Write enstrophy data to output files."""
        if self._output_path is None or self._enstrophy is None:
            return

        info_file = self._output_path / "enstrophy.info"
        with open(info_file, "w") as f:
            f.write("# Enstrophy (0.5 * |vorticity|^2)\n")
            f.write(f"# Field: {self._field_name}\n")
            f.write(f"# Shape: {self._enstrophy.shape}\n")
            f.write(f"# eps_avg: {self._enstrophy.mean().item():.6e}\n")
            f.write(f"# eps_max: {self._enstrophy.max().item():.6e}\n")
            f.write(f"# Times computed: {len(self._times)}\n")
        logger.info("Wrote enstrophy info to %s", info_file)


# Register
FunctionObjectRegistry.register("enstrophy", Enstrophy)
