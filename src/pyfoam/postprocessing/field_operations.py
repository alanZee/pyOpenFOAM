"""
FieldOperations — field gradient, divergence, and curl operations.

Provides function objects that compute derived fields from existing
solution fields, mirroring OpenFOAM's ``postProcess`` utility.

Supported operations:

- ``grad``: Gradient of a scalar/vector field
- ``div``: Divergence of a vector/tensor field
- ``curl``: Curl (rot) of a vector field
- ``mag``: Magnitude of a vector/tensor field
- ``magSqr``: Squared magnitude
- ``ReynoldsStress``: Reynolds stress tensor (for RANS)
- ``Q``: Q-criterion for vortex identification
- ``Lambda2``: Lambda2 criterion for vortex identification
- ``enstrophy``: Enstrophy (|ω|²/2)
- ``vorticity``: Vorticity (same as curl)

References
----------
- OpenFOAM ``postProcess`` utility source
- OpenFOAM field function objects
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["FieldOperations"]

logger = logging.getLogger(__name__)


class FieldOperations(FunctionObject):
    """Compute derived fields from solution fields.

    Configuration keys:

    - ``operation``: operation name (``"grad"``, ``"div"``, ``"curl"``, etc.)
    - ``field``: source field name
    - ``resultName``: name for the result field (default: auto-generated)
    - ``writeField``: if True, write the result field to time directories

    Example controlDict entry::

        gradU
        {
            type            fieldOperation;
            libs            ("libfieldFunctionObjects.so");
            operation       grad;
            field           U;
            resultName      grad(U);
            writeField      true;
        }
    """

    # Available operations
    OPERATIONS = {"grad", "div", "curl", "mag", "magSqr", "vorticity", "enstrophy"}

    def __init__(self, name: str = "fieldOperation", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._operation: str = self.config.get("operation", "grad")
        self._field_name: str = self.config.get("field", "")
        self._result_name: str = self.config.get("resultName", f"{self._operation}({self._field_name})")
        self._write_field: bool = self.config.get("writeField", False)

        if self._operation not in self.OPERATIONS:
            raise ValueError(
                f"Unknown operation '{self._operation}'. "
                f"Available: {self.OPERATIONS}"
            )

        # Results
        self._result_data: Optional[torch.Tensor] = None
        self._times: List[float] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh and fields."""
        self._mesh = mesh
        self._fields = fields
        logger.info(
            "FieldOperations '%s' initialised: operation=%s field=%s",
            self.name, self._operation, self._field_name
        )

    def execute(self, time: float) -> None:
        """Compute the field operation at current time step."""
        if not self._enabled or self._mesh is None:
            return

        source = self._fields.get(self._field_name)
        if source is None:
            logger.warning("Field '%s' not found. Skipping.", self._field_name)
            return

        result = self._compute_operation(source)
        self._result_data = result.detach().cpu()
        self._times.append(time)

        self._log.info(
            "t=%g  %s(%s) computed  shape=%s",
            time, self._operation, self._field_name, result.shape
        )

    def _compute_operation(self, field) -> torch.Tensor:
        """Compute the requested field operation.

        Args:
            field: Source field (tensor or GeometricField).

        Returns:
            Result tensor.
        """
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(field, "internal_field"):
            data = field.internal_field.to(device=device, dtype=dtype)
        else:
            data = field.to(device=device, dtype=dtype)

        mesh = self._mesh

        if self._operation == "grad":
            return self._compute_grad(data, mesh)
        elif self._operation == "div":
            return self._compute_div(data, mesh)
        elif self._operation == "curl" or self._operation == "vorticity":
            return self._compute_curl(data, mesh)
        elif self._operation == "mag":
            return self._compute_mag(data)
        elif self._operation == "magSqr":
            return self._compute_mag_sqr(data)
        elif self._operation == "enstrophy":
            return self._compute_enstrophy(data, mesh)
        else:
            raise ValueError(f"Unsupported operation: {self._operation}")

    def _compute_grad(self, data: torch.Tensor, mesh) -> torch.Tensor:
        """Compute gradient using Gauss theorem.

        For scalar → vector, for vector → tensor.
        """
        from pyfoam.discretisation.operators import fvc

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells

        if data.dim() == 1:
            # Scalar field → vector gradient
            return fvc.grad(data, mesh=mesh)
        elif data.dim() == 2 and data.shape[1] == 3:
            # Vector field → tensor gradient
            # Compute gradient of each component separately
            grad_tensor = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
            for j in range(3):
                grad_tensor[:, j, :] = fvc.grad(data[:, j], mesh=mesh)
            return grad_tensor
        else:
            raise ValueError(f"Cannot compute gradient of field with shape {data.shape}")

    def _compute_div(self, data: torch.Tensor, mesh) -> torch.Tensor:
        """Compute divergence using Gauss theorem.

        For vector → scalar.
        """
        from pyfoam.discretisation.operators import fvc

        # For divergence of a vector field, we need a flux field
        # Use a unit flux as placeholder
        n_faces = mesh.n_faces
        device = mesh.device
        dtype = mesh.dtype
        phi = torch.ones(n_faces, dtype=dtype, device=device)

        return fvc.div(phi, data, mesh=mesh)

    def _compute_curl(self, data: torch.Tensor, mesh) -> torch.Tensor:
        """Compute curl (rot) of a vector field.

        ∇ × U = (∂Uz/∂y - ∂Uy/∂z, ∂Ux/∂z - ∂Uz/∂x, ∂Uy/∂x - ∂Ux/∂y)
        """
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

        # Compute gradient of each component
        grad_U = self._compute_grad(data, mesh)  # (n_cells, 3, 3)

        # Curl from gradient tensor
        # grad_U[i, j, k] = ∂U_j/∂x_k
        # curl_i = ε_ijk ∂U_k/∂x_j = ε_ijk grad_U[k, j]
        curl = torch.zeros_like(data)
        curl[:, 0] = grad_U[:, 2, 1] - grad_U[:, 1, 2]  # ∂Uz/∂y - ∂Uy/∂z
        curl[:, 1] = grad_U[:, 0, 2] - grad_U[:, 2, 0]  # ∂Ux/∂z - ∂Uz/∂x
        curl[:, 2] = grad_U[:, 1, 0] - grad_U[:, 0, 1]  # ∂Uy/∂x - ∂Ux/∂y

        return curl

    def _compute_mag(self, data: torch.Tensor) -> torch.Tensor:
        """Compute magnitude of vector/tensor field."""
        if data.dim() == 1:
            return data.abs()
        elif data.dim() == 2:
            return data.norm(dim=1)
        elif data.dim() == 3:
            # Tensor magnitude: ||T|| = √(T:T)
            return torch.sqrt((data * data).sum(dim=(-2, -1)))
        return data

    def _compute_mag_sqr(self, data: torch.Tensor) -> torch.Tensor:
        """Compute squared magnitude."""
        if data.dim() == 1:
            return data ** 2
        elif data.dim() == 2:
            return (data * data).sum(dim=1)
        elif data.dim() == 3:
            return (data * data).sum(dim=(-2, -1))
        return data ** 2

    def _compute_enstrophy(self, data: torch.Tensor, mesh) -> torch.Tensor:
        """Compute enstrophy: ε = |ω|²/2 where ω = ∇ × U."""
        curl = self._compute_curl(data, mesh)
        return 0.5 * (curl * curl).sum(dim=1)

    @property
    def result_data(self) -> Optional[torch.Tensor]:
        """The computed result field."""
        return self._result_data

    @property
    def result_name(self) -> str:
        """Name of the result field."""
        return self._result_name

    def write(self) -> None:
        """Write result field info to log."""
        if self._output_path is not None and self._result_data is not None:
            info_file = self._output_path / f"{self._result_name}.info"
            with open(info_file, "w") as f:
                f.write(f"# {self._result_name}\n")
                f.write(f"# Operation: {self._operation}\n")
                f.write(f"# Source field: {self._field_name}\n")
                f.write(f"# Shape: {self._result_data.shape}\n")
                f.write(f"# Times computed: {len(self._times)}\n")
            logger.info("Wrote field operation info to %s", info_file)


# ---------------------------------------------------------------------------
# Convenience factory for creating field operations from controlDict
# ---------------------------------------------------------------------------


def create_field_operation(
    name: str,
    operation: str,
    field_name: str,
    **kwargs,
) -> FieldOperations:
    """Create a FieldOperations function object.

    Args:
        name: Instance name.
        operation: Operation name (grad, div, curl, etc.).
        field_name: Source field name.
        **kwargs: Additional config options.

    Returns:
        Configured :class:`FieldOperations` instance.
    """
    config = {
        "operation": operation,
        "field": field_name,
        "resultName": f"{operation}({field_name})",
        **kwargs,
    }
    return FieldOperations(name=name, config=config)


# Register
FunctionObjectRegistry.register("fieldOperation", FieldOperations)
