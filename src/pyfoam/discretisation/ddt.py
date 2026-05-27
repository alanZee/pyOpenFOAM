"""
Time derivative discretisation schemes (ddt).

Provides the :class:`DdtScheme` abstract base and concrete implementations
for use in finite volume solvers:

- :class:`EulerDdt` — first-order implicit Euler (backward Euler).
- :class:`SteadyStateDdt` — zero time derivative for steady-state solvers.
- :class:`CrankNicolsonDdt` — second-order Crank-Nicolson with blending.

Usage with the :class:`~pyfoam.discretisation.operators._FvmNamespace`::

    from pyfoam.discretisation.ddt import EulerDdt, SteadyStateDdt

    scheme = EulerDdt(mesh)
    mat = scheme.ddt(coeff=1.0, phi=phi_old, dt=0.001)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix

__all__ = [
    "DdtScheme",
    "EulerDdt",
    "SteadyStateDdt",
    "CrankNicolsonDdt",
    "DDT_REGISTRY",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DdtScheme(ABC):
    """Abstract base class for time derivative discretisation schemes.

    A ddt scheme produces an :class:`FvMatrix` representing the discretised
    time derivative term ``coeff * d(phi)/dt`` in a finite volume equation.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh (required for cell volumes).
    """

    def __init__(self, mesh: Any) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Any:
        """The finite volume mesh."""
        return self._mesh

    def _extract_phi(self, phi: Any) -> torch.Tensor:
        """Extract field data and resolve mesh from a GeometricField or tensor.

        Args:
            phi: A GeometricField (with ``.internal_field`` and ``.mesh``)
                 or a raw ``torch.Tensor``.

        Returns:
            The internal-field tensor.
        """
        if hasattr(phi, "internal_field"):
            return phi.internal_field
        return phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)

    @abstractmethod
    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Discretise the time derivative ``coeff * d(phi)/dt``.

        Args:
            coeff: Scalar coefficient (e.g. density ρ or 1).
            phi: Current field values ``(n_cells,)`` or ``(n_cells, 3)``.
                 May be a ``GeometricField`` (``.internal_field`` accessed)
                 or a plain ``torch.Tensor``.
            dt: Time step size.
            mesh: The ``FvMesh``.  Inferred from *phi* when it carries a
                  ``.mesh`` attribute; otherwise required.
            phi_old: Previous time-step field (used by some schemes; e.g.
                     Crank-Nicolson).  Defaults to *phi* when not given.

        Returns:
            :class:`~pyfoam.core.fv_matrix.FvMatrix` with time derivative
            coefficients.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mesh={self._mesh})"


# ---------------------------------------------------------------------------
# Euler (first-order implicit backward Euler)
# ---------------------------------------------------------------------------


class EulerDdt(DdtScheme):
    """First-order implicit Euler time derivative.

    Discretisation::

        diag_i   = coeff * V_i / dt
        source_i = coeff * V_i * phi_old_i / dt

    This is the standard first-order time scheme used by most transient
    solvers.  When *phi_old* is not provided it defaults to *phi*.
    """

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """First-order implicit Euler ddt."""
        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype
        )
        if phi_old is None:
            phi_old_data = phi_data
        else:
            phi_old_data = self._extract_phi(phi_old).to(
                device=mesh.device, dtype=mesh.dtype
            )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # Diagonal: coeff * V / dt
        mat.diag = coeff * cell_volumes / dt

        # Source: coeff * V * phi_old / dt
        if phi_old_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_old_data / dt
        else:
            mat.source = coeff * cell_volumes * phi_old_data.sum(dim=-1) / dt

        return mat


# ---------------------------------------------------------------------------
# SteadyState (zero time derivative)
# ---------------------------------------------------------------------------


class SteadyStateDdt(DdtScheme):
    """Zero time derivative — used in steady-state solvers.

    Returns an :class:`FvMatrix` with all-zero diagonal and source,
    effectively disabling the time derivative term in the equation.
    """

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Zero ddt for steady-state."""
        mesh = mesh or self._mesh

        n_cells = mesh.n_cells
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        mat.diag = torch.zeros(n_cells, dtype=mesh.dtype, device=mesh.device)
        mat.source = torch.zeros(n_cells, dtype=mesh.dtype, device=mesh.device)
        return mat


# ---------------------------------------------------------------------------
# Crank-Nicolson (second-order with blending)
# ---------------------------------------------------------------------------


class CrankNicolsonDdt(DdtScheme):
    """Second-order Crank-Nicolson time derivative with blending.

    Blends the new (current iteration) and old (previous time-step) field
    contributions to achieve second-order temporal accuracy::

        diag_i   = coeff * V_i * theta / dt
        source_i = coeff * V_i / dt *
                   [ theta * phi_old_i + (1 - theta) * phi_i ]

    where *theta* is the blending coefficient (1.0 = pure CN, 0.0 = pure
    Euler).  The typical default is *theta* = 1.0.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    theta : float, optional
        Blending coefficient in [0, 1].  Default is 1.0 (pure CN).

    Notes
    -----
    Requires *phi_old* (field at the previous completed time-step) to be
    explicitly passed to :meth:`ddt`.  A :class:`ValueError` is raised if
    *phi_old* is not provided.
    """

    def __init__(self, mesh: Any, theta: float = 1.0) -> None:
        super().__init__(mesh)
        if not 0.0 <= theta <= 1.0:
            raise ValueError(
                f"theta must be in [0, 1], got {theta}"
            )
        self._theta = theta

    @property
    def theta(self) -> float:
        """Blending coefficient."""
        return self._theta

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Second-order Crank-Nicolson ddt.

        Args:
            coeff: Scalar coefficient (e.g. density ρ or 1).
            phi: Current iteration field values.
            dt: Time step size.
            mesh: The ``FvMesh`` (inferred from *phi* when possible).
            phi_old: **Required.**  Field at the previous completed
                     time-step.

        Raises:
            ValueError: If *phi_old* is not provided.
        """
        if phi_old is None:
            raise ValueError(
                "CrankNicolsonDdt requires phi_old (previous time-step field)"
            )

        mesh = mesh or self._mesh
        theta = self._theta

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype
        )
        phi_old_data = self._extract_phi(phi_old).to(
            device=mesh.device, dtype=mesh.dtype
        )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # Diagonal: coeff * V * theta / dt
        mat.diag = coeff * cell_volumes * theta / dt

        # Source: coeff * V / dt * [theta * phi_old + (1 - theta) * phi]
        if phi_old_data.dim() == 1:
            blended = theta * phi_old_data + (1.0 - theta) * phi_data
            mat.source = coeff * cell_volumes * blended / dt
        else:
            blended = theta * phi_old_data + (1.0 - theta) * phi_data
            mat.source = coeff * cell_volumes * blended.sum(dim=-1) / dt

        return mat


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

DDT_REGISTRY: dict[str, type[DdtScheme]] = {
    "Euler": EulerDdt,
    "steadyState": SteadyStateDdt,
    "CrankNicolson": CrankNicolsonDdt,
}


def create_ddt_scheme(
    name: str,
    mesh: Any,
    **kwargs: Any,
) -> DdtScheme:
    """Create a ddt scheme from a name (case-sensitive).

    Args:
        name: Scheme name, one of ``"Euler"``, ``"steadyState"``,
              ``"CrankNicolson"``.
        mesh: The ``FvMesh``.
        **kwargs: Extra keyword arguments forwarded to the scheme
                  constructor (e.g. ``theta`` for Crank-Nicolson).

    Returns:
        A :class:`DdtScheme` instance.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in DDT_REGISTRY:
        raise ValueError(
            f"Unknown ddt scheme '{name}'. "
            f"Available: {list(DDT_REGISTRY.keys())}"
        )
    scheme_cls = DDT_REGISTRY[name]
    return scheme_cls(mesh, **kwargs)
