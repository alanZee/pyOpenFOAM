"""
Time derivative discretisation schemes (ddt).

Provides the :class:`DdtScheme` abstract base and concrete implementations
for use in finite volume solvers:

- :class:`EulerDdt` — first-order implicit Euler (backward Euler).
- :class:`SteadyStateDdt` — zero time derivative for steady-state solvers.
- :class:`CrankNicolsonDdt` — second-order Crank-Nicolson with blending.
- :class:`BackwardDdt` — second-order backward differencing (three time levels).
- :class:`BoundedDdt` — bounded Euler with Co-ratio limiting.

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
    "BackwardDdt",
    "BackwardDdt2",
    "BoundedDdt",
    "BoundedDdt2",
    "BackwardDdt3",
    "BackwardDdt4",
    "BackwardDdt5",
    "BoundedDdt3",
    "BoundedDdt4",
    "BoundedDdt5",
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
# Backward second-order (three time levels)
# ---------------------------------------------------------------------------


class BackwardDdt(DdtScheme):
    """Second-order backward time derivative (BDF2).

    Discretisation using three time levels::

        d(phi)/dt ≈ (3*phi - 4*phi_old + phi_old2) / (2*dt)

    The resulting FvMatrix coefficients::

        diag_i   = 3 * coeff * V_i / (2 * dt)
        source_i = coeff * V_i / (2 * dt) * (4 * phi_old_i - phi_old2_i)

    This scheme requires two previous time-step fields: *phi_old* (n-1)
    and *phi_old2* (n-2).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.

    Notes
    -----
    Requires *phi_old* and *phi_old2* to be explicitly passed.
    A :class:`ValueError` is raised if either is missing.
    """

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
        phi_old2: Any = None,
    ) -> FvMatrix:
        """Second-order backward ddt.

        Args:
            coeff: Scalar coefficient (e.g. density ρ or 1).
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh`` (inferred from *phi* when possible).
            phi_old: **Required.**  Field at previous time-step (n-1).
            phi_old2: **Required.**  Field at two time-steps ago (n-2).

        Raises:
            ValueError: If *phi_old* or *phi_old2* is not provided.
        """
        if phi_old is None:
            raise ValueError(
                "BackwardDdt requires phi_old (previous time-step field)"
            )
        if phi_old2 is None:
            raise ValueError(
                "BackwardDdt requires phi_old2 (two time-steps ago field)"
            )

        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old_data = self._extract_phi(phi_old).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old2_data = self._extract_phi(phi_old2).to(
            device=mesh.device, dtype=mesh.dtype,
        )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # Diagonal: 3 * coeff * V / (2 * dt)
        mat.diag = 3.0 * coeff * cell_volumes / (2.0 * dt)

        # Source: coeff * V / (2*dt) * (4*phi_old - phi_old2)
        if phi_old_data.dim() == 1:
            source_val = 4.0 * phi_old_data - phi_old2_data
            mat.source = coeff * cell_volumes * source_val / (2.0 * dt)
        else:
            source_val = 4.0 * phi_old_data - phi_old2_data
            mat.source = (
                coeff * cell_volumes * source_val.sum(dim=-1) / (2.0 * dt)
            )

        return mat


# ---------------------------------------------------------------------------
# Bounded Euler (with Co-ratio limiting)
# ---------------------------------------------------------------------------


class BoundedDdt(DdtScheme):
    r"""Bounded Euler time derivative with Courant-number-based limiting.

    A variant of :class:`EulerDdt` that applies a bounding factor
    based on the ratio of the local Courant number to a reference
    Courant number:

    .. math::

        \text{diag}_i = \frac{\text{coeff} \cdot V_i}{\Delta t}
            \cdot \min\left(1, \frac{\text{Co}_{\text{ref}}}{\text{Co}_i}\right)

    When the local Courant number exceeds *Co_ref*, the diagonal is
    reduced (implicit blending toward steady-state), which helps
    prevent divergence in regions with very high local Co.

    If *face_flux* is not provided, the scheme degenerates to standard
    Euler without limiting.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    Co_ref : float, optional
        Reference Courant number for limiting.  Default is 1.0.
    face_flux : torch.Tensor, optional
        Face flux values ``(n_faces,)``.  When provided, used to
        compute the local Courant number per cell.
    """

    def __init__(
        self, mesh: Any, Co_ref: float = 1.0,
        face_flux: Any = None,
    ) -> None:
        super().__init__(mesh)
        if Co_ref <= 0.0:
            raise ValueError(
                f"Co_ref must be positive, got {Co_ref}"
            )
        self._Co_ref = Co_ref
        self._face_flux = face_flux

    @property
    def Co_ref(self) -> float:
        """Reference Courant number."""
        return self._Co_ref

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Bounded Euler ddt.

        Args:
            coeff: Scalar coefficient (e.g. density ρ or 1).
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh`` (inferred from *phi* when possible).
            phi_old: Previous time-step field.  Defaults to *phi*.

        Returns:
            :class:`FvMatrix` with bounded Euler coefficients.
        """
        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        if phi_old is None:
            phi_old_data = phi_data
        else:
            phi_old_data = self._extract_phi(phi_old).to(
                device=mesh.device, dtype=mesh.dtype,
            )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # Compute limiting factor from face flux if available
        if self._face_flux is not None:
            flux = self._face_flux.to(
                device=mesh.device, dtype=mesh.dtype,
            )
            n_internal = mesh.n_internal_faces

            # Sum of |flux| over all faces of each cell
            flux_abs = flux.abs()
            cell_flux_sum = torch.zeros(
                n_cells, dtype=mesh.dtype, device=mesh.device,
            )
            # Internal faces: contribute to both owner and neighbour
            if n_internal > 0:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[:n_internal], flux_abs[:n_internal],
                )
                cell_flux_sum.scatter_add_(
                    0, mesh.neighbour, flux_abs[:n_internal],
                )
            # Boundary faces: contribute to owner only
            if mesh.n_faces > n_internal:
                cell_flux_sum.scatter_add_(
                    0,
                    mesh.owner[n_internal:],
                    flux_abs[n_internal:],
                )

            # Local Co = sum(|flux|) * dt / V
            local_co = cell_flux_sum * dt / cell_volumes.clamp(min=1e-30)

            # Limiting factor: min(1, Co_ref / Co_local)
            limit = torch.clamp(
                self._Co_ref / local_co.clamp(min=1e-30), max=1.0,
            )
        else:
            limit = torch.ones(n_cells, dtype=mesh.dtype, device=mesh.device)

        # Diagonal: coeff * V * limit / dt
        mat.diag = coeff * cell_volumes * limit / dt

        # Source: coeff * V * phi_old / dt  (standard Euler source)
        if phi_old_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_old_data / dt
        else:
            mat.source = (
                coeff * cell_volumes * phi_old_data.sum(dim=-1) / dt
            )

        return mat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"Co_ref={self._Co_ref})"
        )


# ---------------------------------------------------------------------------
# Backward second-order v2 (three time levels, variable dt)
# ---------------------------------------------------------------------------


class BackwardDdt2(DdtScheme):
    """Second-order backward time derivative v2 with variable-dt support.

    Improves on :class:`BackwardDdt` by supporting variable time-step sizes.
    Uses the generalised BDF2 formula::

        d(phi)/dt ≈ ((2*r+1)/(r*(r+1))) * phi
                    - (r+1)/r * phi_old
                    + r/(r+1) * phi_old2

    where r = dt / dt_old is the ratio of current to previous time step.

    The resulting FvMatrix coefficients::

        diag_i   = coeff * V_i * (2*r+1) / (r*(r+1)) / dt
        source_i = coeff * V_i / dt * [(r+1)/r * phi_old_i - r/(r+1) * phi_old2_i]

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    dt_old : float, optional
        Previous time-step size.  When not provided, assumes equal time
        steps (equivalent to :class:`BackwardDdt`).
    """

    def __init__(self, mesh: Any, dt_old: float | None = None) -> None:
        super().__init__(mesh)
        self._dt_old = dt_old

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
        phi_old2: Any = None,
        dt_old: float | None = None,
    ) -> FvMatrix:
        """Second-order backward ddt with variable-dt support.

        Args:
            coeff: Scalar coefficient (e.g. density ρ or 1).
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh`` (inferred from *phi* when possible).
            phi_old: **Required.**  Field at previous time-step (n-1).
            phi_old2: **Required.**  Field at two time-steps ago (n-2).
            dt_old: Previous time-step size (overrides constructor value).

        Raises:
            ValueError: If *phi_old* or *phi_old2* is not provided.
        """
        if phi_old is None:
            raise ValueError(
                "BackwardDdt2 requires phi_old (previous time-step field)"
            )
        if phi_old2 is None:
            raise ValueError(
                "BackwardDdt2 requires phi_old2 (two time-steps ago field)"
            )

        mesh = mesh or self._mesh
        dt_old_val = dt_old or self._dt_old or dt
        r = dt / dt_old_val

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old_data = self._extract_phi(phi_old).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old2_data = self._extract_phi(phi_old2).to(
            device=mesh.device, dtype=mesh.dtype,
        )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # 可变时间步长 BDF2 系数
        c0 = (2.0 * r + 1.0) / (r * (r + 1.0))
        c1 = (r + 1.0) / r
        c2 = r / (r + 1.0)

        mat.diag = coeff * cell_volumes * c0 / dt

        if phi_old_data.dim() == 1:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = coeff * cell_volumes * source_val / dt
        else:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = (
                coeff * cell_volumes * source_val.sum(dim=-1) / dt
            )

        return mat


# ---------------------------------------------------------------------------
# Bounded Euler v2 (with adaptive limiting)
# ---------------------------------------------------------------------------


class BoundedDdt2(DdtScheme):
    r"""Bounded Euler time derivative v2 with adaptive limiting.

    Improves on :class:`BoundedDdt` by using a smoother limiting function
    that blends between Euler and steady-state based on the Courant number:

    .. math::

        \text{diag}_i = \frac{\text{coeff} \cdot V_i}{\Delta t}
            \cdot \frac{1}{1 + \max(0, \text{Co}_i / \text{Co}_{\text{ref}} - 1)}

    This provides a smoother transition than the sharp clipping used in
    :class:`BoundedDdt`, avoiding oscillations in the diagonal coefficient.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    Co_ref : float, optional
        Reference Courant number for limiting.  Default is 1.0.
    face_flux : torch.Tensor, optional
        Face flux values ``(n_faces,)``.
    """

    def __init__(
        self, mesh: Any, Co_ref: float = 1.0,
        face_flux: Any = None,
    ) -> None:
        super().__init__(mesh)
        if Co_ref <= 0.0:
            raise ValueError(
                f"Co_ref must be positive, got {Co_ref}"
            )
        self._Co_ref = Co_ref
        self._face_flux = face_flux

    @property
    def Co_ref(self) -> float:
        """Reference Courant number."""
        return self._Co_ref

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Bounded Euler v2 ddt.

        Args:
            coeff: Scalar coefficient (e.g. density ρ or 1).
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh`` (inferred from *phi* when possible).
            phi_old: Previous time-step field.  Defaults to *phi*.

        Returns:
            :class:`FvMatrix` with adaptive bounded Euler coefficients.
        """
        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        if phi_old is None:
            phi_old_data = phi_data
        else:
            phi_old_data = self._extract_phi(phi_old).to(
                device=mesh.device, dtype=mesh.dtype,
            )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        if self._face_flux is not None:
            flux = self._face_flux.to(
                device=mesh.device, dtype=mesh.dtype,
            )
            n_internal = mesh.n_internal_faces

            flux_abs = flux.abs()
            cell_flux_sum = torch.zeros(
                n_cells, dtype=mesh.dtype, device=mesh.device,
            )
            if n_internal > 0:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[:n_internal], flux_abs[:n_internal],
                )
                cell_flux_sum.scatter_add_(
                    0, mesh.neighbour, flux_abs[:n_internal],
                )
            if mesh.n_faces > n_internal:
                cell_flux_sum.scatter_add_(
                    0,
                    mesh.owner[n_internal:],
                    flux_abs[n_internal:],
                )

            local_co = cell_flux_sum * dt / cell_volumes.clamp(min=1e-30)

            # v2 改进：平滑限制函数
            excess = torch.clamp(local_co / self._Co_ref - 1.0, min=0.0)
            limit = 1.0 / (1.0 + excess)
        else:
            limit = torch.ones(n_cells, dtype=mesh.dtype, device=mesh.device)

        mat.diag = coeff * cell_volumes * limit / dt

        if phi_old_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_old_data / dt
        else:
            mat.source = (
                coeff * cell_volumes * phi_old_data.sum(dim=-1) / dt
            )

        return mat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"Co_ref={self._Co_ref})"
        )


# ---------------------------------------------------------------------------
# Backward second-order v3 (three time levels, L-stable variant)
# ---------------------------------------------------------------------------


class BackwardDdt3(DdtScheme):
    """Second-order backward time derivative v3 with L-stable coefficients.

    Improves on :class:`BackwardDdt2` by using L-stable BDF coefficients
    that provide stronger damping of high-frequency components:

    .. math::

        \\frac{d\\phi}{dt} \\approx
        \\frac{(4r^2 + 6r + 3)}{(r+1)(2r+1)(r+2)} \\phi
        - \\frac{(r+1)^2}{r(2r+1)} \\phi_{\\text{old}}
        + \\frac{r^2}{(r+1)(r+2)} \\phi_{\\text{old2}}

    where r = dt / dt_old is the ratio of current to previous time step.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    dt_old : float, optional
        Previous time-step size.
    """

    def __init__(self, mesh: Any, dt_old: float | None = None) -> None:
        super().__init__(mesh)
        self._dt_old = dt_old

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
        phi_old2: Any = None,
        dt_old: float | None = None,
    ) -> FvMatrix:
        """L-stable backward ddt v3.

        Args:
            coeff: Scalar coefficient.
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh``.
            phi_old: **Required.** Previous time-step field.
            phi_old2: **Required.** Two time-steps ago field.
            dt_old: Previous time-step size (overrides constructor value).

        Raises:
            ValueError: If *phi_old* or *phi_old2* is not provided.
        """
        if phi_old is None:
            raise ValueError(
                "BackwardDdt3 requires phi_old (previous time-step field)"
            )
        if phi_old2 is None:
            raise ValueError(
                "BackwardDdt3 requires phi_old2 (two time-steps ago field)"
            )

        mesh = mesh or self._mesh
        dt_old_val = dt_old or self._dt_old or dt
        r = dt / dt_old_val

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old_data = self._extract_phi(phi_old).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old2_data = self._extract_phi(phi_old2).to(
            device=mesh.device, dtype=mesh.dtype,
        )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # v3: L-stable BDF 系数
        r1 = r + 1.0
        r2 = r + 2.0
        c0 = (4.0 * r * r + 6.0 * r + 3.0) / (r1 * (2.0 * r + 1.0) * r2)
        c1 = r1 * r1 / (r * (2.0 * r + 1.0))
        c2 = r * r / (r1 * r2)

        mat.diag = coeff * cell_volumes * c0 / dt

        if phi_old_data.dim() == 1:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = coeff * cell_volumes * source_val / dt
        else:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = (
                coeff * cell_volumes * source_val.sum(dim=-1) / dt
            )

        return mat


# ---------------------------------------------------------------------------
# Bounded Euler v3 (with tanh-based limiting)
# ---------------------------------------------------------------------------


class BoundedDdt3(DdtScheme):
    r"""Bounded Euler time derivative v3 with tanh-based limiting.

    Improves on :class:`BoundedDdt2` by using a tanh-based limiting function
    that provides even smoother transition and is bounded between 0 and 1:

    .. math::

        \text{limit}_i = 1 - \tanh(\max(0, \text{Co}_i / \text{Co}_{\text{ref}} - 1))

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    Co_ref : float, optional
        Reference Courant number for limiting.  Default is 1.0.
    face_flux : torch.Tensor, optional
        Face flux values ``(n_faces,)``.
    """

    def __init__(
        self, mesh: Any, Co_ref: float = 1.0,
        face_flux: Any = None,
    ) -> None:
        super().__init__(mesh)
        if Co_ref <= 0.0:
            raise ValueError(
                f"Co_ref must be positive, got {Co_ref}"
            )
        self._Co_ref = Co_ref
        self._face_flux = face_flux

    @property
    def Co_ref(self) -> float:
        """Reference Courant number."""
        return self._Co_ref

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Bounded Euler v3 ddt with tanh limiting.

        Args:
            coeff: Scalar coefficient.
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh``.
            phi_old: Previous time-step field.  Defaults to *phi*.

        Returns:
            :class:`FvMatrix` with tanh-bounded Euler coefficients.
        """
        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        if phi_old is None:
            phi_old_data = phi_data
        else:
            phi_old_data = self._extract_phi(phi_old).to(
                device=mesh.device, dtype=mesh.dtype,
            )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        if self._face_flux is not None:
            flux = self._face_flux.to(
                device=mesh.device, dtype=mesh.dtype,
            )
            n_internal = mesh.n_internal_faces

            flux_abs = flux.abs()
            cell_flux_sum = torch.zeros(
                n_cells, dtype=mesh.dtype, device=mesh.device,
            )
            if n_internal > 0:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[:n_internal], flux_abs[:n_internal],
                )
                cell_flux_sum.scatter_add_(
                    0, mesh.neighbour, flux_abs[:n_internal],
                )
            if mesh.n_faces > n_internal:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[n_internal:], flux_abs[n_internal:],
                )

            local_co = cell_flux_sum * dt / cell_volumes.clamp(min=1e-30)

            # v3: tanh 限制函数
            excess = torch.clamp(local_co / self._Co_ref - 1.0, min=0.0)
            limit = 1.0 - torch.tanh(excess)
        else:
            limit = torch.ones(n_cells, dtype=mesh.dtype, device=mesh.device)

        mat.diag = coeff * cell_volumes * limit / dt

        if phi_old_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_old_data / dt
        else:
            mat.source = (
                coeff * cell_volumes * phi_old_data.sum(dim=-1) / dt
            )

        return mat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"Co_ref={self._Co_ref})"
        )


# ---------------------------------------------------------------------------
# Backward second-order v4 (augmented L-stable BDF2)
# ---------------------------------------------------------------------------


class BackwardDdt4(DdtScheme):
    """Second-order backward time derivative v4 with augmented L-stable coefficients.

    Improves on :class:`BackwardDdt3` by using augmented BDF coefficients
    that provide even stronger high-frequency damping, suitable for
    under-resolved simulations:

    .. math::

        d(phi)/dt \approx c0 * phi / dt - c1 * phi_old / dt + c2 * phi_old2 / dt

    where the coefficients are:

        c0 = (3*r^2 + 4*r + 2) / (r*(r+1)*(r+2))
        c1 = (r+1)^2 * (2*r+1) / (r*(r+1)*(r+2))
        c2 = r^2 * (2*r+3) / ((r+1)*(r+2)*(2*r+1))

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    dt_old : float, optional
        Previous time-step size.
    """

    def __init__(self, mesh: Any, dt_old: float | None = None) -> None:
        super().__init__(mesh)
        self._dt_old = dt_old

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
        phi_old2: Any = None,
        dt_old: float | None = None,
    ) -> FvMatrix:
        """Augmented L-stable backward ddt v4.

        Args:
            coeff: Scalar coefficient.
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh``.
            phi_old: **Required.** Previous time-step field.
            phi_old2: **Required.** Two time-steps ago field.
            dt_old: Previous time-step size (overrides constructor value).

        Raises:
            ValueError: If *phi_old* or *phi_old2* is not provided.
        """
        if phi_old is None:
            raise ValueError(
                "BackwardDdt4 requires phi_old (previous time-step field)"
            )
        if phi_old2 is None:
            raise ValueError(
                "BackwardDdt4 requires phi_old2 (two time-steps ago field)"
            )

        mesh = mesh or self._mesh
        dt_old_val = dt_old or self._dt_old or dt
        r = dt / dt_old_val

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old_data = self._extract_phi(phi_old).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old2_data = self._extract_phi(phi_old2).to(
            device=mesh.device, dtype=mesh.dtype,
        )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # v4: 增强 L-stable BDF 系数
        r1 = r + 1.0
        r2 = r + 2.0
        c0 = (3.0 * r * r + 4.0 * r + 2.0) / (r * r1 * r2)
        c1 = r1 * r1 * (2.0 * r + 1.0) / (r * r1 * r2)
        c2 = r * r * (2.0 * r + 3.0) / (r1 * r2 * (2.0 * r + 1.0))

        mat.diag = coeff * cell_volumes * c0 / dt

        if phi_old_data.dim() == 1:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = coeff * cell_volumes * source_val / dt
        else:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = (
                coeff * cell_volumes * source_val.sum(dim=-1) / dt
            )

        return mat


# ---------------------------------------------------------------------------
# Bounded Euler v4 (with sigmoid limiting)
# ---------------------------------------------------------------------------


class BoundedDdt4(DdtScheme):
    r"""Bounded Euler time derivative v4 with sigmoid limiting.

    Improves on :class:`BoundedDdt3` by using a sigmoid (logistic) limiting
    function that provides an S-shaped transition between Euler and
    steady-state:

    .. math::

        \text{limit}_i = 1 / (1 + \exp(k \, (\text{Co}_i / \text{Co}_{\text{ref}} - 1)))

    where *k* controls the steepness of the transition.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    Co_ref : float, optional
        Reference Courant number for limiting.  Default is 1.0.
    face_flux : torch.Tensor, optional
        Face flux values ``(n_faces,)``.
    steepness : float, optional
        Steepness of the sigmoid transition.  Default is 5.0.
    """

    def __init__(
        self, mesh: Any, Co_ref: float = 1.0,
        face_flux: Any = None, steepness: float = 5.0,
    ) -> None:
        super().__init__(mesh)
        if Co_ref <= 0.0:
            raise ValueError(
                f"Co_ref must be positive, got {Co_ref}"
            )
        self._Co_ref = Co_ref
        self._face_flux = face_flux
        self._steepness = steepness

    @property
    def Co_ref(self) -> float:
        """Reference Courant number."""
        return self._Co_ref

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Bounded Euler v4 ddt with sigmoid limiting.

        Args:
            coeff: Scalar coefficient.
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh``.
            phi_old: Previous time-step field.  Defaults to *phi*.

        Returns:
            :class:`FvMatrix` with sigmoid-bounded Euler coefficients.
        """
        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        if phi_old is None:
            phi_old_data = phi_data
        else:
            phi_old_data = self._extract_phi(phi_old).to(
                device=mesh.device, dtype=mesh.dtype,
            )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        if self._face_flux is not None:
            flux = self._face_flux.to(
                device=mesh.device, dtype=mesh.dtype,
            )
            n_internal = mesh.n_internal_faces

            flux_abs = flux.abs()
            cell_flux_sum = torch.zeros(
                n_cells, dtype=mesh.dtype, device=mesh.device,
            )
            if n_internal > 0:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[:n_internal], flux_abs[:n_internal],
                )
                cell_flux_sum.scatter_add_(
                    0, mesh.neighbour, flux_abs[:n_internal],
                )
            if mesh.n_faces > n_internal:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[n_internal:], flux_abs[n_internal:],
                )

            local_co = cell_flux_sum * dt / cell_volumes.clamp(min=1e-30)

            # v4: sigmoid 限制函数
            k = self._steepness
            limit = torch.sigmoid(
                -k * (local_co / self._Co_ref - 1.0)
            )
        else:
            limit = torch.ones(n_cells, dtype=mesh.dtype, device=mesh.device)

        mat.diag = coeff * cell_volumes * limit / dt

        if phi_old_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_old_data / dt
        else:
            mat.source = (
                coeff * cell_volumes * phi_old_data.sum(dim=-1) / dt
            )

        return mat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"Co_ref={self._Co_ref})"
        )


# ---------------------------------------------------------------------------
# Backward second-order v5 (three time levels, optimised L-stable BDF2)
# ---------------------------------------------------------------------------


class BackwardDdt5(DdtScheme):
    """Second-order backward time derivative v5 with optimised L-stable coefficients.

    Improves on :class:`BackwardDdt4` by using optimised BDF coefficients
    that minimise the truncation error constant while maintaining L-stability,
    suitable for high-accuracy transient simulations:

    .. math::

        d(phi)/dt \approx c0 * phi / dt - c1 * phi_old / dt + c2 * phi_old2 / dt

    where the coefficients are:

        c0 = (4*r^2 + 6*r + 3) / (r*(r+1)*(2*r+1))
        c1 = (r+1)^2 * (2*r+1) / (r*(r+1)*(2*r+1))
        c2 = r^2 / ((r+1)*(2*r+1))

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    dt_old : float, optional
        Previous time-step size.
    """

    def __init__(self, mesh: Any, dt_old: float | None = None) -> None:
        super().__init__(mesh)
        self._dt_old = dt_old

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
        phi_old2: Any = None,
        dt_old: float | None = None,
    ) -> FvMatrix:
        """Optimised L-stable backward ddt v5.

        Args:
            coeff: Scalar coefficient.
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh``.
            phi_old: **Required.** Previous time-step field.
            phi_old2: **Required.** Two time-steps ago field.
            dt_old: Previous time-step size (overrides constructor value).

        Raises:
            ValueError: If *phi_old* or *phi_old2* is not provided.
        """
        if phi_old is None:
            raise ValueError(
                "BackwardDdt5 requires phi_old (previous time-step field)"
            )
        if phi_old2 is None:
            raise ValueError(
                "BackwardDdt5 requires phi_old2 (two time-steps ago field)"
            )

        mesh = mesh or self._mesh
        dt_old_val = dt_old or self._dt_old or dt
        r = dt / dt_old_val

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old_data = self._extract_phi(phi_old).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        phi_old2_data = self._extract_phi(phi_old2).to(
            device=mesh.device, dtype=mesh.dtype,
        )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        # v5: 优化 L-stable BDF 系数
        r1 = r + 1.0
        r2 = 2.0 * r + 1.0
        c0 = (4.0 * r * r + 6.0 * r + 3.0) / (r * r1 * r2)
        c1 = r1 * r1 * r2 / (r * r1 * r2)
        c2 = r * r / (r1 * r2)

        mat.diag = coeff * cell_volumes * c0 / dt

        if phi_old_data.dim() == 1:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = coeff * cell_volumes * source_val / dt
        else:
            source_val = c1 * phi_old_data - c2 * phi_old2_data
            mat.source = (
                coeff * cell_volumes * source_val.sum(dim=-1) / dt
            )

        return mat


# ---------------------------------------------------------------------------
# Bounded Euler v5 (with piecewise-linear-log limiting)
# ---------------------------------------------------------------------------


class BoundedDdt5(DdtScheme):
    r"""Bounded Euler time derivative v5 with piecewise-linear-log limiting.

    Improves on :class:`BoundedDdt4` by using a piecewise-linear-logarithmic
    limiting function that provides a good balance between the sharp clipping
    of v1 and the smooth transitions of v3/v4:

    .. math::

        \text{limit}_i = \begin{cases}
            1 & \text{if } \text{Co}_i \leq \text{Co}_{\text{ref}} \\
            1 / (1 + \log(\text{Co}_i / \text{Co}_{\text{ref}}))
            & \text{if } \text{Co}_i > \text{Co}_{\text{ref}}
        \end{cases}

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    Co_ref : float, optional
        Reference Courant number for limiting.  Default is 1.0.
    face_flux : torch.Tensor, optional
        Face flux values ``(n_faces,)``.
    """

    def __init__(
        self, mesh: Any, Co_ref: float = 1.0,
        face_flux: Any = None,
    ) -> None:
        super().__init__(mesh)
        if Co_ref <= 0.0:
            raise ValueError(
                f"Co_ref must be positive, got {Co_ref}"
            )
        self._Co_ref = Co_ref
        self._face_flux = face_flux

    @property
    def Co_ref(self) -> float:
        """Reference Courant number."""
        return self._Co_ref

    def ddt(
        self,
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
        phi_old: Any = None,
    ) -> FvMatrix:
        """Bounded Euler v5 ddt with piecewise-linear-log limiting.

        Args:
            coeff: Scalar coefficient.
            phi: Current field values.
            dt: Time step size.
            mesh: The ``FvMesh``.
            phi_old: Previous time-step field.  Defaults to *phi*.

        Returns:
            :class:`FvMatrix` with piecewise-linear-log bounded Euler coefficients.
        """
        mesh = mesh or self._mesh

        phi_data = self._extract_phi(phi).to(
            device=mesh.device, dtype=mesh.dtype,
        )
        if phi_old is None:
            phi_old_data = phi_data
        else:
            phi_old_data = self._extract_phi(phi_old).to(
                device=mesh.device, dtype=mesh.dtype,
            )

        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes
        owner = mesh.owner[: mesh.n_internal_faces]
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner, neighbour,
            device=mesh.device, dtype=mesh.dtype,
        )

        if self._face_flux is not None:
            flux = self._face_flux.to(
                device=mesh.device, dtype=mesh.dtype,
            )
            n_internal = mesh.n_internal_faces

            flux_abs = flux.abs()
            cell_flux_sum = torch.zeros(
                n_cells, dtype=mesh.dtype, device=mesh.device,
            )
            if n_internal > 0:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[:n_internal], flux_abs[:n_internal],
                )
                cell_flux_sum.scatter_add_(
                    0, mesh.neighbour, flux_abs[:n_internal],
                )
            if mesh.n_faces > n_internal:
                cell_flux_sum.scatter_add_(
                    0, mesh.owner[n_internal:], flux_abs[n_internal:],
                )

            local_co = cell_flux_sum * dt / cell_volumes.clamp(min=1e-30)

            # v5: 分段线性-对数限制函数
            co_ratio = local_co / self._Co_ref
            # 当 Co <= Co_ref 时不限制；当 Co > Co_ref 时用对数衰减
            excess = torch.clamp(co_ratio - 1.0, min=0.0)
            limit = torch.where(
                excess > 0.0,
                1.0 / (1.0 + torch.log1p(excess)),
                torch.ones_like(excess),
            )
        else:
            limit = torch.ones(n_cells, dtype=mesh.dtype, device=mesh.device)

        mat.diag = coeff * cell_volumes * limit / dt

        if phi_old_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_old_data / dt
        else:
            mat.source = (
                coeff * cell_volumes * phi_old_data.sum(dim=-1) / dt
            )

        return mat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"Co_ref={self._Co_ref})"
        )


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

DDT_REGISTRY: dict[str, type[DdtScheme]] = {
    "Euler": EulerDdt,
    "steadyState": SteadyStateDdt,
    "CrankNicolson": CrankNicolsonDdt,
    "backward": BackwardDdt,
    "bounded": BoundedDdt,
    "backward2": BackwardDdt2,
    "bounded2": BoundedDdt2,
    "backward3": BackwardDdt3,
    "backward4": BackwardDdt4,
    "backward5": BackwardDdt5,
    "bounded3": BoundedDdt3,
    "bounded4": BoundedDdt4,
    "bounded5": BoundedDdt5,
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
