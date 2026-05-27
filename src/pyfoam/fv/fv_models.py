"""
fvModels — source term framework for finite volume equations.

Provides an RTS (Run-Time Selection) registry of source models that add
explicit and implicit contributions to the discretised equation *before*
solving.

In OpenFOAM, ``fvModels`` inject volumetric source terms into the governing
equations (momentum, energy, continuity, species transport).  The generic
semi-implicit form adds::

    Su + Sp * phi

to the right-hand side and diagonal of the FvMatrix, where ``Su`` is the
explicit (constant) part and ``Sp`` is the implicit (field-proportional)
part that aids diagonal dominance.

Models:

- **semiImplicitSource** — generic ``Su + Sp * phi`` volumetric source
- **massSource** — mass source / sink in the continuity equation
- **heatSource** — volumetric heat source in the energy equation
- **porosityForce** — Darcy-Forchheimer porosity resistance
- **codedFvModel** — user-defined Python function as source

Usage::

    from pyfoam.fv.fv_models import FvModel, SemiImplicitSource

    # Decorator-based registration
    @FvModel.register("mySource")
    class MySource(FvModel):
        ...

    # Factory creation
    model = FvModel.create("semiImplicitSource", Su=100.0, Sp=-0.5)
    model.apply(matrix, field)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Type

import torch

from pyfoam.core.fv_matrix import FvMatrix

__all__ = [
    "FvModel",
    "SemiImplicitSource",
    "MassSource",
    "HeatSource",
    "PorosityForce",
    "CodedFvModel",
    "create_fv_model",
]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class FvModel(ABC):
    """Abstract base for all fvModels.

    Subclasses must implement :meth:`apply`.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @FvModel.register("semiImplicitSource")
        class SemiImplicitSource(FvModel):
            ...

        m = FvModel.create("semiImplicitSource", Su=100.0, Sp=-0.5)
    """

    _registry: ClassVar[dict[str, Type[FvModel]]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self._coeffs: dict[str, Any] = kwargs
        self._active: bool = True

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model class under *name*.

        Usage::

            @FvModel.register("semiImplicitSource")
            class SemiImplicitSource(FvModel):
                ...
        """

        def decorator(model_cls: Type[FvModel]) -> Type[FvModel]:
            if name in cls._registry:
                raise ValueError(
                    f"fvModel '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> FvModel:
        """Factory: create a model instance by registered *name*.

        Args:
            name: Registered model type name (e.g. ``"semiImplicitSource"``).
            **kwargs: Model parameters (e.g. ``Su``, ``Sp``, ``cells``).

        Returns:
            Instantiated model.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown fvModel type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def type_name(self) -> str:
        """Return the registered type name for this model class."""
        for name, model_cls in self._registry.items():
            if isinstance(self, model_cls):
                return name
        return self.__class__.__name__

    @property
    def coeffs(self) -> dict[str, Any]:
        """Return the model coefficient dictionary."""
        return self._coeffs

    @property
    def active(self) -> bool:
        """Whether this model is currently active."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(
        self,
        matrix: FvMatrix,
        field: torch.Tensor,
    ) -> None:
        """Apply the source model to the discretised equation.

        Modifies *matrix* in-place by adding source contributions
        (explicit ``Su`` to ``matrix.source``, implicit ``Sp`` to
        ``matrix.diag``).

        Args:
            matrix: The :class:`~pyfoam.core.fv_matrix.FvMatrix` to modify.
            field: Current field values ``(n_cells,)``.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._coeffs})"


# ---------------------------------------------------------------------------
# SemiImplicitSource
# ---------------------------------------------------------------------------


@FvModel.register("semiImplicitSource")
class SemiImplicitSource(FvModel):
    """Generic semi-implicit volumetric source: ``Su + Sp * phi``.

    This is the most common fvModel.  It adds:

    - **Su** (explicit) to ``matrix.source`` (right-hand side)
    - **Sp * field** (implicit) to ``matrix.diag`` (diagonal)

    The implicit contribution ``Sp`` improves diagonal dominance when
    ``Sp <= 0`` (source is a sink).  A positive ``Sp`` weakens diagonal
    dominance and should be used with care.

    Corresponds to OpenFOAM's ``semiImplicitSource``.

    Parameters
    ----------
    Su : float | torch.Tensor
        Explicit source (constant part).  Scalar broadcast to all cells,
        or per-cell ``(n_cells,)`` tensor.
    Sp : float | torch.Tensor
        Implicit source coefficient (field-proportional part).
        Scalar broadcast to all cells, or per-cell ``(n_cells,)`` tensor.
    cells : list[int] | torch.Tensor | None
        Restrict source to specific cell indices.  ``None`` means all cells.

    Examples::

        model = SemiImplicitSource(Su=100.0, Sp=-0.5)
        model.apply(matrix, field)
    """

    def __init__(
        self,
        *,
        Su: float | torch.Tensor = 0.0,
        Sp: float | torch.Tensor = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(Su=Su, Sp=Sp, cells=cells, **kwargs)
        self._Su = Su
        self._Sp = Sp
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def Su(self) -> float | torch.Tensor:
        """Explicit source coefficient."""
        return self._Su

    @property
    def Sp(self) -> float | torch.Tensor:
        """Implicit source coefficient."""
        return self._Sp

    @property
    def cells(self) -> list[int] | torch.Tensor | None:
        """Cell indices restriction."""
        return self._cells

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Add ``Su + Sp * field`` to the matrix equation."""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # Helper: broadcast scalar or tensor to (n_cells,)
        def _to_cells(val: float | torch.Tensor) -> torch.Tensor:
            if isinstance(val, (int, float)):
                return torch.full(
                    (n,), float(val), device=device, dtype=dtype
                )
            return val.to(device=device, dtype=dtype)

        su = _to_cells(self._Su)
        sp = _to_cells(self._Sp)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
            su = su * mask
            sp = sp * mask

        # Explicit: source += Su
        matrix._source = matrix._source + su
        # Implicit: diag += Sp (multiplied by field during Ax product)
        matrix._diag = matrix._diag + sp


# ---------------------------------------------------------------------------
# MassSource
# ---------------------------------------------------------------------------


@FvModel.register("massSource")
class MassSource(FvModel):
    """Mass source / sink in the continuity equation.

    Adds a volumetric mass source to the continuity equation.  Positive
    values inject mass (e.g. injection), negative values remove mass
    (e.g. suction).

    The source is split into semi-implicit form using a linearisation
    coefficient ``alpha``::

        Su = (1 - alpha) * mass_source
        Sp = alpha * mass_source / field

    By default ``alpha = 0`` (purely explicit).

    Corresponds to OpenFOAM's ``massSource``.

    Parameters
    ----------
    mass_source : float | torch.Tensor
        Volumetric mass source rate [kg/m^3/s].  Scalar or per-cell.
    alpha : float
        Linearisation fraction in [0, 1].  ``0`` = fully explicit,
        ``1`` = fully implicit.  Default ``0.0``.
    cells : list[int] | torch.Tensor | None
        Restrict to specific cell indices.  ``None`` = all cells.

    Examples::

        model = MassSource(mass_source=0.01, alpha=0.5)
        model.apply(continuity_matrix, rho_field)
    """

    def __init__(
        self,
        *,
        mass_source: float | torch.Tensor = 0.0,
        alpha: float = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mass_source=mass_source, alpha=alpha, cells=cells, **kwargs)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}"
            )
        self._mass_source = mass_source
        self._alpha = alpha
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def mass_source(self) -> float | torch.Tensor:
        """Volumetric mass source rate."""
        return self._mass_source

    @property
    def alpha(self) -> float:
        """Linearisation fraction."""
        return self._alpha

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Add mass source to continuity matrix."""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        def _to_cells(val: float | torch.Tensor) -> torch.Tensor:
            if isinstance(val, (int, float)):
                return torch.full(
                    (n,), float(val), device=device, dtype=dtype
                )
            return val.to(device=device, dtype=dtype)

        ms = _to_cells(self._mass_source)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
            ms = ms * mask

        # Semi-implicit split
        Su = (1.0 - self._alpha) * ms
        # For implicit part: Sp * phi = alpha * ms => Sp = alpha * ms / phi
        # Avoid division by zero: clamp field away from zero
        field_safe = field.to(device=device, dtype=dtype).clamp(min=1e-30)
        Sp = self._alpha * ms / field_safe

        matrix._source = matrix._source + Su
        matrix._diag = matrix._diag + Sp


# ---------------------------------------------------------------------------
# HeatSource
# ---------------------------------------------------------------------------


@FvModel.register("heatSource")
class HeatSource(FvModel):
    """Volumetric heat source in the energy equation.

    Adds a heat source [W/m^3] to the energy equation.  Supports
    semi-implicit linearisation via ``alpha``::

        Su = (1 - alpha) * Q
        Sp = -alpha * Q / (Cp * T)

    The default ``alpha = 0`` is purely explicit.  Setting ``alpha > 0``
    linearises the source with respect to temperature, improving
    convergence for temperature-dependent heat sources.

    Parameters
    ----------
    Q : float | torch.Tensor
        Volumetric heat source [W/m^3].  Positive = heating.
    Cp : float
        Specific heat capacity [J/(kg K)].  Required for implicit
        linearisation.  Default ``1005.0`` (air).
    alpha : float
        Linearisation fraction in [0, 1].  Default ``0.0``.
    cells : list[int] | torch.Tensor | None
        Restrict to specific cell indices.  ``None`` = all cells.

    Examples::

        model = HeatSource(Q=1e6, Cp=1005.0, alpha=0.3)
        model.apply(energy_matrix, T_field)
    """

    def __init__(
        self,
        *,
        Q: float | torch.Tensor = 0.0,
        Cp: float = 1005.0,
        alpha: float = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(Q=Q, Cp=Cp, alpha=alpha, cells=cells, **kwargs)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if Cp <= 0.0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        self._Q = Q
        self._Cp = Cp
        self._alpha = alpha
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def Q(self) -> float | torch.Tensor:
        """Volumetric heat source [W/m^3]."""
        return self._Q

    @property
    def Cp(self) -> float:
        """Specific heat capacity [J/(kg K)]."""
        return self._Cp

    @property
    def alpha(self) -> float:
        """Linearisation fraction."""
        return self._alpha

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Add heat source to energy matrix."""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        def _to_cells(val: float | torch.Tensor) -> torch.Tensor:
            if isinstance(val, (int, float)):
                return torch.full(
                    (n,), float(val), device=device, dtype=dtype
                )
            return val.to(device=device, dtype=dtype)

        q = _to_cells(self._Q)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
            q = q * mask

        # Semi-implicit split
        Su = (1.0 - self._alpha) * q
        # Sp = -alpha * Q / (Cp * T), negative for stability
        T_safe = field.to(device=device, dtype=dtype).clamp(min=1e-30)
        Sp = -self._alpha * q / (self._Cp * T_safe)

        matrix._source = matrix._source + Su
        matrix._diag = matrix._diag + Sp


# ---------------------------------------------------------------------------
# PorosityForce — Darcy-Forchheimer
# ---------------------------------------------------------------------------


@FvModel.register("porosityForce")
class PorosityForce(FvModel):
    """Darcy-Forchheimer porosity resistance model.

    Models flow resistance in porous media using the standard
    Darcy-Forchheimer equation::

        F = -(mu * D * U + 0.5 * rho * F_coeff * |U| * U)

    where:

    - ``D`` (Darcy coefficient) — viscous resistance [1/m^2]
    - ``F_coeff`` (Forchheimer coefficient) — inertial resistance [1/m]
    - ``mu`` — dynamic viscosity [Pa s]
    - ``rho`` — fluid density [kg/m^3]

    The force is linearised implicitly for the Darcy term (proportional
    to velocity) and semi-implicitly for the Forchheimer term.

    Corresponds to OpenFOAM's ``porosityForce`` with Darcy-Forchheimer
    model.

    Parameters
    ----------
    D : float
        Darcy (viscous) resistance coefficient [1/m^2].
    F_coeff : float
        Forchheimer (inertial) resistance coefficient [1/m].
    mu : float
        Dynamic viscosity [Pa s].  Default ``1.8e-5`` (air).
    rho : float
        Fluid density [kg/m^3].  Default ``1.225`` (air at sea level).
    cells : list[int] | torch.Tensor | None
        Restrict to specific cell indices.  ``None`` = all cells.

    Examples::

        model = PorosityForce(D=1e8, F_coeff=0.0, mu=1e-3, rho=1000.0)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        D: float = 0.0,
        F_coeff: float = 0.0,
        mu: float = 1.8e-5,
        rho: float = 1.225,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            D=D, F_coeff=F_coeff, mu=mu, rho=rho, cells=cells, **kwargs
        )
        if D < 0.0:
            raise ValueError(f"D must be >= 0, got {D}")
        if F_coeff < 0.0:
            raise ValueError(f"F_coeff must be >= 0, got {F_coeff}")
        self._D = D
        self._F_coeff = F_coeff
        self._mu = mu
        self._rho = rho
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def D(self) -> float:
        """Darcy resistance coefficient [1/m^2]."""
        return self._D

    @property
    def F_coeff(self) -> float:
        """Forchheimer resistance coefficient [1/m]."""
        return self._F_coeff

    @property
    def mu(self) -> float:
        """Dynamic viscosity [Pa s]."""
        return self._mu

    @property
    def rho(self) -> float:
        """Fluid density [kg/m^3]."""
        return self._rho

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Add Darcy-Forchheimer resistance to momentum matrix.

        The Darcy term ``-mu * D * U`` is fully implicit (added to
        diagonal).  The Forchheimer term is linearised as::

            -0.5 * rho * F_coeff * |U| * U ~ Sp * U
            Sp = -0.5 * rho * F_coeff * |U|

        This is semi-implicit: the magnitude |U| is taken from the
        current field values.
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        U = field.to(device=device, dtype=dtype)

        # Build cell mask
        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
        else:
            mask = torch.ones(n, device=device, dtype=dtype)

        # Darcy: fully implicit — Sp_darcy = -mu * D
        Sp_darcy = -self._mu * self._D * mask

        # Forchheimer: semi-implicit — Sp_forch = -0.5 * rho * F_coeff * |U|
        U_mag = U.abs()
        Sp_forch = -0.5 * self._rho * self._F_coeff * U_mag * mask

        # Total implicit contribution
        Sp_total = Sp_darcy + Sp_forch

        # Forchheimer explicit remainder:
        # The actual force is -0.5*rho*F*|U|*U = Sp_forch * U
        # Since we linearised |U| around current value, there is no
        # explicit remainder in this approximation.
        matrix._diag = matrix._diag + Sp_total


# ---------------------------------------------------------------------------
# CodedFvModel
# ---------------------------------------------------------------------------


@FvModel.register("codedFvModel")
class CodedFvModel(FvModel):
    """User-defined Python function as a source term.

    Allows arbitrary source term computation via a user-supplied
    callable.  The function receives the current field values and
    must return ``(Su, Sp)`` tensors (or scalars broadcast to cells).

    This is the Python equivalent of OpenFOAM's ``codedfvModel``.

    Parameters
    ----------
    code : callable
        A function with signature ``(field: torch.Tensor) -> tuple``
        returning ``(Su, Sp)`` where each is a scalar or
        ``(n_cells,)`` tensor.
    name : str
        Descriptive name for repr / debugging.  Default ``"coded"``.

    Examples::

        def gravity_source(field):
            # Constant gravitational body force
            g = 9.81
            Su = torch.full_like(field, -g)
            Sp = torch.zeros_like(field)
            return Su, Sp

        model = CodedFvModel(code=gravity_source, name="gravity")
        model.apply(matrix, field)
    """

    def __init__(
        self,
        *,
        code: Callable[[torch.Tensor], tuple],
        name: str = "coded",
        **kwargs: Any,
    ) -> None:
        super().__init__(code=code, name=name, **kwargs)
        if not callable(code):
            raise TypeError("code must be a callable")
        self._code = code
        self._name = name

    @property
    def code(self) -> Callable[[torch.Tensor], tuple]:
        """The user-supplied source function."""
        return self._code

    @property
    def name(self) -> str:
        """Descriptive name."""
        return self._name

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Evaluate user function and add source to matrix."""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype

        Su, Sp = self._code(field.to(device=device, dtype=dtype))

        if isinstance(Su, (int, float)):
            Su = torch.full(
                (matrix._n_cells,), float(Su), device=device, dtype=dtype
            )
        else:
            Su = Su.to(device=device, dtype=dtype)

        if isinstance(Sp, (int, float)):
            Sp = torch.full(
                (matrix._n_cells,), float(Sp), device=device, dtype=dtype
            )
        else:
            Sp = Sp.to(device=device, dtype=dtype)

        matrix._source = matrix._source + Su
        matrix._diag = matrix._diag + Sp

    def __repr__(self) -> str:
        return f"CodedFvModel(name='{self._name}')"


# ---------------------------------------------------------------------------
# Factory convenience function
# ---------------------------------------------------------------------------


def create_fv_model(name: str, **kwargs: Any) -> FvModel:
    """Create an fvModel by registered name.

    Convenience wrapper around :meth:`FvModel.create`.

    Args:
        name: Model type name.
        **kwargs: Model parameters.

    Returns:
        Instantiated model.
    """
    return FvModel.create(name, **kwargs)
