"""
ODE solver base class with RTS (Run-Time Selection) registry.

Provides:

- :class:`ODESolver` -- abstract base for all ODE solvers
- :func:`create_ode_solver` -- factory function for solver selection by name
- :func:`ode_solver_from_dict` -- create solver from dictionary parameters

Solver names:
- ``"Euler"`` -- Forward Euler (1st order, explicit)
- ``"RK4"`` -- Classical 4th-order Runge-Kutta (explicit)
- ``"RKF45"`` -- Runge-Kutta-Fehlberg adaptive (explicit, 4(5) pair)
- ``"RKCK45"`` -- Runge-Kutta-Cash-Karp adaptive (explicit, 4(5) pair)
- ``"RKDP45"`` -- Runge-Kutta-Dormand-Prince adaptive (explicit, 4(5) pair)
- ``"Trapezoid"`` -- Implicit trapezoidal rule (2nd order, A-stable)
- ``"Rosenbrock12"`` -- Rosenbrock 1(2) adaptive (stiff, L-stable)
- ``"Rosenbrock23"`` -- Rosenbrock 2(3) adaptive (stiff, L-stable)
- ``"Rosenbrock34"`` -- Rosenbrock 3(4) adaptive (stiff, L-stable)
- ``"SIS"`` -- Semi-Implicit Solver (extrapolation)
- ``"SEulex"`` -- Semi-Explicit Extrapolation solver

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Type

import torch

__all__ = [
    "ODESolver",
    "create_ode_solver",
    "ode_solver_from_dict",
]


# Type alias for the RHS function: f(t, y) -> dy/dt
RHSFunc = Callable[[float, torch.Tensor], torch.Tensor]


class ODESolver(ABC):
    """Abstract base class for ODE solvers.

    Solves the initial value problem::

        dy/dt = f(t, y),  y(t0) = y0

    Subclasses implement :meth:`step` with the specific algorithm.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @ODESolver.register("RK4")
        class RK4Solver(ODESolver):
            ...

        solver = ODESolver.create("RK4")
        y_new = solver.step(f, t, y, dt)

    Parameters
    ----------
    rtol : float
        Relative tolerance (used by adaptive solvers).
    atol : float
        Absolute tolerance (used by adaptive solvers).
    """

    # Class-level RTS registry: name -> class
    _registry: ClassVar[dict[str, Type[ODESolver]]] = {}

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> None:
        self._rtol = rtol
        self._atol = atol

    @property
    def rtol(self) -> float:
        """Relative tolerance."""
        return self._rtol

    @property
    def atol(self) -> float:
        """Absolute tolerance."""
        return self._atol

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an ODE solver class under *name*.

        Usage::

            @ODESolver.register("Euler")
            class EulerSolver(ODESolver):
                ...
        """

        def decorator(solver_cls: Type[ODESolver]) -> Type[ODESolver]:
            if name in cls._registry:
                raise ValueError(
                    f"ODE solver '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = solver_cls
            return solver_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ODESolver:
        """Factory: create an ODE solver instance by registered *name*.

        Args:
            name: Solver name (case-sensitive). One of:
                ``"Euler"``, ``"RK4"``, ``"RKF45"``, ``"RKCK45"``,
                ``"RKDP45"``, ``"Trapezoid"``, ``"Rosenbrock12"``,
                ``"Rosenbrock23"``, ``"Rosenbrock34"``, ``"SIS"``,
                ``"SEulex"``.
            **kwargs: Solver-specific parameters (rtol, atol, etc.).

        Returns:
            Solver instance.

        Raises:
            ValueError: If solver name is not recognised.
        """
        # Lazy import to trigger registration
        if not cls._registry:
            from pyfoam.ode.euler import EulerSolver  # noqa: F401
            from pyfoam.ode.runge_kutta import RK4Solver, RKF45Solver  # noqa: F401
            from pyfoam.ode.runge_kutta_ck_dp import (  # noqa: F401
                RKCK45Solver,
                RKDP45Solver,
            )
            from pyfoam.ode.implicit import (  # noqa: F401
                TrapezoidSolver,
                Rosenbrock12Solver,
            )
            from pyfoam.ode.rosenbrock import (  # noqa: F401
                Rosenbrock23Solver,
                Rosenbrock34Solver,
            )
            from pyfoam.ode.semi_implicit import (  # noqa: F401
                SISSolver,
                SEulexSolver,
            )

        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown ODE solver '{name}'. Available: {available}"
            )

        return cls._registry[name](**kwargs)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance the solution by one step.

        Args:
            f: Right-hand side function ``f(t, y) -> dy/dt``.
            t: Current time.
            y: Current state tensor.
            dt: Time step size.

        Returns:
            State tensor at time ``t + dt``.
        """

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate the ODE over a time span with fixed step size.

        Args:
            f: Right-hand side function.
            t_span: ``(t_start, t_end)``.
            y0: Initial state.
            dt: Fixed time step.

        Returns:
            Tuple of ``(times, states)`` lists.
        """
        t_start, t_end = t_span
        t = t_start
        y = y0.clone()
        times = [t]
        states = [y.clone()]

        while t < t_end - 1e-12 * abs(dt):
            current_dt = min(dt, t_end - t)
            y = self.step(f, t, y, current_dt)
            t += current_dt
            times.append(t)
            states.append(y.clone())

        return times, states

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rtol={self._rtol}, atol={self._atol})"


# ---------------------------------------------------------------------------
# Factory convenience
# ---------------------------------------------------------------------------


def create_ode_solver(name: str, **kwargs: Any) -> ODESolver:
    """Create an ODE solver by name.

    Args:
        name: Solver name. One of: ``"Euler"``, ``"RK4"``, ``"RKF45"``,
            ``"RKCK45"``, ``"RKDP45"``, ``"Trapezoid"``, ``"Rosenbrock12"``,
            ``"Rosenbrock23"``, ``"Rosenbrock34"``, ``"SIS"``, ``"SEulex"``.
        **kwargs: Solver parameters (rtol, atol).

    Returns:
        Solver instance.
    """
    return ODESolver.create(name, **kwargs)


def ode_solver_from_dict(solver_dict: dict[str, Any]) -> ODESolver:
    """Create an ODE solver from a dictionary.

    Expected keys::

        type    Euler;
        rtol    1e-6;
        atol    1e-8;

    Args:
        solver_dict: Dictionary with solver parameters.

    Returns:
        Configured solver instance.
    """
    name = solver_dict["type"]
    return create_ode_solver(
        name,
        rtol=float(solver_dict.get("rtol", 1e-6)),
        atol=float(solver_dict.get("atol", 1e-8)),
    )
