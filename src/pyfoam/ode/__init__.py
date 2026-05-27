"""
pyfoam.ode -- ODE solver framework for time integration.

Provides:

Explicit solvers:
- **Euler**: Forward Euler (1st order)
- **RK4**: Classical 4th-order Runge-Kutta
- **RKF45**: Runge-Kutta-Fehlberg adaptive (4th/5th pair)

Implicit solvers:
- **Trapezoid**: Implicit trapezoidal rule (2nd order, A-stable)
- **Rosenbrock12**: Rosenbrock adaptive (L-stable, stiff)

Usage::

    from pyfoam.ode import ODESolver, create_ode_solver

    # Create by name
    solver = create_ode_solver("RK4")

    # Or use RTS directly
    solver = ODESolver.create("RKF45", rtol=1e-8)

    # Solve
    def f(t, y):
        return -y

    y_new = solver.step(f, t=0.0, y=torch.tensor([1.0]), dt=0.01)

    # Or integrate over a span
    times, states = solver.integrate(f, (0.0, 1.0), torch.tensor([1.0]), dt=0.01)
"""

from pyfoam.ode.ode_solver import (
    ODESolver,
    create_ode_solver,
    ode_solver_from_dict,
)
from pyfoam.ode.euler import EulerSolver
from pyfoam.ode.runge_kutta import RK4Solver, RKF45Solver
from pyfoam.ode.implicit import TrapezoidSolver, Rosenbrock12Solver

__all__ = [
    # Base
    "ODESolver",
    "create_ode_solver",
    "ode_solver_from_dict",
    # Explicit
    "EulerSolver",
    "RK4Solver",
    "RKF45Solver",
    # Implicit
    "TrapezoidSolver",
    "Rosenbrock12Solver",
]
