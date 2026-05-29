"""
pyfoam.ode -- ODE solver framework for time integration.

Provides:

Explicit solvers:
- **Euler**: Forward Euler (1st order)
- **RK4**: Classical 4th-order Runge-Kutta
- **RKF45**: Runge-Kutta-Fehlberg adaptive (4th/5th pair)
- **RKCK45**: Runge-Kutta-Cash-Karp adaptive (4th/5th pair)
- **RKDP45**: Runge-Kutta-Dormand-Prince adaptive (4th/5th pair)

Implicit / stiff solvers:
- **Trapezoid**: Implicit trapezoidal rule (2nd order, A-stable)
- **Rosenbrock12**: Rosenbrock 1(2) adaptive (L-stable, stiff)
- **Rosenbrock23**: Rosenbrock 2(3) adaptive (L-stable, stiff)
- **Rosenbrock34**: Rosenbrock 3(4) adaptive (L-stable, stiff)

Semi-implicit / extrapolation solvers:
- **SIS**: Semi-Implicit Solver (extrapolation, LSODA backend)
- **SEulex**: Semi-Explicit Extrapolation (Radau backend)
- **SIBS**: Semi-Implicit Bulirsch-Stoer (BDF backend)

Usage::

    from pyfoam.ode import ODESolver, create_ode_solver

    # Create by name
    solver = create_ode_solver("RK4")

    # Or use RTS directly
    solver = ODESolver.create("RKDP45", rtol=1e-8)

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
from pyfoam.ode.runge_kutta_ck_dp import RKCK45Solver, RKDP45Solver
from pyfoam.ode.implicit import TrapezoidSolver, Rosenbrock12Solver
from pyfoam.ode.rosenbrock import Rosenbrock23Solver, Rosenbrock34Solver
from pyfoam.ode.semi_implicit import SISSolver, SEulexSolver, SIBSSolver

__all__ = [
    # Base
    "ODESolver",
    "create_ode_solver",
    "ode_solver_from_dict",
    # Explicit
    "EulerSolver",
    "RK4Solver",
    "RKF45Solver",
    "RKCK45Solver",
    "RKDP45Solver",
    # Implicit / stiff
    "TrapezoidSolver",
    "Rosenbrock12Solver",
    "Rosenbrock23Solver",
    "Rosenbrock34Solver",
    # Semi-implicit / extrapolation
    "SISSolver",
    "SEulexSolver",
    "SIBSSolver",
]
