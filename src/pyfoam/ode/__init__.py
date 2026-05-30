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
from pyfoam.ode.ode_solvers_v2 import (
    RKCK45Solver_v2,
    RKDP45Solver_v2,
    Rosenbrock12Solver_v2,
    Rosenbrock23Solver_v2,
    Rosenbrock34Solver_v2,
    SISSolver_v2,
    SEulexSolver_v2,
)
from pyfoam.ode.ode_solvers_v3 import (
    RKCK45Solver_v3,
    RKDP45Solver_v3,
    Rosenbrock12Solver_v3,
    Rosenbrock23Solver_v3,
    Rosenbrock34Solver_v3,
    SISSolver_v3,
    SEulexSolver_v3,
)
from pyfoam.ode.ode_solvers_v4 import (
    RKCK45Solver_v4,
    RKDP45Solver_v4,
    Rosenbrock12Solver_v4,
    Rosenbrock23Solver_v4,
    Rosenbrock34Solver_v4,
    SISSolver_v4,
    SEulexSolver_v4,
)
from pyfoam.ode.ode_solvers_v5 import (
    RKCK45Solver_v5,
    RKDP45Solver_v5,
    Rosenbrock12Solver_v5,
    Rosenbrock23Solver_v5,
    Rosenbrock34Solver_v5,
    SISSolver_v5,
    SEulexSolver_v5,
)
from pyfoam.ode.ode_solvers_v6 import (
    RKCK45Solver_v6,
    RKDP45Solver_v6,
    Rosenbrock12Solver_v6,
    Rosenbrock23Solver_v6,
    Rosenbrock34Solver_v6,
    SISSolver_v6,
    SEulexSolver_v6,
)
from pyfoam.ode.ode_solvers_v7 import (
    RKCK45Solver_v7,
    RKDP45Solver_v7,
    Rosenbrock12Solver_v7,
    Rosenbrock23Solver_v7,
    Rosenbrock34Solver_v7,
    SISSolver_v7,
    SEulexSolver_v7,
)
from pyfoam.ode.ode_solvers_v8 import (
    RKCK45Solver_v8,
    RKDP45Solver_v8,
    Rosenbrock12Solver_v8,
    Rosenbrock23Solver_v8,
    Rosenbrock34Solver_v8,
    SISSolver_v8,
    SEulexSolver_v8,
)
from pyfoam.ode.ode_solvers_v9 import (
    RKCK45Solver_v9,
    RKDP45Solver_v9,
    Rosenbrock12Solver_v9,
    Rosenbrock23Solver_v9,
    Rosenbrock34Solver_v9,
    SISSolver_v9,
    SEulexSolver_v9,
)

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
    # Explicit v2
    "RKCK45Solver_v2",
    "RKDP45Solver_v2",
    # Implicit / stiff
    "TrapezoidSolver",
    "Rosenbrock12Solver",
    "Rosenbrock23Solver",
    "Rosenbrock34Solver",
    # Implicit / stiff v2
    "Rosenbrock12Solver_v2",
    "Rosenbrock23Solver_v2",
    "Rosenbrock34Solver_v2",
    # Semi-implicit / extrapolation
    "SISSolver",
    "SEulexSolver",
    "SIBSSolver",
    # Semi-implicit / extrapolation v2
    "SISSolver_v2",
    "SEulexSolver_v2",
    # Explicit v3
    "RKCK45Solver_v3",
    "RKDP45Solver_v3",
    # Implicit / stiff v3
    "Rosenbrock12Solver_v3",
    "Rosenbrock23Solver_v3",
    "Rosenbrock34Solver_v3",
    # Semi-implicit / extrapolation v3
    "SISSolver_v3",
    "SEulexSolver_v3",
    # Explicit v4
    "RKCK45Solver_v4",
    "RKDP45Solver_v4",
    # Implicit / stiff v4
    "Rosenbrock12Solver_v4",
    "Rosenbrock23Solver_v4",
    "Rosenbrock34Solver_v4",
    # Semi-implicit / extrapolation v4
    "SISSolver_v4",
    "SEulexSolver_v4",
    # Explicit v5
    "RKCK45Solver_v5",
    "RKDP45Solver_v5",
    # Implicit / stiff v5
    "Rosenbrock12Solver_v5",
    "Rosenbrock23Solver_v5",
    "Rosenbrock34Solver_v5",
    # Semi-implicit / extrapolation v5
    "SISSolver_v5",
    "SEulexSolver_v5",
    # Explicit v6
    "RKCK45Solver_v6",
    "RKDP45Solver_v6",
    # Implicit / stiff v6
    "Rosenbrock12Solver_v6",
    "Rosenbrock23Solver_v6",
    "Rosenbrock34Solver_v6",
    # Semi-implicit / extrapolation v6
    "SISSolver_v6",
    "SEulexSolver_v6",
    # Explicit v7
    "RKCK45Solver_v7",
    "RKDP45Solver_v7",
    # Implicit / stiff v7
    "Rosenbrock12Solver_v7",
    "Rosenbrock23Solver_v7",
    "Rosenbrock34Solver_v7",
    # Semi-implicit / extrapolation v7
    "SISSolver_v7",
    "SEulexSolver_v7",
    # Explicit v8
    "RKCK45Solver_v8",
    "RKDP45Solver_v8",
    # Implicit / stiff v8
    "Rosenbrock12Solver_v8",
    "Rosenbrock23Solver_v8",
    "Rosenbrock34Solver_v8",
    # Semi-implicit / extrapolation v8
    "SISSolver_v8",
    "SEulexSolver_v8",
    # Explicit v9
    "RKCK45Solver_v9",
    "RKDP45Solver_v9",
    # Implicit / stiff v9
    "Rosenbrock12Solver_v9",
    "Rosenbrock23Solver_v9",
    "Rosenbrock34Solver_v9",
    # Semi-implicit / extrapolation v9
    "SISSolver_v9",
    "SEulexSolver_v9",
]
