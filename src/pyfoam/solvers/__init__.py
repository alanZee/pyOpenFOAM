"""
pyfoam.solvers — Iterative linear solvers and pressure-velocity coupling.

Provides:

Linear solvers:
- **PCG**: Preconditioned Conjugate Gradient (symmetric positive-definite)
- **PBiCGSTAB**: Preconditioned Bi-Conjugate Gradient Stabilised (asymmetric)
- **GAMG**: Algebraic Multigrid with aggregation coarsening

Preconditioners:
- **DIC**: Diagonal Incomplete Cholesky (for symmetric matrices)
- **DILU**: Diagonal Incomplete LU (for general matrices)

Pressure-velocity coupled solvers:
- **SIMPLE**: Semi-Implicit Method for Pressure-Linked Equations (steady-state)
- **PISO**: Pressure-Implicit with Splitting of Operators (transient)
- **PIMPLE**: PISO + SIMPLE hybrid (transient, large time steps)

Support modules:
- **Rhie-Chow interpolation**: Prevents checkerboard pressure oscillations
- **Pressure equation assembly**: Builds the pressure Poisson equation

Usage::

    from pyfoam.solvers import create_solver, PCGSolver, PBiCGSTABSolver

    # Create solver by name
    solver = create_solver("PCG", tolerance=1e-6, max_iter=1000)

    # Or instantiate directly
    solver = PCGSolver(tolerance=1e-6)

    # Solve
    solution, iterations, residual = solver(matrix, source, x0)

    # Coupled solvers
    from pyfoam.solvers import SIMPLESolver, PISOSolver, PIMPLESolver

    solver = SIMPLESolver(mesh)
    U, p, phi, convergence = solver.solve(U, p, phi)
"""

from pyfoam.solvers.linear_solver import (
    LinearSolverBase,
    create_solver,
    solver_from_dict,
)
from pyfoam.solvers.pcg import PCGSolver
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver
from pyfoam.solvers.gamg import GAMGSolver
from pyfoam.solvers.preconditioners import (
    DICPreconditioner,
    DILUPreconditioner,
    Preconditioner,
)
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor
from pyfoam.solvers.coupled_solver import (
    CoupledSolverBase,
    CoupledSolverConfig,
)
from pyfoam.solvers.rhie_chow import (
    compute_HbyA,
    compute_face_flux_HbyA,
    rhie_chow_correction,
    compute_face_flux,
)
from pyfoam.solvers.pressure_equation import (
    assemble_pressure_equation,
    solve_pressure_equation,
    correct_velocity,
    correct_face_flux,
)
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig

__all__ = [
    # Linear solver classes
    "LinearSolverBase",
    "PCGSolver",
    "PBiCGSTABSolver",
    "GAMGSolver",
    # Preconditioners
    "Preconditioner",
    "DICPreconditioner",
    "DILUPreconditioner",
    # Factory functions
    "create_solver",
    "solver_from_dict",
    # Monitoring
    "ResidualMonitor",
    "ConvergenceInfo",
    # Coupled solver base
    "CoupledSolverBase",
    "CoupledSolverConfig",
    # Rhie-Chow interpolation
    "compute_HbyA",
    "compute_face_flux_HbyA",
    "rhie_chow_correction",
    "compute_face_flux",
    # Pressure equation
    "assemble_pressure_equation",
    "solve_pressure_equation",
    "correct_velocity",
    "correct_face_flux",
    # SIMPLE
    "SIMPLESolver",
    "SIMPLEConfig",
    # PISO
    "PISOSolver",
    "PISOConfig",
    # PIMPLE
    "PIMPLESolver",
    "PIMPLEConfig",
]
