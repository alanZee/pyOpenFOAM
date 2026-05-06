"""
pyfoam.parallel — Domain decomposition and parallel solving via MPI.

Provides:

- :class:`Decomposition` — split a mesh into subdomains (geometric or Scotch)
- :class:`SubDomain` — a subregion of the global mesh with ghost cell mapping
- :class:`ProcessorPatch` — describes ghost cells on a processor boundary
- :class:`HaloExchange` — inter-processor ghost cell communication
- :class:`ParallelField` — parallel-aware field with gather/scatter/reduce
- :class:`ParallelSolver` — domain-decomposed solver wrapper
- :class:`ParallelWriter` / :class:`ParallelReader` — processor directory I/O

All operations respect the global device/dtype from :mod:`pyfoam.core`.
MPI is optional — all operations have serial fallbacks for testing.
"""

from pyfoam.parallel.decomposition import Decomposition, SubDomain
from pyfoam.parallel.processor_patch import ProcessorPatch, HaloExchange
from pyfoam.parallel.parallel_field import ParallelField
from pyfoam.parallel.parallel_io import ParallelReader, ParallelWriter
from pyfoam.parallel.parallel_solver import ParallelSolver, ParallelSolverConfig

__all__ = [
    # Decomposition
    "Decomposition",
    "SubDomain",
    # Processor patches
    "ProcessorPatch",
    "HaloExchange",
    # Parallel field
    "ParallelField",
    # Parallel I/O
    "ParallelWriter",
    "ParallelReader",
    # Parallel solver
    "ParallelSolver",
    "ParallelSolverConfig",
]
