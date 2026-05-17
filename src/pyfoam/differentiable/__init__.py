"""
pyfoam.differentiable — Differentiable CFD operators.

Provides ``torch.autograd.Function`` subclasses for finite volume
discretisation operators, enabling end-to-end differentiable CFD
simulations with PyTorch's autograd.

Operators
---------
DifferentiableGradient
    Gradient operator ∇φ with correct backward pass.
DifferentiableDivergence
    Divergence operator ∇·(φU) with correct backward pass.
DifferentiableLaplacian
    Laplacian operator ∇·(D∇φ) with correct backward pass.
DifferentiableLinearSolve
    Linear system solver Ax = b with implicit differentiation.
DifferentiableSIMPLE
    SIMPLE algorithm with fixed-point iteration differentiation.
"""

from pyfoam.differentiable.operators import (
    DifferentiableGradient,
    DifferentiableDivergence,
    DifferentiableLaplacian,
)
from pyfoam.differentiable.linear_solver import DifferentiableLinearSolve
from pyfoam.differentiable.simple import DifferentiableSIMPLE

__all__ = [
    "DifferentiableGradient",
    "DifferentiableDivergence",
    "DifferentiableLaplacian",
    "DifferentiableLinearSolve",
    "DifferentiableSIMPLE",
]
