# pyOpenFOAM Roadmap

## Current Status (v0.1.0)

### What's Implemented
- ✅ Core mesh data structures (PolyMesh, FvMesh)
- ✅ Field classes (volScalarField, volVectorField, surfaceScalarField)
- ✅ OpenFOAM file format I/O (ASCII + binary)
- ✅ Boundary conditions (9 types)
- ✅ FVM discretization operators (grad, div, laplacian)
- ✅ Linear solvers (PCG, PBiCGSTAB, GAMG)
- ✅ Pressure-velocity coupling (SIMPLE, PISO, PIMPLE)
- ✅ Turbulence models (k-ε, k-ω SST, S-A, Smagorinsky, WALE)
- ✅ Thermophysical models (perfect gas, Sutherland)
- ✅ Multiphase (VOF)
- ✅ MPI parallel support
- ✅ GPU acceleration via PyTorch tensors

### Current Capabilities
- **GPU Acceleration**: All field operations use PyTorch tensors on CUDA/MPS
- **OpenFOAM Compatibility**: Read/write existing OpenFOAM cases
- **Python Integration**: Works with NumPy, SciPy, PyTorch ecosystem

### Current Limitations
- ❌ **Not end-to-end differentiable**: Discretization operators don't support `torch.autograd`
- ❌ **Simplified solvers**: Validation cases use Jacobi iteration, not full SIMPLE
- ❌ **No OpenFOAM benchmark comparison**: Haven't run identical cases in OpenFOAM

---

## Short-term Goals (v0.2.0) - Q2 2025

### 1. Validation Against OpenFOAM
- [ ] Run identical test cases in OpenFOAM (using WSL singularity container)
- [ ] Compare velocity profiles, pressure fields, residuals
- [ ] Document differences and possible causes
- [ ] Add OpenFOAM reference data to validation suite

### 2. Full SIMPLE Solver Integration
- [ ] Fix validation cases to use actual SIMPLE solver (not Jacobi)
- [ ] Implement proper boundary condition enforcement in SIMPLE
- [ ] Add turbulence model coupling to SIMPLE solver
- [ ] Validate against Ghia et al. (1982) benchmark data

### 3. Performance Optimization
- [ ] Profile GPU memory usage for large meshes
- [ ] Optimize sparse matrix operations
- [ ] Add mesh partitioning for multi-GPU
- [ ] Benchmark against OpenFOAM performance

---

## Medium-term Goals (v0.3.0) - Q4 2025

### 1. Differentiable Solver Foundation
- [ ] Implement custom `torch.autograd.Function` for key operations:
  - Face interpolation (linear, upwind)
  - Gradient computation (Gauss theorem)
  - Divergence computation
  - Laplacian computation
- [ ] Add implicit differentiation for linear solvers:
  - Differentiable PCG solver
  - Differentiable BiCGSTAB solver
  - Adjoint method for gradient computation
- [ ] Test differentiability with simple cases:
  - Differentiable Poisson equation
  - Differentiable heat equation

### 2. Physics-Informed Neural Networks (PINN) Support
- [ ] Implement PDE residual computation using autograd
- [ ] Add boundary condition loss terms
- [ ] Create PINN training examples:
  - Lid-driven cavity PINN
  - Flow around cylinder PINN
- [ ] Benchmark PINN convergence vs traditional solvers

### 3. Differentiable Turbulence Models
- [ ] Make RANS models differentiable:
  - k-ε with autograd
  - k-ω SST with autograd
- [ ] Add neural network turbulence models:
  - Neural network for Reynolds stress tensor
  - Data-driven turbulence closure
- [ ] Validate differentiable turbulence models

---

## Long-term Goals (v1.0.0) - 2026

### 1. End-to-End Differentiable CFD
- [ ] Full SIMPLE/PISO algorithm differentiable through:
  - Momentum predictor (autograd-compatible)
  - Pressure equation (implicit differentiation)
  - Velocity correction (autograd-compatible)
  - Turbulence coupling (autograd-compatible)
- [ ] Gradient-based design optimization:
  - Shape optimization
  - Parameter identification
  - Control optimization
- [ ] Adjoint solver for efficient gradient computation

### 2. Neural Operator Integration
- [ ] Fourier Neural Operator (FNO) for flow prediction
- [ ] Graph Neural Operator for unstructured meshes
- [ ] Neural operator pre-training on CFD data
- [ ] Hybrid neural-physics solvers

### 3. Multi-fidelity Simulation
- [ ] Coarse-to-fine neural network correction
- [ ] Transfer learning between mesh resolutions
- [ ] Uncertainty quantification with differentiable solvers

---

## Technical Approach for Differentiability

### Challenge
Traditional CFD solvers use:
1. Iterative linear solvers (not differentiable)
2. Pressure-velocity coupling (implicit, not differentiable)
3. Turbulence models (empirical, not differentiable)

### Solution Strategy

#### 1. Custom Autograd Functions
```python
class DifferentiableLaplacian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phi, mesh):
        # Forward: compute Laplacian using FVM
        # Save intermediate values for backward
        ctx.save_for_backward(phi, mesh.connectivity)
        return fvm.laplacian(phi, mesh)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward: compute gradient using adjoint method
        phi, connectivity = ctx.saved_tensors
        # Adjoint Laplacian is self-adjoint for symmetric operators
        return fvm.laplacian(grad_output, mesh), None
```

#### 2. Implicit Differentiation
For linear systems `Ax = b`:
```python
# Forward: solve Ax = b
x = solver(A, b)

# Backward: solve A^T λ = ∂L/∂x
# Then: ∂L/∂b = λ, ∂L/∂A = -λ x^T
```

#### 3. Differentiable Pressure-Velocity Coupling
- Use fixed-point iteration formulation
- Apply implicit function theorem
- Compute gradients through the coupled system

---

## References

### Differentiable CFD
- [1] Bezgin et al. "JAX-Fluids: A fully-differentiable high-order CFD solver" (2023)
- [2] Kochkov et al. "Machine learning-accelerated computational fluid dynamics" (2021)
- [3] Um et al. "Solver-in-the-Loop: Learning from Differentiable Physics" (2020)

### Physics-Informed Neural Networks
- [4] Raissi et al. "Physics-Informed Neural Networks" (2019)
- [5] Karniadakis et al. "Physics-informed machine learning" (2021)

### Adjoint Methods
- [6] Giles & Pierce "An introduction to the adjoint approach to design" (2000)
- [7] Jameson "Aerodynamic design via control theory" (1988)

---

## Contributing

We welcome contributions to any of these roadmap items. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Priority Areas
1. **Validation**: Help us validate against OpenFOAM
2. **Differentiability**: Implement custom autograd functions
3. **Performance**: Optimize GPU memory and computation
4. **Documentation**: Improve tutorials and examples

---

## Notes

- This roadmap is aspirational and may change based on community needs
- Differentiable CFD is an active research area - we'll adapt to new techniques
- We prioritize correctness over speed - validation is essential
- We aim to be honest about limitations and capabilities
