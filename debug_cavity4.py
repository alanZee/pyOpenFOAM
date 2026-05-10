import torch
import logging
logging.basicConfig(level=logging.WARNING)
from validation.cases.lid_driven_cavity import LidDrivenCavityCase
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.pressure_equation import assemble_pressure_equation, solve_pressure_equation
from pyfoam.solvers.rhie_chow import compute_HbyA, compute_face_flux_HbyA
from pyfoam.core.backend import scatter_add, gather

case = LidDrivenCavityCase(n_cells=4, Re=100.0, max_iterations=1, tolerance=1e-4)
case.setup()

mesh = case._mesh
U_bc = case._U_bc
U = case._U_init.clone()
p = case._p_init.clone()
phi = case._phi_init.clone()

config = SIMPLEConfig(
    relaxation_factor_U=0.7,
    relaxation_factor_p=0.3,
    nu=case.nu,
    consistent=False,
    p_tolerance=1e-6,
    p_max_iter=50,
)
solver = SIMPLESolver(mesh, config)

# Run momentum predictor
U_out, A_p, H, mat_lower, mat_upper = solver._momentum_predictor(U, p, phi, U_bc=U_bc)
print(f"After momentum predictor:")
print(f"  U_out range: [{U_out.min():.6e}, {U_out.max():.6e}]")
print(f"  A_p range: [{A_p.min():.6e}, {A_p.max():.6e}]")
print(f"  H range: [{H.min():.6e}, {H.max():.6e}]")
print(f"  Any NaN in U_out: {torch.isnan(U_out).any()}")
print(f"  Any NaN in A_p: {torch.isnan(A_p).any()}")
print(f"  Any NaN in H: {torch.isnan(H).any()}")

# Compute HbyA
HbyA = compute_HbyA(H, A_p)
bc_mask = ~torch.isnan(U_bc[:, 0])
if bc_mask.any():
    HbyA[bc_mask] = U_bc[bc_mask]
print(f"\nAfter HbyA:")
print(f"  HbyA range: [{HbyA.min():.6e}, {HbyA.max():.6e}]")
print(f"  Any NaN in HbyA: {torch.isnan(HbyA).any()}")

# Compute phiHbyA
phiHbyA = compute_face_flux_HbyA(
    HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
    mesh.n_internal_faces, mesh.face_weights,
)
print(f"\nAfter phiHbyA:")
print(f"  phiHbyA range: [{phiHbyA.min():.6e}, {phiHbyA.max():.6e}]")
print(f"  Any NaN in phiHbyA: {torch.isnan(phiHbyA).any()}")

# Assemble pressure equation
p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
print(f"\nPressure equation:")
print(f"  diag range: [{p_eqn.diag.min():.6e}, {p_eqn.diag.max():.6e}]")
print(f"  source range: [{p_eqn.source.min():.6e}, {p_eqn.source.max():.6e}]")
print(f"  Any NaN in diag: {torch.isnan(p_eqn.diag).any()}")
print(f"  Any NaN in source: {torch.isnan(p_eqn.source).any()}")

# Solve pressure equation
from pyfoam.solvers.pcg import PCGSolver
p_solver = PCGSolver(tolerance=1e-6, max_iter=50)
p_prime, p_iters, p_res = solve_pressure_equation(
    p_eqn, torch.zeros_like(p), p_solver,
    tolerance=1e-6, max_iter=50,
)
print(f"\nPressure solution:")
print(f"  p_prime range: [{p_prime.min():.6e}, {p_prime.max():.6e}]")
print(f"  p_iters: {p_iters}, p_res: {p_res:.6e}")
print(f"  Any NaN in p_prime: {torch.isnan(p_prime).any()}")
