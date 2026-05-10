import torch
import logging
logging.basicConfig(level=logging.WARNING)
from validation.cases.lid_driven_cavity import LidDrivenCavityCase
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.fv_matrix import FvMatrix

case = LidDrivenCavityCase(n_cells=4, Re=100.0, max_iterations=1, tolerance=1e-4)
case.setup()

mesh = case._mesh
U_bc = case._U_bc
U = case._U_init.clone()
p = case._p_init.clone()
phi = case._phi_init.clone()

config = SIMPLEConfig(relaxation_factor_U=0.7, relaxation_factor_p=0.3, nu=case.nu, consistent=False)

# Manually replicate _momentum_predictor
dtype = torch.float64
device = torch.device("cpu")
n_cells = mesh.n_cells
n_internal = mesh.n_internal_faces
n_faces = mesh.n_faces
owner = mesh.owner
neighbour = mesh.neighbour
cell_volumes = mesh.cell_volumes
nu = config.nu
alpha_U = config.relaxation_factor_U

face_areas = mesh.face_areas
delta_coeffs = mesh.delta_coefficients
cell_volumes_safe = cell_volumes.clamp(min=1e-30)

# Build matrix
mat = FvMatrix(n_cells, owner[:n_internal], neighbour, device=device, dtype=dtype)
int_owner = owner[:n_internal]
int_neigh = neighbour

S_mag = face_areas[:n_internal].norm(dim=1)
delta_f = delta_coeffs[:n_internal]
diff_coeff = nu * S_mag * delta_f

flux = phi[:n_internal]
flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

V_P = gather(cell_volumes_safe, int_owner)
V_N = gather(cell_volumes_safe, int_neigh)

mat.lower = (-diff_coeff + flux_neg) / V_P
mat.upper = (-diff_coeff - flux_pos) / V_N

diag = torch.zeros(n_cells, dtype=dtype, device=device)
diag = diag + scatter_add((diff_coeff - flux_neg) / V_P, int_owner, n_cells)
diag = diag + scatter_add((diff_coeff + flux_pos) / V_N, int_neigh, n_cells)

# H_old from U=0
H_old = torch.zeros(n_cells, 3, dtype=dtype, device=device)

# grad_p from p=0
grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)

source = H_old - grad_p
print(f"After H_old - grad_p: source range=[{source.min():.6e}, {source.max():.6e}]")

# BC enforcement
bc_mask = ~torch.isnan(U_bc[:, 0])
bnd_owner = owner[n_internal:]
bnd_areas = face_areas[n_internal:]
bnd_face_centres = mesh.face_centres[n_internal:]
owner_centres = mesh.cell_centres[bnd_owner]
d_P = bnd_face_centres - owner_centres
bnd_S_mag = bnd_areas.norm(dim=1)
safe_S_mag = torch.where(bnd_S_mag > 1e-30, bnd_S_mag, torch.ones_like(bnd_S_mag))
n_hat = bnd_areas / safe_S_mag.unsqueeze(-1)
d_dot_n = (d_P * n_hat).sum(dim=1).abs()
bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)
bnd_face_coeff = nu * bnd_S_mag * bnd_delta

bnd_bc_mask = bc_mask[bnd_owner]
bnd_face_coeff_masked = bnd_face_coeff * bnd_bc_mask.float()

diag = diag + scatter_add(bnd_face_coeff_masked, bnd_owner, n_cells)
for comp in range(3):
    source_contrib = bnd_face_coeff_masked * U_bc[bnd_owner, comp]
    source[:, comp] = source[:, comp] + scatter_add(source_contrib, bnd_owner, n_cells)

print(f"After BC: diag=[{diag.min():.6e}, {diag.max():.6e}]")
print(f"After BC: source=[{source.min():.6e}, {source.max():.6e}]")

# Relaxation
sum_off = torch.zeros(n_cells, dtype=dtype, device=device)
sum_off = sum_off + scatter_add(mat.lower.abs(), int_owner, n_cells)
sum_off = sum_off + scatter_add(mat.upper.abs(), int_neigh, n_cells)
D_dominant = torch.max(diag.abs(), sum_off)
D_new = D_dominant / alpha_U
mat.diag = D_new
source = source + (D_new - diag).unsqueeze(-1) * U

print(f"After relaxation: D_new=[{D_new.min():.6e}, {D_new.max():.6e}]")
print(f"After relaxation: source=[{source.min():.6e}, {source.max():.6e}]")

# Solve
from pyfoam.solvers.pcg import PCGSolver
U_solver = PCGSolver(tolerance=1e-4, max_iter=100)

for comp in range(3):
    mat.source = source[:, comp]
    U_comp, iters, res = mat.solve(U_solver, U[:, comp], tolerance=1e-4, max_iter=100)
    print(f"Comp {comp}: U range=[{U_comp.min():.6e}, {U_comp.max():.6e}], iters={iters}, res={res:.6e}, NaN={torch.isnan(U_comp).any()}")
