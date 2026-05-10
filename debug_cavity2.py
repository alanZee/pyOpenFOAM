import torch
import logging
logging.basicConfig(level=logging.INFO)
from pyfoam.mesh import PolyMesh, FvMesh
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.pressure_equation import assemble_pressure_equation
from pyfoam.core.backend import scatter_add, gather
from validation.cases.lid_driven_cavity import LidDrivenCavityCase

case = LidDrivenCavityCase(n_cells=4, Re=100.0, max_iterations=2, tolerance=1e-4)
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

# Manually run one iteration
n_cells = mesh.n_cells
n_internal = mesh.n_internal_faces
n_faces = mesh.n_faces
owner = mesh.owner
neighbour = mesh.neighbour
cell_volumes = mesh.cell_volumes
nu = config.nu

# Check what happens in _momentum_predictor
face_areas = mesh.face_areas
delta_coeffs = mesh.delta_coefficients
cell_volumes_safe = cell_volumes.clamp(min=1e-30)

# Internal face diffusion
S_mag = face_areas[:n_internal].norm(dim=1)
delta_f = delta_coeffs[:n_internal]
diff_coeff = nu * S_mag * delta_f
print(f"Internal diff_coeff: [{diff_coeff.min():.6e}, {diff_coeff.max():.6e}]")

# Boundary face diffusion
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
print(f"Boundary bnd_delta: [{bnd_delta.min():.6e}, {bnd_delta.max():.6e}]")
print(f"Boundary bnd_face_coeff: [{bnd_face_coeff.min():.6e}, {bnd_face_coeff.max():.6e}]")

# Compute diag with BCs
diag = torch.zeros(n_cells, dtype=torch.float64)
int_owner = owner[:n_internal]
int_neigh = neighbour
diag = diag + scatter_add((diff_coeff) / gather(cell_volumes_safe, int_owner), int_owner, n_cells)
diag = diag + scatter_add((diff_coeff) / gather(cell_volumes_safe, int_neigh), int_neigh, n_cells)
print(f"Diag before BC: [{diag.min():.6e}, {diag.max():.6e}]")

bnd_bc_mask = bc_mask[bnd_owner]
bnd_face_coeff_masked = bnd_face_coeff * bnd_bc_mask.float()
diag_bc = scatter_add(bnd_face_coeff_masked, bnd_owner, n_cells)
print(f"Diag BC contribution: [{diag_bc.min():.6e}, {diag_bc.max():.6e}]")
diag = diag + diag_bc
print(f"Diag after BC: [{diag.min():.6e}, {diag.max():.6e}]")

# Now check pressure equation
phiHbyA = torch.zeros(n_faces, dtype=torch.float64)
A_p = diag.clone()
p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
print(f"Pressure eqn diag: [{p_eqn.diag.min():.6e}, {p_eqn.diag.max():.6e}]")
print(f"Pressure eqn source: [{p_eqn.source.min():.6e}, {p_eqn.source.max():.6e}]")
print(f"Any NaN in diag: {torch.isnan(p_eqn.diag).any()}")
print(f"Any NaN in source: {torch.isnan(p_eqn.source).any()}")
print(f"Any inf in diag: {torch.isinf(p_eqn.diag).any()}")
print(f"Any inf in source: {torch.isinf(p_eqn.source).any()}")
