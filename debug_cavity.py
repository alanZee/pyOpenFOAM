import torch
import logging
logging.basicConfig(level=logging.DEBUG)
from validation.cases.lid_driven_cavity import LidDrivenCavityCase
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

case = LidDrivenCavityCase(n_cells=4, Re=100.0, max_iterations=2, tolerance=1e-4)
case.setup()

mesh = case._mesh
U_bc = case._U_bc
U = case._U_init.clone()
p = case._p_init.clone()

# Check boundary face properties
n_internal = mesh.n_internal_faces
n_faces = mesh.n_faces
bnd_owner = mesh.owner[n_internal:]
bnd_areas = mesh.face_areas[n_internal:]
bnd_face_centres = mesh.face_centres[n_internal:]
owner_centres = mesh.cell_centres[bnd_owner]
d_P = bnd_face_centres - owner_centres
bnd_S_mag = bnd_areas.norm(dim=1)
safe_S_mag = torch.where(bnd_S_mag > 1e-30, bnd_S_mag, torch.ones_like(bnd_S_mag))
n_hat = bnd_areas / safe_S_mag.unsqueeze(-1)
d_dot_n = (d_P * n_hat).sum(dim=1)

print(f"n_internal={n_internal}, n_faces={n_faces}, n_boundary={n_faces-n_internal}")
print(f"bnd_S_mag range: [{bnd_S_mag.min():.6e}, {bnd_S_mag.max():.6e}]")
print(f"d_dot_n range: [{d_dot_n.min():.6e}, {d_dot_n.max():.6e}]")
print(f"d_dot_n (first 10): {d_dot_n[:10].tolist()}")

# Check BC mask
bc_mask = ~torch.isnan(U_bc[:, 0])
print(f"BC cells: {bc_mask.sum().item()} / {mesh.n_cells}")
print(f"bnd_owner (first 10): {bnd_owner[:10].tolist()}")
print(f"bnd_bc_mask (first 10): {bc_mask[bnd_owner[:10]].tolist()}")

# Check nu
print(f"nu={case.nu}")

# Check cell volumes
print(f"cell_volumes: [{mesh.cell_volumes.min():.6e}, {mesh.cell_volumes.max():.6e}]")
