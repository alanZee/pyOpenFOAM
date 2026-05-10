import torch
import logging
logging.basicConfig(level=logging.INFO)
from validation.cases.lid_driven_cavity import LidDrivenCavityCase

case = LidDrivenCavityCase(n_cells=16, Re=100.0, max_iterations=500, tolerance=1e-4)
case.setup()
result = case.run()

print(f'Converged: {result["converged"]}')
print(f'Iterations: {result["iterations"]}')
print(f'Final residual: {result["final_residual"]:.6e}')

# Check velocity at y=0.5
U = case.get_computed()['U']
nx, ny = 16, 16
j_mid = ny // 2
u_vals = [U[j_mid * nx + i, 0].item() for i in range(nx)]
u_min = min(u_vals)
print(f'Peak u at y=0.5: {u_min:.6f}')
print(f'Expected: ~-0.20581')
print(f'Ratio: {abs(u_min) / 0.20581:.2f}x')
print(f'L2 error target: < 15%')
