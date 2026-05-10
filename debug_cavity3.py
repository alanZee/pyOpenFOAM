import torch
import logging
logging.basicConfig(level=logging.DEBUG)
from validation.cases.lid_driven_cavity import LidDrivenCavityCase

case = LidDrivenCavityCase(n_cells=4, Re=100.0, max_iterations=3, tolerance=1e-4)
case.setup()
result = case.run()
print(f'Converged: {result["converged"]}')
print(f'Iterations: {result["iterations"]}')
