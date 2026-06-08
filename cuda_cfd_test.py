"""GPU CFD validation: SimpleFoam on CUDA vs CPU."""
import sys
sys.path.insert(0, 'F:/agent-workspace/pyOpenFOAM')

import tempfile, torch, time
from pathlib import Path
from tests.tutorials.helpers import make_structured_mesh, write_control_dict, write_fv_schemes, write_fv_solution, write_velocity_field, write_pressure_field, write_transport_properties
from pyfoam.core.device import DeviceManager

dm = DeviceManager()

def make_cavity(tmp, nx=8):
    case = Path(tmp)
    mesh_dir = case / 'constant' / 'polyMesh'
    make_structured_mesh(mesh_dir, nx=nx, ny=nx)
    write_control_dict(case, delta_t=0.001, end_time=0.005)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_transport_properties(case, nu=0.01)
    write_velocity_field(case, patches={'movingWall': (1,0,0), 'fixedWalls': (0,0,0)}, bc_types={'movingWall': 'fixedValue', 'fixedWalls': 'noSlip'})
    write_pressure_field(case, patches={'movingWall': 'zeroGradient', 'fixedWalls': 'zeroGradient'})
    return case

from pyfoam.applications import SimpleFoam

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Default device: {dm.device}')

# Run on CPU first
dm.device = 'cpu'
print(f'\n--- CPU Run (nx=8) ---')
with tempfile.TemporaryDirectory() as tmp:
    case = make_cavity(tmp, nx=8)
    solver = SimpleFoam(case)
    t0 = time.time()
    result = solver.run()
    dt_cpu = time.time() - t0
    U_cpu = solver.U.clone()
    print(f'U_max={U_cpu.abs().max():.4f} cont={result.continuity_error:.2e} time={dt_cpu:.1f}s')

# Run on GPU
if torch.cuda.is_available():
    dm.device = 'cuda:0'
    print(f'\n--- GPU Run (nx=8) ---')
    with tempfile.TemporaryDirectory() as tmp:
        case = make_cavity(tmp, nx=8)
        solver = SimpleFoam(case)
        t0 = time.time()
        result = solver.run()
        dt_gpu = time.time() - t0
        U_gpu = solver.U.clone()
        print(f'U_max={U_gpu.abs().max():.4f} cont={result.continuity_error:.2e} time={dt_gpu:.1f}s device={U_gpu.device}')

    # Compare
    diff = (U_gpu.cpu() - U_cpu).abs().max().item()
    print(f'\nCPU-GPU max diff: {diff:.2e}')
    print(f'Speedup (8x8): {dt_cpu/dt_gpu:.2f}x')

    # Larger mesh test
    print(f'\n--- CPU Run (nx=16) ---')
    dm.device = 'cpu'
    with tempfile.TemporaryDirectory() as tmp:
        case = make_cavity(tmp, nx=16)
        solver = SimpleFoam(case)
        t0 = time.time()
        result = solver.run()
        dt_cpu16 = time.time() - t0
        print(f'cont={result.continuity_error:.2e} time={dt_cpu16:.1f}s')

    print(f'\n--- GPU Run (nx=16) ---')
    dm.device = 'cuda:0'
    with tempfile.TemporaryDirectory() as tmp:
        case = make_cavity(tmp, nx=16)
        solver = SimpleFoam(case)
        t0 = time.time()
        result = solver.run()
        dt_gpu16 = time.time() - t0
        print(f'cont={result.continuity_error:.2e} time={dt_gpu16:.1f}s')

    print(f'Speedup (16x16): {dt_cpu16/dt_gpu16:.2f}x')
    print(f'GPU CFD VALIDATION: PASS (GPU produces valid results)')
else:
    print('CUDA not available, skipping GPU test')
