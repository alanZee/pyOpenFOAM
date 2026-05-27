"""Inline test for mesh operations."""
import torch, sys
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.refine_mesh import refine_mesh, split_face, _PointManager
from pyfoam.tools.split_mesh_regions import split_mesh_regions
from pyfoam.tools.merge_meshes import merge_meshes

def _h1():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    return FvMesh(points=torch.tensor(pts, dtype=torch.float64), faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
                  owner=torch.zeros(6, dtype=INDEX_DTYPE), neighbour=torch.tensor([], dtype=INDEX_DTYPE),
                  boundary=[{"name":"all","type":"wall","startFace":0,"nFaces":6}])

def _h2z():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,2],[1,0,2],[1,1,2],[0,1,2]]
    fc = [[4,5,6,7],[0,3,2,1],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5],[8,9,10,11],[4,5,9,8],[7,11,10,6],[4,8,11,7],[5,6,10,9]]
    return FvMesh(points=torch.tensor(pts, dtype=torch.float64), faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
                  owner=torch.tensor([0,0,0,0,0,0,1,1,1,1,1], dtype=INDEX_DTYPE), neighbour=torch.tensor([1], dtype=INDEX_DTYPE),
                  boundary=[{"name":"b","type":"wall","startFace":1,"nFaces":5},{"name":"t","type":"wall","startFace":6,"nFaces":5}])

def _hx(o=(0,0,0)):
    ox,oy,oz = o
    pts = [[ox,oy,oz],[ox+1,oy,oz],[ox+1,oy+1,oz],[ox,oy+1,oz],[ox,oy,oz+1],[ox+1,oy,oz+1],[ox+1,oy+1,oz+1],[ox,oy+1,oz+1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    return FvMesh(points=torch.tensor(pts, dtype=torch.float64), faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
                  owner=torch.zeros(6, dtype=INDEX_DTYPE), neighbour=torch.tensor([], dtype=INDEX_DTYPE),
                  boundary=[{"name":"all","type":"wall","startFace":0,"nFaces":6}])

def _2x2():
    nx, ny, nz = 2, 2, 1
    pts = [[i,j,k] for k in range(nz+1) for j in range(ny+1) for i in range(nx+1)]
    def pi(i,j,k): return k*(ny+1)*(nx+1)+j*(nx+1)+i
    faces, owner, nbr = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx-1):
                faces.append([pi(i+1,j,k),pi(i+1,j+1,k),pi(i+1,j+1,k+1),pi(i+1,j,k+1)]); owner.append(k*ny*nx+j*nx+i); nbr.append(k*ny*nx+j*nx+i+1)
    for k in range(nz):
        for j in range(ny-1):
            for i in range(nx):
                faces.append([pi(i,j+1,k),pi(i+1,j+1,k),pi(i+1,j+1,k+1),pi(i,j+1,k+1)]); owner.append(k*ny*nx+j*nx+i); nbr.append(k*ny*nx+(j+1)*nx+i)
    ni = len(faces)
    for j in range(ny):
        for i in range(nx): faces.append([pi(i,j,0),pi(i+1,j,0),pi(i+1,j+1,0),pi(i,j+1,0)]); owner.append(j*nx+i)
    for j in range(ny):
        for i in range(nx): faces.append([pi(i,j,1),pi(i,j+1,1),pi(i+1,j+1,1),pi(i+1,j,1)]); owner.append(j*nx+i)
    for k in range(nz):
        for i in range(nx): faces.append([pi(i,0,k),pi(i,0,k+1),pi(i+1,0,k+1),pi(i+1,0,k)]); owner.append(k*nx+i)
    for k in range(nz):
        for i in range(nx): faces.append([pi(i,ny,k),pi(i+1,ny,k),pi(i+1,ny,k+1),pi(i,ny,k+1)]); owner.append(k*ny*nx+(ny-1)*nx+i)
    for k in range(nz):
        for j in range(ny): faces.append([pi(0,j,k),pi(0,j+1,k),pi(0,j+1,k+1),pi(0,j,k+1)]); owner.append(k*ny*nx+j*nx)
    for k in range(nz):
        for j in range(ny): faces.append([pi(nx,j,k),pi(nx,j,k+1),pi(nx,j+1,k+1),pi(nx,j+1,k)]); owner.append(k*ny*nx+j*nx+(nx-1))
    bnd = [{"name":"bottom","type":"wall","startFace":ni,"nFaces":4},{"name":"top","type":"wall","startFace":ni+4,"nFaces":4},
           {"name":"front","type":"wall","startFace":ni+8,"nFaces":2},{"name":"back","type":"wall","startFace":ni+10,"nFaces":2},
           {"name":"left","type":"wall","startFace":ni+12,"nFaces":2},{"name":"right","type":"wall","startFace":ni+14,"nFaces":2}]
    return FvMesh(points=torch.tensor(pts, dtype=torch.float64), faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in faces],
                  owner=torch.tensor(owner, dtype=INDEX_DTYPE), neighbour=torch.tensor(nbr, dtype=INDEX_DTYPE), boundary=bnd)

passed = failed = 0; errors = []
def check(name, cond, msg=""):
    global passed, failed
    if cond: passed += 1
    else: failed += 1; errors.append(f"{name}: {msg}")

# refine_mesh
try:
    r = refine_mesh(_h1(), [0], "all")
    check("refine_type", isinstance(r, FvMesh))
    check("refine_all_8", r.n_cells == 8, f"n_cells={r.n_cells}")
    ok = all(r.owner[i].item() < r.neighbour[i].item() for i in range(r.n_internal_faces))
    check("refine_owner_lt_all", ok)
    r.compute_geometry()
    check("refine_vol_all", abs(r.total_volume.item() - 1.0) < 1e-6, f"vol={r.total_volume.item()}")
    check("refine_sub_cell_eq", all(abs(v - 0.125) < 1e-6 for v in r.cell_volumes.detach().cpu().numpy()))
except Exception as e:
    failed += 1; errors.append(f"refine_all: {e}")

for d in ["x", "y", "z"]:
    try:
        r = refine_mesh(_h1(), [0], d)
        check(f"refine_{d}_2", r.n_cells == 2, f"n_cells={r.n_cells}")
        r.compute_geometry()
        check(f"refine_{d}_vol", abs(r.total_volume.item() - 1.0) < 1e-6, f"vol={r.total_volume.item()}")
    except Exception as e:
        failed += 1; errors.append(f"refine_{d}: {e}")

try:
    m = _h1(); r = refine_mesh(m, [], "all")
    check("refine_empty", r.n_cells == m.n_cells)
except Exception as e:
    failed += 1; errors.append(f"refine_empty: {e}")

try:
    r = refine_mesh(_h2z(), [0], "all")
    check("refine_one_of_two", r.n_cells == 9, f"n_cells={r.n_cells}")
except Exception as e:
    failed += 1; errors.append(f"refine_one_of_two: {e}")

try:
    check("refine_tensor", refine_mesh(_h1(), torch.tensor([0], dtype=INDEX_DTYPE), "x").n_cells == 2)
except Exception as e:
    failed += 1; errors.append(f"refine_tensor: {e}")

# split_mesh_regions
try:
    r = split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]})
    check("split_dict", isinstance(r, dict) and len(r) == 2)
    check("split_cells", r["l"].n_cells == 2 and r["r"].n_cells == 2)
    check("split_total", sum(s.n_cells for s in r.values()) == 4)
    check("split_single", len(split_mesh_regions(_2x2(), {"all":[0,1,2,3]})) == 1)
    for s in r.values():
        check("split_topo", s.owner.min().item() >= 0 and s.owner.max().item() < s.n_cells)
    check("split_int", r["l"].n_internal_faces >= 1)
    m = _2x2(); r = split_mesh_regions(m, {"l":[0,2],"r":[1,3]})
    check("split_vol", abs(sum(s.cell_volumes.sum().item() for s in r.values()) - m.total_volume.item()) < 1e-10)
except Exception as e:
    failed += 1; errors.append(f"split: {e}")

try:
    split_mesh_regions(_2x2(), {"a":[0,1],"b":[1,2]})
    check("split_overlap", False, "should have raised")
except ValueError:
    check("split_overlap", True)

# merge_meshes
try:
    r = merge_meshes([_hx(), _hx((2,0,0))])
    check("merge_type", isinstance(r, FvMesh))
    check("merge_disjoint", r.n_cells == 2)
    check("merge_disjoint_f", r.n_faces == 12)
    check("merge_single", merge_meshes([_hx()]).n_cells == 1)
except Exception as e:
    failed += 1; errors.append(f"merge_basic: {e}")

try:
    r = merge_meshes([_hx((0,0,0)), _hx((1,0,0))])
    check("merge_shared", r.n_internal_faces >= 1 and r.n_cells == 2)
    check("merge_pts", r.points.shape[0] == 12)
    ok = all(r.owner[i].item() < r.neighbour[i].item() for i in range(r.n_internal_faces))
    check("merge_owner_lt", ok)
    r.compute_geometry()
    check("merge_vol", abs(r.total_volume.item() - 2.0) < 1e-10)
except Exception as e:
    failed += 1; errors.append(f"merge_shared: {e}")

try:
    merge_meshes([])
    check("merge_empty", False, "should have raised")
except ValueError:
    check("merge_empty", True)

try:
    ms = [_hx((i,0,0)) for i in range(3)]
    r = merge_meshes(ms); r.compute_geometry()
    check("merge_three", r.n_cells == 3)
    check("merge_three_vol", abs(r.total_volume.item() - 3.0) < 1e-10)
except Exception as e:
    failed += 1; errors.append(f"merge_three: {e}")

print(f"\n=== Results: {passed} passed, {failed} failed ===")
for e in errors: print(f"  FAIL: {e}")
sys.exit(1 if failed > 0 else 0)
