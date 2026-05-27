"""Tests for split_mesh_regions."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.split_mesh_regions import split_mesh_regions

def _2x2():
    nx, ny, nz = 2, 2, 1
    pts = [[i,j,k] for k in range(nz+1) for j in range(ny+1) for i in range(nx+1)]
    def pi(i,j,k): return k*(ny+1)*(nx+1)+j*(nx+1)+i
    faces, owner, nbr = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx-1):
                faces.append([pi(i+1,j,k),pi(i+1,j+1,k),pi(i+1,j+1,k+1),pi(i+1,j,k+1)])
                owner.append(k*ny*nx+j*nx+i); nbr.append(k*ny*nx+j*nx+i+1)
    for k in range(nz):
        for j in range(ny-1):
            for i in range(nx):
                faces.append([pi(i,j+1,k),pi(i+1,j+1,k),pi(i+1,j+1,k+1),pi(i,j+1,k+1)])
                owner.append(k*ny*nx+j*nx+i); nbr.append(k*ny*nx+(j+1)*nx+i)
    ni = len(faces)
    for j in range(ny):
        for i in range(nx):
            faces.append([pi(i,j,0),pi(i+1,j,0),pi(i+1,j+1,0),pi(i,j+1,0)]); owner.append(j*nx+i)
    for j in range(ny):
        for i in range(nx):
            faces.append([pi(i,j,1),pi(i,j+1,1),pi(i+1,j+1,1),pi(i+1,j,1)]); owner.append(j*nx+i)
    for k in range(nz):
        for i in range(nx):
            faces.append([pi(i,0,k),pi(i,0,k+1),pi(i+1,0,k+1),pi(i+1,0,k)]); owner.append(k*nx+i)
    for k in range(nz):
        for i in range(nx):
            faces.append([pi(i,ny,k),pi(i+1,ny,k),pi(i+1,ny,k+1),pi(i,ny,k+1)]); owner.append(k*ny*nx+(ny-1)*nx+i)
    for k in range(nz):
        for j in range(ny):
            faces.append([pi(0,j,k),pi(0,j+1,k),pi(0,j+1,k+1),pi(0,j,k+1)]); owner.append(k*ny*nx+j*nx)
    for k in range(nz):
        for j in range(ny):
            faces.append([pi(nx,j,k),pi(nx,j,k+1),pi(nx,j+1,k+1),pi(nx,j+1,k)]); owner.append(k*ny*nx+j*nx+(nx-1))
    bnd = [{"name":"bottom","type":"wall","startFace":ni,"nFaces":4},{"name":"top","type":"wall","startFace":ni+4,"nFaces":4},
           {"name":"front","type":"wall","startFace":ni+8,"nFaces":2},{"name":"back","type":"wall","startFace":ni+10,"nFaces":2},
           {"name":"left","type":"wall","startFace":ni+12,"nFaces":2},{"name":"right","type":"wall","startFace":ni+14,"nFaces":2}]
    return FvMesh(points=torch.tensor(pts, dtype=torch.float64),
                  faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in faces],
                  owner=torch.tensor(owner, dtype=INDEX_DTYPE),
                  neighbour=torch.tensor(nbr, dtype=INDEX_DTYPE), boundary=bnd)

class TestBasic:
    def test_returns_dict(self):
        r = split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]})
        assert isinstance(r, dict) and len(r) == 2
        for s in r.values(): assert isinstance(s, FvMesh)
    def test_names(self): assert set(split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]}).keys()) == {"l","r"}
    def test_cells(self):
        r = split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]})
        assert r["l"].n_cells == 2 and r["r"].n_cells == 2
    def test_total_cells(self): assert sum(s.n_cells for s in split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]}).values()) == 4
    def test_single(self):
        r = split_mesh_regions(_2x2(), {"all":[0,1,2,3]})
        assert len(r) == 1 and r["all"].n_cells == 4
    def test_overlap(self):
        with pytest.raises(ValueError): split_mesh_regions(_2x2(), {"a":[0,1],"b":[1,2]})

class TestTopology:
    def test_valid(self):
        for s in split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]}).values():
            assert s.owner.min().item() >= 0 and s.owner.max().item() < s.n_cells
    def test_internal(self):
        assert split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]})["l"].n_internal_faces >= 1
    def test_boundary(self):
        for s in split_mesh_regions(_2x2(), {"l":[0,2],"r":[1,3]}).values():
            assert len(s.boundary) >= 1

class TestVolume:
    def test_preserved(self):
        m = _2x2()
        r = split_mesh_regions(m, {"l":[0,2],"r":[1,3]})
        assert abs(sum(s.cell_volumes.sum().item() for s in r.values()) - m.total_volume.item()) < 1e-10
