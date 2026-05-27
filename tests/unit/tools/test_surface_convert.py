"""Tests for surface_convert."""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.surface_convert import surface_convert

def _tri():
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float64)
    n=np.array([[0,0,1]],dtype=np.float64); f=np.array([[0,1,2]],dtype=np.int32)
    return v,n,f

def _two():
    v=np.array([[0,0,0],[1,0,0],[.5,1,0],[.5,-1,0]],dtype=np.float64)
    n=np.array([[0,0,1],[0,0,1]],dtype=np.float64); f=np.array([[0,1,2],[0,3,1]],dtype=np.int32)
    return v,n,f

def _wstla(p,v,n,f):
    with open(p,"w") as fo:
        fo.write("solid m\n")
        for fi in range(f.shape[0]):
            nn=n[fi] if fi<n.shape[0] else [0,0,1]
            fo.write(f"  facet normal {nn[0]} {nn[1]} {nn[2]}\n    outer loop\n")
            for vi in range(3):
                pt=v[f[fi,vi]]; fo.write(f"      vertex {pt[0]} {pt[1]} {pt[2]}\n")
            fo.write("    endloop\n  endfacet\n")
        fo.write("endsolid m\n")

def _wstlb(p,v,n,f):
    with open(p,"wb") as fo:
        fo.write(b"\0"*80); fo.write(struct.pack("<I",f.shape[0]))
        for fi in range(f.shape[0]):
            nn=n[fi] if fi<n.shape[0] else [0.,0.,1.]
            fo.write(struct.pack("<fff",*nn))
            for vi in range(3): fo.write(struct.pack("<fff",*v[f[fi,vi]]))
            fo.write(struct.pack("<H",0))

def _wobj(p,v,n,f):
    with open(p,"w") as fo:
        for x in v: fo.write(f"v {x[0]} {x[1]} {x[2]}\n")
        for x in n: fo.write(f"vn {x[0]} {x[1]} {x[2]}\n")
        for fi in range(f.shape[0]): fo.write(f"f {f[fi,0]+1}//{fi+1} {f[fi,1]+1}//{fi+1} {f[fi,2]+1}//{fi+1}\n")

def _wvtk(p,v,n,f):
    with open(p,"w") as fo:
        fo.write("# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\n")
        fo.write(f"POINTS {v.shape[0]} double\n")
        for x in v: fo.write(f"{x[0]} {x[1]} {x[2]}\n")
        fo.write(f"POLYGONS {f.shape[0]} {f.shape[0]*4}\n")
        for fi in range(f.shape[0]): fo.write(f"3 {f[fi,0]} {f[fi,1]} {f[fi,2]}\n")
        fo.write("NORMALS face_normals double\n")
        for x in n: fo.write(f"{x[0]} {x[1]} {x[2]}\n")

class TestSurfaceConvert:
    def test_stl_to_obj(self,tmp_path):
        v,n,f=_tri(); ip=tmp_path/"i.stl"; op=tmp_path/"o.obj"
        _wstla(ip,v,n,f); r=surface_convert(ip,op)
        assert r.exists() and "v " in op.read_text()
    def test_stl_to_vtk(self,tmp_path):
        v,n,f=_tri(); ip=tmp_path/"i.stl"; op=tmp_path/"o.vtk"
        _wstla(ip,v,n,f); surface_convert(ip,op)
        c=op.read_text(); assert "DATASET POLYDATA" in c and "POLYGONS" in c
    def test_obj_to_stl(self,tmp_path):
        v,n,f=_tri(); ip=tmp_path/"i.obj"; op=tmp_path/"o.stl"
        _wobj(ip,v,n,f); surface_convert(ip,op)
        c=op.read_text(); assert "solid" in c and "facet normal" in c
    def test_vtk_to_obj(self,tmp_path):
        v,n,f=_tri(); ip=tmp_path/"i.vtk"; op=tmp_path/"o.obj"
        _wvtk(ip,v,n,f); surface_convert(ip,op)
        assert "v " in op.read_text()
    def test_format_from_extension(self,tmp_path):
        v,n,f=_tri(); ip=tmp_path/"i.stl"; op=tmp_path/"o.obj"
        _wstla(ip,v,n,f); surface_convert(ip,op)
        assert "v " in op.read_text()
    def test_nonexistent_raises(self,tmp_path):
        with pytest.raises(FileNotFoundError): surface_convert(str(tmp_path/"x.stl"),str(tmp_path/"o.obj"))
    def test_two_triangle(self,tmp_path):
        v,n,f=_two(); ip=tmp_path/"i.stl"; op=tmp_path/"o.obj"
        _wstla(ip,v,n,f); surface_convert(ip,op)
        assert len([l for l in op.read_text().splitlines() if l.startswith("v ")])==6
    def test_binary_stl(self,tmp_path):
        v,n,f=_tri(); ip=tmp_path/"i.stl"; op=tmp_path/"o.obj"
        _wstlb(ip,v,n,f); r=surface_convert(ip,op)
        assert r.exists() and "v " in op.read_text()
