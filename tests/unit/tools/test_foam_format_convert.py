"""Tests for foam_format_convert."""
from __future__ import annotations
import re
from pathlib import Path
import pytest
from pyfoam.tools.foam_format_convert import foam_format_convert

def _waf(p,cn,on,n=5):
    hdr=f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cn};\n    object      {on};\n}}\n"
    if "Scalar" in cn:
        vals="\n".join(f"{float(i)}" for i in range(n))
        body=f"\ndimensions      [0 2 -2 0 0 0 0];\n\ninternalField   nonuniform List<scalar> {n}\n(\n{vals}\n);\n\nboundaryField\n{{\n    inlet\n    {{\n        type        fixedValue;\n        value       uniform 0;\n    }}\n}}\n"
    else:
        vecs="\n".join(f"({i} {i+1} {i+2})" for i in range(n))
        body=f"\ndimensions      [0 1 -1 0 0 0 0];\n\ninternalField   nonuniform List<vector> {n}\n(\n{vecs}\n);\n\nboundaryField\n{{\n    inlet\n    {{\n        type        fixedValue;\n        value       uniform (0 0 0);\n    }}\n}}\n"
    p.write_text(hdr+body)

def _wap(p,n=4):
    hdr="FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       vectorField;\n    object      points;\n}\n"
    pts="\n".join(f"({i} {i*2} {i*3})" for i in range(n))
    p.write_text(hdr+f"\n{n}\n(\n{pts}\n)\n")

def _sc(td):
    d=td/"tc"; d.mkdir(); t0=d/"0"; t0.mkdir()
    _waf(t0/"p","volScalarField","p"); _waf(t0/"U","volVectorField","U")
    pm=d/"constant"/"polyMesh"; pm.mkdir(parents=True); _wap(pm/"points"); return d

class TestFoamFormatConvert:
    def test_ascii_to_binary(self,tmp_path):
        d=_sc(tmp_path); r=foam_format_convert(d,target_format="binary")
        assert r["converted"]==3 and r["skipped"]==0 and len(r["errors"])==0
        assert "format      binary" in (d/"0"/"p").read_text(encoding="latin-1")
    def test_binary_to_ascii(self,tmp_path):
        d=_sc(tmp_path); foam_format_convert(d,target_format="binary")
        r=foam_format_convert(d,target_format="ascii")
        assert r["converted"]==3; assert "format      ascii" in (d/"0"/"p").read_text(encoding="latin-1")
    def test_skip(self,tmp_path):
        r=foam_format_convert(_sc(tmp_path),target_format="ascii")
        assert r["converted"]==0 and r["skipped"]==3
    def test_nonexistent(self,tmp_path):
        with pytest.raises(FileNotFoundError): foam_format_convert(str(tmp_path/"x"))
    def test_invalid(self,tmp_path):
        with pytest.raises(ValueError,match="target_format"): foam_format_convert(_sc(tmp_path),target_format="xml")
    def test_time_dirs(self,tmp_path):
        d=_sc(tmp_path); t1=d/"1"; t1.mkdir(); _waf(t1/"p","volScalarField","p",3)
        r=foam_format_convert(d,target_format="binary",time_dirs=[0])
        assert r["converted"]==3; assert "format      ascii" in (t1/"p").read_text(encoding="latin-1")
    def test_scalar_values(self,tmp_path):
        d=_sc(tmp_path); foam_format_convert(d,target_format="binary"); foam_format_convert(d,target_format="ascii")
        c=(d/"0"/"p").read_text(encoding="latin-1"); m=re.search(r"nonuniform List<scalar> \d+\n\((.*?)\)",c,re.DOTALL)
        assert m is not None; vals=[float(v) for v in m.group(1).strip().split()]; assert vals==[0.,1.,2.,3.,4.]
    def test_vector_values(self,tmp_path):
        d=_sc(tmp_path); foam_format_convert(d,target_format="binary"); foam_format_convert(d,target_format="ascii")
        c=(d/"0"/"U").read_text(encoding="latin-1"); m=re.search(r"nonuniform List<vector> \d+\n\((.+)\)",c,re.DOTALL)
        assert m is not None
        vecs=re.findall(r"\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)",m.group(1))
        assert len(vecs)==5
        assert abs(float(vecs[0][0]))<1e-6 and abs(float(vecs[0][1])-1.0)<1e-6 and abs(float(vecs[0][2])-2.0)<1e-6
    def test_empty(self,tmp_path):
        d=tmp_path/"e"; d.mkdir(); (d/"0").mkdir(); (d/"constant"/"polyMesh").mkdir(parents=True)
        r=foam_format_convert(d,target_format="binary"); assert r["converted"]==0 and r["skipped"]==0
