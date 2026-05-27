"""Tests for patch_summary."""
from __future__ import annotations
from pathlib import Path
import pytest
from pyfoam.tools.patch_summary import patch_summary

def _wbp(pm,patches):
    hdr="FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       polyBoundaryMesh;\n    location    \"constant/polyMesh\";\n    object      boundary;\n}\n"
    body=f"\n{len(patches)}\n(\n"
    for p in patches: body+=f"    {p['name']}\n    {{\n        type        {p.get('type','patch')};\n        nFaces      {p['nFaces']};\n        startFace   {p['startFace']};\n    }}\n"
    body+=")\n"; (pm/"boundary").write_text(hdr+body)

def _wsf(p,name,cn,pv):
    hdr=f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cn};\n    object      {name};\n}}\n"
    body="\ndimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 1.0;\n\nboundaryField\n{\n"
    for pn,val in pv.items(): body+=f"    {pn}\n    {{\n        type        fixedValue;\n        value       uniform {val};\n    }}\n"
    body+="}\n"; p.write_text(hdr+body)

def _wnuf(p,name,pv):
    hdr=f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n    object      {name};\n}}\n"
    body="\ndimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 1.0;\n\nboundaryField\n{\n"
    for pn,vals in pv.items():
        n=len(vals); vl="\n".join(f"{v}" for v in vals)
        body+=f"    {pn}\n    {{\n        type        fixedValue;\n        value       nonuniform List<scalar> {n}\n        (\n{vl}\n        );\n    }}\n"
    body+="}\n"; p.write_text(hdr+body)

def _sc(td):
    d=td/"tc"; d.mkdir(); pm=d/"constant"/"polyMesh"; pm.mkdir(parents=True)
    _wbp(pm,[{"name":"inlet","nFaces":4,"startFace":10},{"name":"outlet","nFaces":4,"startFace":14},{"name":"walls","nFaces":8,"startFace":18}])
    t0=d/"0"; t0.mkdir(); _wsf(t0/"p","p","volScalarField",{"inlet":100.0,"outlet":50.0,"walls":0.0}); return d

class TestPatchSummary:
    def test_basic(self,tmp_path):
        r=patch_summary(_sc(tmp_path),time=0); assert "inlet" in r and "outlet" in r and "walls" in r
    def test_uniform_scalar(self,tmp_path):
        r=patch_summary(_sc(tmp_path),time=0); assert r["inlet"]["p"]=={"min":100.0,"max":100.0,"average":100.0}
    def test_multiple_patches(self,tmp_path):
        r=patch_summary(_sc(tmp_path),time=0)
        assert r["inlet"]["p"]["min"]==100.0 and r["outlet"]["p"]["min"]==50.0 and r["walls"]["p"]["min"]==0.0
    def test_nonuniform(self,tmp_path):
        d=_sc(tmp_path); _wnuf(d/"0"/"T","T",{"inlet":[10.,20.,30.],"outlet":[5.,15.]})
        r=patch_summary(d,time=0)
        assert r["inlet"]["T"]=={"min":10.0,"max":30.0,"average":20.0}
        assert r["outlet"]["T"]=={"min":5.0,"max":15.0,"average":10.0}
    def test_nonexistent_case(self,tmp_path):
        with pytest.raises(FileNotFoundError): patch_summary(str(tmp_path/"x"))
    def test_nonexistent_time(self,tmp_path):
        with pytest.raises(FileNotFoundError): patch_summary(_sc(tmp_path),time=999)
    def test_default_time(self,tmp_path):
        r=patch_summary(_sc(tmp_path)); assert "inlet" in r and "p" in r["inlet"]
    def test_latest_time(self,tmp_path):
        d=_sc(tmp_path); t1=d/"1"; t1.mkdir()
        _wsf(t1/"p","p","volScalarField",{"inlet":200.0,"outlet":150.0})
        r=patch_summary(d,time="latestTime"); assert r["inlet"]["p"]["min"]==200.0
