"""patchSummary - per-patch field statistics."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
__all__ = ["patch_summary"]
_FC = {"volScalarField","volVectorField","volSymmTensorField","volTensorField","surfaceScalarField","surfaceVectorField","pointScalarField","pointVectorField"}
_TDR = re.compile(r"^\d+(?:\.\d+)?$")

def patch_summary(case_path, time=None):
    cd = Path(case_path).resolve()
    if not cd.is_dir(): raise FileNotFoundError(f"Case directory not found: {cd}")
    td = _rtd(cd, time)
    if not td.is_dir(): raise FileNotFoundError(f"Time directory not found: {td}")
    bp = cd/"constant"/"polyMesh"/"boundary"
    patches = _rbp(bp); result = {}
    for ff in _iff(td):
        try:
            fs = _cfs(ff, patches)
            for pn, stats in fs.items():
                if pn not in result: result[pn] = {}
                result[pn][ff.name] = stats
        except: continue
    return result

def _rtd(cd, time):
    if time is None: return cd/"0"
    if isinstance(time,str) and time=="latestTime":
        ts=[]
        for e in cd.iterdir():
            if e.is_dir() and _TDR.match(e.name):
                try: ts.append((float(e.name),e))
                except: continue
        if not ts: raise FileNotFoundError("No time directories found")
        return max(ts,key=lambda x:x[0])[1]
    ts2 = str(time) if isinstance(time,(int,float)) else time
    td = cd/ts2
    if td.is_dir(): return td
    try:
        tv=float(time)
        for e in cd.iterdir():
            if e.is_dir() and _TDR.match(e.name):
                try:
                    if abs(float(e.name)-tv)<1e-12: return e
                except: continue
    except: pass
    raise FileNotFoundError(f"Time directory '{time}' not found in {cd}")

class _PI:
    __slots__=("name","n_faces","start_face")
    def __init__(s,n,nf,sf): s.name=n;s.n_faces=nf;s.start_face=sf

def _rbp(bp):
    patches={}
    if not bp.is_file(): return patches
    t=bp.read_text(encoding="utf-8",errors="replace")
    for bm in re.finditer(r"(\w+)\s*\{([^}]*)\}",t,re.DOTALL):
        name,block=bm.group(1),bm.group(2); nf=sf=0
        for kv in re.finditer(r"(\w+)\s+(.+?)\s*;",block):
            k,v=kv.group(1),kv.group(2).strip()
            if k=="nFaces": nf=int(v)
            elif k=="startFace": sf=int(v)
        if nf>0: patches[name]=_PI(name,nf,sf)
    return patches

def _iff(td):
    for fe in sorted(td.iterdir()):
        if not fe.is_file() or fe.name.startswith("."): continue
        try:
            t=fe.read_text(encoding="utf-8",errors="replace")
            if "FoamFile" not in t: continue
            cm=re.search(r"class\s+(\w+)",t)
            if cm and cm.group(1) in _FC: yield fe
        except: continue

def _cfs(ff, patches):
    t=ff.read_text(encoding="utf-8",errors="replace")
    cm=re.search(r"class\s+(\w+)",t)
    cn=cm.group(1) if cm else "volScalarField"
    iv="Vector" in cn or "vector" in cn
    bm=re.search(r"boundaryField\s*\{",t)
    if bm is None: return {}
    bs=bm.end()-1; be=_fmb(t,bs); bf=t[bs+1:be]; result={}
    for pm in re.finditer(r"(\w+)\s*\{([^}]*)\}",bf,re.DOTALL):
        stats=_eps(pm.group(2),iv)
        if stats is not None: result[pm.group(1)]=stats
    return result

def _eps(pb,iv):
    vm=re.search(r"value\s+(.+?)(?:;|$)",pb,re.DOTALL)
    if vm is None: return None
    vs=vm.group(1).strip()
    if vs.startswith("uniform"):
        vt=vs[7:].strip()
        if iv:
            vt=vt.strip("()"); pts=vt.split()
            if len(pts)>=3:
                try: mag=float(np.linalg.norm([float(p) for p in pts[:3]])); return {"min":mag,"max":mag,"average":mag}
                except: return None
            return None
        try: val=float(vt); return {"min":val,"max":val,"average":val}
        except: return None
    if vs.startswith("nonuniform"):
        ps=vs.find("(")
        if ps==-1: return None
        dt=vs[ps:]
        if iv:
            vecs=re.findall(r"\(\s*([^)]+)\)",dt)
            if not vecs: return None
            mags=[]
            for v in vecs:
                pts=v.split()
                if len(pts)>=3:
                    try: mags.append(float(np.linalg.norm([float(p) for p in pts[:3]])))
                    except: continue
            if not mags: return None
            a=np.array(mags)
        else:
            inner=dt.strip("() \n"); tk=inner.split()
            if not tk: return None
            try: a=np.array([float(t) for t in tk])
            except: return None
        return {"min":float(a.min()),"max":float(a.max()),"average":float(a.mean())}
    return None

def _fmb(t,s):
    d=0
    for i in range(s,len(t)):
        if t[i]=="{": d+=1
        elif t[i]=="}":
            d-=1
            if d==0: return i
    raise ValueError("Unmatched brace")
