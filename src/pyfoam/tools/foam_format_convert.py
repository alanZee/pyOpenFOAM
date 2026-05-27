"""foamFormatConvert - convert between ASCII and binary formats."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
__all__ = ["foam_format_convert"]
_SKIP = {"boundary","uniform","polyMesh"}
_TDR = re.compile(r"^\d+(?:\.\d+)?$")
_FC = {"volScalarField","volVectorField","volSymmTensorField","volTensorField","surfaceScalarField","surfaceVectorField","pointScalarField","pointVectorField"}
_MN = {"points","faces","owner","neighbour"}

def foam_format_convert(case_path, target_format="ascii", *, time_dirs=None):
    cd = Path(case_path).resolve()
    if not cd.is_dir(): raise FileNotFoundError(f"Case directory not found: {cd}")
    fmt = target_format.lower()
    if fmt not in ("ascii","binary"): raise ValueError(f"target_format must be 'ascii' or 'binary', got {target_format!r}")
    ta = fmt == "ascii"; conv = skip = 0; errs = []; fc = []
    for ff in _cff(cd, time_dirs):
        try:
            r = _cf(ff, ta)
            if r == "conv": conv += 1; fc.append(str(ff))
            else: skip += 1
        except Exception as e: errs.append((str(ff), str(e)))
    return {"converted":conv,"skipped":skip,"errors":errs,"files":fc}

def _cff(cd, td):
    fs = []
    for e in sorted(cd.iterdir()):
        if not e.is_dir() or not _TDR.match(e.name): continue
        if td is not None:
            try: tv = float(e.name)
            except ValueError: continue
            if not any(abs(tv-float(t))<1e-12 for t in td): continue
        for f in _iff(e): fs.append(f)
    pm = cd/"constant"/"polyMesh"
    if pm.is_dir():
        for f in _iff(pm): fs.append(f)
    return fs

def _iff(d):
    for f in sorted(d.iterdir()):
        if not f.is_file() or f.name in _SKIP: continue
        try:
            t = f.read_text(encoding="latin-1")
            if "FoamFile" in t and "{" in t: yield f
        except: continue

def _cf(p, ta):
    t = p.read_text(encoding="latin-1")
    hm = re.search(r"FoamFile\s*\{([^}]*)\}", t, re.DOTALL)
    if hm is None: return "skip"
    hb = hm.group(1)
    fm = re.search(r"format\s+(ascii|binary)", hb, re.I)
    if fm is None: return "skip"
    if (fm.group(1).lower()=="ascii") == ta: return "skip"
    cm = re.search(r"class\s+(\w+)", hb)
    cn = cm.group(1) if cm else ""
    om = re.search(r"object\s+(\w+)", hb)
    on = om.group(1) if om else p.name
    _cfg(p, t, hm, ta)
    return "conv"

def _cff2(p, ta):
    from pyfoam.io.field_io import read_field, write_field
    from pyfoam.io.foam_file import FileFormat
    fd = read_field(p)
    fd.header.format = FileFormat.ASCII if ta else FileFormat.BINARY
    write_field(p, fd, overwrite=True)

def _cfm(p, on, ta):
    from pyfoam.io.foam_file import FileFormat
    from pyfoam.io.mesh_io import read_points,read_faces,read_owner,read_neighbour,write_points,write_faces,write_owner,write_neighbour
    rd = {"points":(read_points,write_points),"faces":(read_faces,write_faces),"owner":(read_owner,write_owner),"neighbour":(read_neighbour,write_neighbour)}
    if on not in rd: return
    r,w = rd[on]; h,d = r(p); h.format = FileFormat.ASCII if ta else FileFormat.BINARY; w(p,h,d,overwrite=True)

def _cfg(p, t, hm, ta):
    nt = _b2a(t,hm) if ta else _a2b(t,hm)
    p.write_text(nt, encoding="latin-1")

def _b2a(t, hm):
    import numpy as np
    hs,he = hm.start(),hm.end()
    nh = t[hs:he].replace("format      binary","format      ascii")
    body = t[he:]; parts = []; i = 0
    while i < len(body):
        m = re.match(r"(\d+)\s*\(", body[i:])
        if m is None: parts.append(body[i]); i+=1; continue
        c = int(m.group(1)); pp = i+m.end()-1; cs = pp+1
        # Determine component multiplier from type tag context
        ctx = body[max(0,i-120):i]
        mult = 3 if ("vector" in ctx or "Vector" in ctx) else 6 if "symmTensor" in ctx else 9 if "tensor" in ctx else 1
        cc = cs + c * mult * 8
        if cc < len(body) and body[cc] == ")":
            try:
                a = np.frombuffer(body[cs:cc].encode("latin-1"), dtype=">f8")
                if len(a)==c*mult:
                    if mult > 1:
                        lines = []
                        for j in range(0, len(a), mult):
                            vals = " ".join(f"{a[j+k]}" for k in range(mult))
                            lines.append(f"({vals})")
                        parts.append(body[i:pp+1]); parts.append("\n"+"\n".join(lines)+"\n)"); i=cc+1; continue
                    else:
                        parts.append(body[i:pp+1]); parts.append("\n"+"\n".join(f"{v}" for v in a)+"\n)"); i=cc+1; continue
            except: pass
        cp = _fcp(body,pp)
        if cp==-1: parts.append(body[i]); i+=1
        else: parts.append(body[i:cp+1]); i=cp+1
    return t[:hs]+nh+"".join(parts)

_NUR = re.compile(r"nonuniform\s+(?:(\w+)<(\w+)>\s+)?(\d+)")

def _a2b(t, hm):
    import numpy as np
    hs,he = hm.start(),hm.end()
    nh = t[hs:he].replace("format      ascii","format      binary")
    body = t[he:]; parts = []; i = 0
    while i < len(body):
        m = _NUR.search(body, i)
        if m is None: parts.append(body[i:]); break
        parts.append(body[i:m.start()]); c = int(m.group(3))
        pp = body.find("(",m.end())
        if pp==-1: parts.append(body[m.start():m.end()]); i=m.end(); continue
        cp = _fcp(body,pp)
        if cp==-1: parts.append(body[m.start():m.end()]); i=m.end(); continue
        v = _pad(body[pp+1:cp].strip(), c)
        if v is not None:
            a = np.array(v, dtype=">f8")
            parts.append(body[m.start():pp+1])
            parts.append(a.tobytes().decode("latin-1")); parts.append(")"); i=cp+1
        else: parts.append(body[m.start():cp+1]); i=cp+1
    return t[:hs]+nh+"".join(parts)

def _fcp(t,o):
    d=0
    for i in range(o,len(t)):
        if t[i]=="(": d+=1
        elif t[i]==")":
            d-=1
            if d==0: return i
    return -1

def _pad(dt,n):
    dt=dt.strip()
    if not dt: return None
    vm=re.findall(r"\(\s*([^)]+)\)",dt)
    if len(vm)==n:
        f=[]
        for v in vm: f.extend(float(x) for x in v.split())
        return f
    tk=dt.split()
    if len(tk)==n:
        try: return [float(t) for t in tk]
        except: return None
    return None
