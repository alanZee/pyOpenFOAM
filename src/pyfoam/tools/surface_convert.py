"""surfaceConvert - convert surface mesh files between STL/OBJ/VTK."""
from __future__ import annotations
import re, struct
from pathlib import Path
from typing import Optional, Union
import numpy as np
__all__ = ["surface_convert"]
_FMTS = {"stl","obj","vtk"}
_VTKH = {"POINTS","POLYGONS","CELLS","NORMALS","SCALARS","VECTORS","POINT_DATA","CELL_DATA","DATASET"}

def surface_convert(input_path, output_path, output_format=None):
    ip=Path(input_path).resolve(); op=Path(output_path).resolve()
    if not ip.is_file(): raise FileNotFoundError(f"Input file not found: {ip}")
    inf=_df(ip); outf=output_format.lower() if output_format else _df(op)
    if outf not in _FMTS: raise ValueError(f"Unsupported format: {outf!r}. Supported: {sorted(_FMTS)}")
    v,n,f=_rs(ip,inf); op.parent.mkdir(parents=True,exist_ok=True); _ws(op,outf,v,n,f); return op

def _df(p):
    ext=p.suffix.lower(); m={".stl":"stl",".obj":"obj",".vtk":"vtk"}
    if ext in m: return m[ext]
    raise ValueError(f"Cannot determine format from '{ext}'")

def _rs(p,fmt):
    if fmt=="stl": return _rstl(p)
    if fmt=="obj": return _robj(p)
    if fmt=="vtk": return _rvtk(p)
    raise ValueError(f"Unsupported format: {fmt!r}")

def _rstl(p):
    with open(p,"rb") as f: hdr=f.read(80)
    if hdr.lstrip().startswith(b"solid"):
        t=p.read_text(encoding="utf-8",errors="replace")
        if "facet" in t and "normal" in t: return _rstla(t)
    return _rstlb(p)

def _rstla(t):
    vl,nl,fl=[],[],[]
    fp=re.compile(r"facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+outer\s+loop(.*?)endfacet",re.DOTALL)
    vp=re.compile(r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
    for m in fp.finditer(t):
        nl.append([float(m.group(i)) for i in (1,2,3)])
        verts=vp.findall(m.group(4))
        if len(verts)!=3: continue
        fi=[]
        for vx,vy,vz in verts:
            idx=len(vl); vl.append([float(vx),float(vy),float(vz)]); fi.append(idx)
        fl.append(fi)
    v=np.array(vl,dtype=np.float64) if vl else np.empty((0,3),dtype=np.float64)
    n=np.array(nl,dtype=np.float64) if nl else np.empty((0,3),dtype=np.float64)
    f=np.array(fl,dtype=np.int32) if fl else np.empty((0,3),dtype=np.int32)
    return v,n,f

def _rstlb(p):
    with open(p,"rb") as f:
        f.read(80); nt=struct.unpack("<I",f.read(4))[0]; vl,nl,fl=[],[],[]
        for _ in range(nt):
            nl.append(list(struct.unpack("<fff",f.read(12)))); fi=[]
            for _ in range(3):
                idx=len(vl); vl.append(list(struct.unpack("<fff",f.read(12)))); fi.append(idx)
            fl.append(fi); f.read(2)
    v=np.array(vl,dtype=np.float64) if vl else np.empty((0,3),dtype=np.float64)
    n=np.array(nl,dtype=np.float64) if nl else np.empty((0,3),dtype=np.float64)
    f=np.array(fl,dtype=np.int32) if fl else np.empty((0,3),dtype=np.int32)
    return v,n,f

def _robj(p):
    t=p.read_text(encoding="utf-8",errors="replace"); vl,nl,fl=[],[],[]
    for line in t.splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        pts=line.split()
        if not pts: continue
        if pts[0]=="v" and len(pts)>=4: vl.append([float(pts[i]) for i in (1,2,3)])
        elif pts[0]=="vn" and len(pts)>=4: nl.append([float(pts[i]) for i in (1,2,3)])
        elif pts[0]=="f":
            fi=[]
            for p2 in pts[1:]:
                idx=int(p2.split("/")[0]); idx=idx-1 if idx>0 else len(vl)+idx; fi.append(idx)
            if len(fi)>=3:
                for i in range(1,len(fi)-1): fl.append([fi[0],fi[i],fi[i+1]])
    v=np.array(vl,dtype=np.float64) if vl else np.empty((0,3),dtype=np.float64)
    n=np.array(nl,dtype=np.float64) if nl else np.empty((0,3),dtype=np.float64)
    f=np.array(fl,dtype=np.int32) if fl else np.empty((0,3),dtype=np.int32)
    return v,n,f

def _rvtk(p):
    t=p.read_text(encoding="utf-8",errors="replace"); lines=t.splitlines(); vl,fl,nl=[],[],[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if line.startswith("POINTS"):
            np2=int(line.split()[1]); i+=1; coords=[]
            while len(coords)<np2*3 and i<len(lines):
                coords.extend(float(x) for x in lines[i].strip().split()); i+=1
            for j in range(np2): vl.append([coords[j*3],coords[j*3+1],coords[j*3+2]])
            continue
        elif line.startswith("POLYGONS") or line.startswith("CELLS"):
            nc=int(line.split()[1]); i+=1
            for _ in range(nc):
                if i>=len(lines): break
                tk=lines[i].strip().split(); nv=int(tk[0]); ix=[int(tk[j+1]) for j in range(nv)]
                for k in range(1,len(ix)-1): fl.append([ix[0],ix[k],ix[k+1]])
                i+=1
            continue
        elif line.startswith("NORMALS"):
            i+=1
            while i<len(lines):
                nl2=lines[i].strip()
                if not nl2 or nl2.startswith("#") or (nl2.split() and nl2.split()[0] in _VTKH): break
                tk=nl2.split()
                for j in range(0,len(tk)-2,3): nl.append([float(tk[j]),float(tk[j+1]),float(tk[j+2])])
                i+=1
            continue
        i+=1
    v=np.array(vl,dtype=np.float64) if vl else np.empty((0,3),dtype=np.float64)
    n=np.array(nl,dtype=np.float64) if nl else np.empty((0,3),dtype=np.float64)
    f=np.array(fl,dtype=np.int32) if fl else np.empty((0,3),dtype=np.int32)
    return v,n,f

def _ws(p,fmt,v,n,f):
    if fmt=="stl": _wstl(p,v,n,f)
    elif fmt=="obj": _wobj(p,v,n,f)
    elif fmt=="vtk": _wvtk(p,v,n,f)

def _wstl(p,v,n,f):
    nf=f.shape[0]
    with open(p,"w") as fo:
        fo.write("solid surface\n")
        for fi in range(nf):
            if fi<n.shape[0]: nx,ny,nz=n[fi]
            else:
                e1=v[f[fi,1]]-v[f[fi,0]]; e2=v[f[fi,2]]-v[f[fi,0]]; nv2=np.cross(e1,e2); nm=np.linalg.norm(nv2)
                if nm>0: nv2/=nm
                nx,ny,nz=nv2
            fo.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n    outer loop\n")
            for vi in range(3):
                pt=v[f[fi,vi]]; fo.write(f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
            fo.write("    endloop\n  endfacet\n")
        fo.write("endsolid surface\n")

def _wobj(p,v,n,f):
    with open(p,"w") as fo:
        fo.write("# OBJ file generated by pyOpenFOAM\n")
        for i in range(v.shape[0]): fo.write(f"v {v[i,0]:.10e} {v[i,1]:.10e} {v[i,2]:.10e}\n")
        for i in range(n.shape[0]): fo.write(f"vn {n[i,0]:.6e} {n[i,1]:.6e} {n[i,2]:.6e}\n")
        if n.shape[0]>0:
            for fi in range(f.shape[0]): fo.write(f"f {f[fi,0]+1}//{fi+1} {f[fi,1]+1}//{fi+1} {f[fi,2]+1}//{fi+1}\n")
        else:
            for fi in range(f.shape[0]): fo.write(f"f {f[fi,0]+1} {f[fi,1]+1} {f[fi,2]+1}\n")

def _wvtk(p,v,n,f):
    np2,nf=v.shape[0],f.shape[0]
    with open(p,"w") as fo:
        fo.write(f"# vtk DataFile Version 3.0\npyOpenFOAM surface\nASCII\nDATASET POLYDATA\n")
        fo.write(f"POINTS {np2} double\n")
        for i in range(np2): fo.write(f"{v[i,0]:.10e} {v[i,1]:.10e} {v[i,2]:.10e}\n")
        fo.write(f"POLYGONS {nf} {nf*4}\n")
        for fi in range(nf): fo.write(f"3 {f[fi,0]} {f[fi,1]} {f[fi,2]}\n")
        if n.shape[0]>0:
            fo.write("NORMALS face_normals double\n")
            for i in range(n.shape[0]): fo.write(f"{n[i,0]:.10e} {n[i,1]:.10e} {n[i,2]:.10e}\n")
