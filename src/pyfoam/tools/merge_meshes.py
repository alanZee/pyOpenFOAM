"""mergeMeshes -- merge multiple meshes."""
from __future__ import annotations
import torch
from pyfoam.core.dtype import INDEX_DTYPE
__all__ = ["merge_meshes"]

def merge_meshes(meshes, tolerance=1e-8):
    from pyfoam.mesh.fv_mesh import FvMesh
    if not meshes: raise ValueError("meshes list is empty")
    if len(meshes) == 1:
        return FvMesh(points=meshes[0].points.clone(), faces=[f.clone() for f in meshes[0].faces],
                       owner=meshes[0].owner.clone(), neighbour=meshes[0].neighbour.clone(),
                       boundary=[dict(b) for b in meshes[0].boundary], validate=False)
    dev, dt = meshes[0].device, meshes[0].dtype
    mp, prm = _merge_pts([m.points for m in meshes], tolerance, dev, dt)
    af = []; ao = []; an = []; ab = []; co = 0; fo = 0
    for mi, m in enumerate(meshes):
        r = prm[mi]
        for fi in range(m.n_faces): af.append(torch.tensor([r[p] for p in m.faces[fi].tolist()], dtype=INDEX_DTYPE, device=dev))
        ao.extend(o + co for o in m.owner.tolist()); an.extend(n + co for n in m.neighbour.tolist())
        for p in m.boundary: ab.append({"name": p["name"], "type": p["type"], "startFace": p["startFace"] + fo, "nFaces": p["nFaces"]})
        co += m.n_cells; fo += m.n_faces
    af, ao, an, ab = _conv_shared(af, ao, an, ab)
    return FvMesh(points=mp, faces=af, owner=torch.tensor(ao, dtype=INDEX_DTYPE, device=dev),
                   neighbour=torch.tensor(an, dtype=INDEX_DTYPE, device=dev), boundary=ab, validate=False)

def _merge_pts(psets, tol, dev, dt):
    merged = []; rems = []
    for pts in psets:
        rem = {}
        for i in range(pts.shape[0]):
            c = pts[i]; found = False
            for j, e in enumerate(merged):
                if (e - c).norm().item() < tol: rem[i] = j; found = True; break
            if not found: rem[i] = len(merged); merged.append(c.unsqueeze(0))
        rems.append(rem)
    mt = torch.cat(merged, dim=0) if merged else torch.empty((0, 3), dtype=dt, device=dev)
    return mt, rems

def _conv_shared(faces, owner, neighbour, boundary):
    ni = len(neighbour); bi = list(range(ni, len(faces)))
    fk = {}
    for fi in bi: fk.setdefault(tuple(sorted(faces[fi].tolist())), []).append(fi)
    rm = set(); ni_new = []
    for k, fis in fk.items():
        if len(fis) < 2: continue
        f1, f2 = fis[0], fis[1]; o1, o2 = owner[f1], owner[f2]
        ni_new.append((faces[f1], min(o1, o2), max(o1, o2))); rm.update([f1, f2])
    if not rm: return faces, owner, neighbour, boundary
    nf = []; no = []; nn = []
    for fi in range(ni): nf.append(faces[fi]); no.append(owner[fi]); nn.append(neighbour[fi])
    for fp, o, n in ni_new: nf.append(fp); no.append(o); nn.append(n)
    nni = len(nn)
    kb = [(faces[fi], owner[fi]) for fi in bi if fi not in rm]
    for fp, o in kb: nf.append(fp); no.append(o)
    nb = []
    if kb: nb = [{"name": boundary[0]["name"] if boundary else "merged", "type": boundary[0]["type"] if boundary else "wall", "startFace": nni, "nFaces": len(kb)}]
    return nf, no, nn, nb
