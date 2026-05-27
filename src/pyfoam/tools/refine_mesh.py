"""
refineMesh -- refine cells by splitting hex cells into sub-cells.
Returns a **new** FvMesh; the input mesh is not modified.
"""
from __future__ import annotations
import torch
from pyfoam.core.dtype import INDEX_DTYPE
__all__ = ["refine_mesh", "split_face", "_PointManager"]

def refine_mesh(mesh, cells, direction="all"):
    from pyfoam.mesh.fv_mesh import FvMesh
    rx, ry, rz = _parse_direction(direction)
    if isinstance(cells, torch.Tensor): cells = cells.tolist()
    refined_set = set(int(c) for c in cells)
    if not refined_set:
        return FvMesh(points=mesh.points.clone(), faces=[f.clone() for f in mesh.faces],
            owner=mesh.owner.clone(), neighbour=mesh.neighbour.clone(),
            boundary=[dict(b) for b in mesh.boundary], validate=False)
    n_sub = (2 if rx else 1) * (2 if ry else 1) * (2 if rz else 1)
    sorted_refined = sorted(refined_set)
    sorted_unrefined = sorted(set(range(mesh.n_cells)) - refined_set)
    new_idx = 0; cell_base = {}
    for c in sorted_refined: cell_base[c] = new_idx; new_idx += n_sub
    for c in sorted_unrefined: cell_base[c] = new_idx; new_idx += 1
    pm = _PointManager(mesh.points); rfaces = {}; ufaces = []
    for fi in range(mesh.n_faces):
        own = int(mesh.owner[fi].item())
        nbr = int(mesh.neighbour[fi].item()) if fi < mesh.n_internal_faces else -1
        own_r = own in refined_set; nbr_r = nbr >= 0 and nbr in refined_set
        if not own_r and not nbr_r:
            ufaces.append((mesh.faces[fi].clone(), cell_base[own], cell_base.get(nbr, -1))); continue
        _process_face(fi, mesh, refined_set, rx, ry, rz, pm, rfaces, cell_base, n_sub)
    for cell_idx in sorted_refined:
        _generate_midplanes(cell_idx, mesh, rx, ry, rz, pm, rfaces, cell_base, n_sub)
    return _assemble(mesh, pm, ufaces, rfaces)

def _parse_direction(d):
    m = {"all": (True, True, True), "x": (True, False, False), "y": (False, True, False), "z": (False, False, True)}
    if d not in m: raise ValueError(f"Invalid direction")
    return m[d]

class _PointManager:
    def __init__(self, orig):
        self._orig = orig; self._extra = []; self._cache = {}; self._n = orig.shape[0]
    def add(self, coord):
        key = tuple(round(float(c), 10) for c in coord.tolist())
        if key in self._cache: return self._cache[key]
        idx = self._n + sum(t.shape[0] for t in self._extra)
        self._extra.append(coord.reshape(1, -1)); self._cache[key] = idx; return idx
    def pts(self):
        if not self._extra: return self._orig.clone()
        return torch.cat([self._orig] + self._extra, dim=0)
    all_points = pts

def _i2b(idx, rx, ry, rz):
    bits = [0, 0, 0]; rem = idx
    for do, ci in [(rx, 0), (ry, 1), (rz, 2)]:
        if do: bits[ci] = rem & 1; rem >>= 1
    return bits

def _b2i(bits, rx, ry, rz):
    idx = 0; bi = 0
    for do, ci in [(rx, 0), (ry, 1), (rz, 2)]:
        if do:
            if bits[ci]: idx |= (1 << bi)
            bi += 1
    return idx

def _cell_corners(ci, mesh):
    s = set()
    for fi in range(mesh.n_faces):
        if int(mesh.owner[fi].item()) == ci: s.update(mesh.faces[fi].tolist())
        elif fi < mesh.n_internal_faces and int(mesh.neighbour[fi].item()) == ci: s.update(mesh.faces[fi].tolist())
    return sorted(s)

def _process_face(fi, mesh, refined_set, rx, ry, rz, pm, rfaces, cell_base, n_sub):
    fp = mesh.faces[fi]
    own = int(mesh.owner[fi].item())
    nbr = int(mesh.neighbour[fi].item()) if fi < mesh.n_internal_faces else -1
    own_r = own in refined_set; nbr_r = nbr >= 0 and nbr in refined_set
    pts = mesh.points[fp]; sf_list = split_face(pts, rx, ry, rz, fp, pm)
    if own_r and nbr_r:
        owner_indices = _sub_cell_idx(mesh, own, sf_list, pm, rx, ry, rz, cell_base)
        ap = pm.pts(); ns = set()
        for fi2 in range(mesh.n_faces):
            if int(mesh.owner[fi2].item()) == nbr: ns.update(mesh.faces[fi2].tolist())
            elif fi2 < mesh.n_internal_faces and int(mesh.neighbour[fi2].item()) == nbr: ns.update(mesh.faces[fi2].tolist())
        ncpt = mesh.points[list(ns)]; nmid = 0.5 * (ncpt.min(dim=0).values + ncpt.max(dim=0).values)
        for i, sf in enumerate(sf_list):
            k = tuple(sorted(sf.tolist())); centre = ap[sf].mean(dim=0)
            nbr_bits = [0, 0, 0]
            for do, c in [(rx, 0), (ry, 1), (rz, 2)]:
                if do and centre[c] > nmid[c]: nbr_bits[c] = 1
            rfaces[k] = (sf, owner_indices[i], cell_base[nbr] + _b2i(nbr_bits, rx, ry, rz))
    elif own_r:
        ci_list = _sub_cell_idx(mesh, own, sf_list, pm, rx, ry, rz, cell_base)
        for i, sf in enumerate(sf_list):
            k = tuple(sorted(sf.tolist()))
            rfaces[k] = (sf, ci_list[i], cell_base[nbr] if nbr >= 0 else -1)
    else:
        ci_list = _sub_cell_idx(mesh, nbr, sf_list, pm, rx, ry, rz, cell_base)
        for i, sf in enumerate(sf_list):
            k = tuple(sorted(sf.tolist()))
            rfaces[k] = (sf, cell_base[own], ci_list[i])

def _sub_cell_idx(mesh, ci, sf_list, pm, rx, ry, rz, cell_base):
    s = set()
    for fi2 in range(mesh.n_faces):
        if int(mesh.owner[fi2].item()) == ci: s.update(mesh.faces[fi2].tolist())
        elif fi2 < mesh.n_internal_faces and int(mesh.neighbour[fi2].item()) == ci: s.update(mesh.faces[fi2].tolist())
    cpt = mesh.points[list(s)]; mid = 0.5 * (cpt.min(dim=0).values + cpt.max(dim=0).values)
    ap = pm.pts(); res = []
    for sf in sf_list:
        sfc = ap[sf].mean(dim=0); bits = [0, 0, 0]
        for do, c in [(rx, 0), (ry, 1), (rz, 2)]:
            if do and sfc[c] > mid[c]: bits[c] = 1
        res.append(cell_base[ci] + _b2i(bits, rx, ry, rz))
    return res

def split_face(pts, rx, ry, rz, fp, pm):
    e1 = pts[1] - pts[0]; e2 = pts[3] - pts[0]
    nm = torch.linalg.cross(e1, e2); n_mag = nm.norm().item()
    nd = nm.abs().argmax().item() if n_mag > 1e-10 else 3 - e1.abs().argmax().item() - e2.abs().argmax().item()
    dirs = []
    if rx and nd != 0: dirs.append(0)
    if ry and nd != 1: dirs.append(1)
    if rz and nd != 2: dirs.append(2)
    if not dirs: return [fp.clone()]
    cur = [fp.clone()]
    for di in dirs:
        nxt = []
        for f in cur: lo, hi = _sq(f, di, pm); nxt.extend([lo, hi])
        cur = nxt
    return cur

def _sq(fi, di, pm):
    v = fi.tolist(); pt = pm.pts()
    e01 = pt[v[1]] - pt[v[0]]; e32 = pt[v[2]] - pt[v[3]]
    if e01.abs()[di] > 0.5 * e01.norm() and e32.abs()[di] > 0.5 * e32.norm():
        # edges v0-v1 and v3-v2 parallel to split direction
        m0 = pm.add(0.5 * (pt[v[0]] + pt[v[1]])); m1 = pm.add(0.5 * (pt[v[3]] + pt[v[2]]))
        return (torch.tensor([v[0], m0, m1, v[3]], dtype=INDEX_DTYPE),
                torch.tensor([m0, v[1], v[2], m1], dtype=INDEX_DTYPE))
    else:
        # edges v0-v3 and v1-v2 parallel to split direction
        m0 = pm.add(0.5 * (pt[v[0]] + pt[v[3]])); m1 = pm.add(0.5 * (pt[v[1]] + pt[v[2]]))
        return (torch.tensor([v[0], v[1], m1, m0], dtype=INDEX_DTYPE),
                torch.tensor([m0, m1, v[2], v[3]], dtype=INDEX_DTYPE))

_HEX_SB = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
_HEX_ED = [(0,1),(3,2),(4,5),(7,6),(0,3),(1,2),(4,7),(5,6),(0,4),(1,5),(3,7),(2,6)]

def _generate_midplanes(ci, mesh, rx, ry, rz, pm, rfaces, cell_base, n_sub):
    corners = _cell_corners(ci, mesh)
    if len(corners) != 8: return
    for d in range(3):
        if not ((d == 0 and rx) or (d == 1 and ry) or (d == 2 and rz)): continue
        all_pts = pm.pts(); mid_idx = []
        for e0, e1 in _HEX_ED:
            if _HEX_SB[e0][d] != _HEX_SB[e1][d]:
                mid_idx.append(pm.add(0.5 * (all_pts[corners[e0]] + all_pts[corners[e1]])))
        if len(mid_idx) != 4: continue
        all_pts = pm.pts(); mp = all_pts[mid_idx]; ctr = mp.mean(dim=0)
        nd = [i for i in range(3) if i != d]
        ang = torch.atan2(mp[:, nd[1]] - ctr[nd[1]], mp[:, nd[0]] - ctr[nd[0]])
        oi = [mid_idx[i] for i in torch.argsort(ang).tolist()]
        p0 = all_pts[oi[0]]; p1 = all_pts[oi[1]]; p2 = all_pts[oi[2]]
        if torch.linalg.cross(p1 - p0, p2 - p0)[d].item() < 0: oi = [oi[0]] + oi[1:][::-1]
        mf = torch.tensor(oi, dtype=INDEX_DTYPE); mpc = all_pts[mf]
        sdirs = [dd for dd in range(3) if dd != d and ((dd == 0 and rx) or (dd == 1 and ry) or (dd == 2 and rz))]
        sfl = split_face(mpc, rx, ry, rz, mf, pm) if sdirs else [mf]
        all_pts = pm.pts()
        cpt = all_pts[[corners[i] for i in range(8)]]
        cmid = 0.5 * (cpt.min(dim=0).values + cpt.max(dim=0).values)
        for sf in sfl:
            sfc = all_pts[sf].float().mean(dim=0); blo = [0,0,0]; bhi = [0,0,0]
            for dd in range(3):
                if dd == d: blo[dd] = 0; bhi[dd] = 1
                elif (dd == 0 and rx) or (dd == 1 and ry) or (dd == 2 and rz):
                    v = 0 if sfc[dd].item() <= cmid[dd].item() else 1; blo[dd] = v; bhi[dd] = v
            o = cell_base[ci] + _b2i(blo, rx, ry, rz); n = cell_base[ci] + _b2i(bhi, rx, ry, rz)
            k = tuple(sorted(sf.tolist()))
            if k not in rfaces: rfaces[k] = (sf, o, n)

def _assemble(mesh, pm, ufaces, rfaces):
    from pyfoam.mesh.fv_mesh import FvMesh
    all_pts = pm.pts(); int_f = []; int_o = []; int_n = []; bnd_f = []; bnd_o = []
    for fp, o, n in ufaces:
        if n >= 0: int_f.append(fp); int_o.append(o); int_n.append(n)
        else: bnd_f.append(fp); bnd_o.append(o)
    for fp, o, n in rfaces.values():
        if n >= 0:
            if o > n: o, n = n, o
            int_f.append(fp); int_o.append(o); int_n.append(n)
        else: bnd_f.append(fp); bnd_o.append(o)
    all_f = int_f + bnd_f; all_o = int_o + bnd_o; n_int = len(int_n); n_total = len(all_f)
    boundary = []; n_bnd = n_total - n_int
    if n_bnd > 0:
        total_orig = sum(p["nFaces"] for p in mesh.boundary); start = n_int; assigned = 0
        for i, patch in enumerate(mesh.boundary):
            nn = n_bnd - assigned if i == len(mesh.boundary) - 1 else max(1, round(patch["nFaces"] * n_bnd / total_orig))
            boundary.append({"name": patch["name"], "type": patch["type"], "startFace": start, "nFaces": nn})
            start += nn; assigned += nn
    return FvMesh(points=all_pts, faces=all_f, owner=torch.tensor(all_o, dtype=INDEX_DTYPE),
        neighbour=torch.tensor(int_n, dtype=INDEX_DTYPE), boundary=boundary, validate=False)
