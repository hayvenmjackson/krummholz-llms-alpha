"""Turn a labeled forest point cloud into a tool result the language model can cite.

Labels come from PointNet++ at run time, or from ground truth when building training
examples. Distances are in the scan's own units; Carl's simulator scenes are unitless.
"""
import json, sys

import numpy as np

TOOL = "krummholz.pointcloud.summarize"
VERSION = "1"


def components(cells):
    """8-connected components of occupied integer grid cells -> list of cell lists."""
    todo, groups = set(map(tuple, cells)), []
    while todo:
        stack, group = [todo.pop()], []
        while stack:
            c = stack.pop(); group.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in todo:
                        todo.remove(n); stack.append(n)
        groups.append(group)
    return groups


def summarize(xyz, label, scene_id="scene", trunk_cell=0.75, min_trunk_points=15, cover_cell=2.0):
    """xyz (N, 3) with z up, label (N,) in 0 terrain, 1 trunk, 2 canopy, 3 understory."""
    n = len(xyz)
    counts = np.bincount(label, minlength=4)[:4]
    ground = float(np.median(xyz[label == 0, 2])) if counts[0] else float(xyz[:, 2].min())
    trunks = []
    tp = xyz[label == 1]
    if len(tp):
        keys = np.floor(tp[:, :2] / trunk_cell).astype(int)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        index = {tuple(c): i for i, c in enumerate(uniq)}
        for g in components(uniq):
            sel = np.isin(inv.ravel(), [index[c] for c in g])
            if sel.sum() >= min_trunk_points:
                pts = tp[sel]
                trunks.append({"x": round(float(pts[:, 0].mean()), 2), "y": round(float(pts[:, 1].mean()), 2),
                               "trunk_top": round(float(pts[:, 2].max() - ground), 2)})
    lo, hi = xyz[:, :2].min(0), xyz[:, :2].max(0)
    grid = np.maximum(np.ceil((hi - lo) / cover_cell).astype(int), 1)
    canopy = xyz[label == 2]
    covered = np.unique(np.floor((canopy[:, :2] - lo) / cover_cell).astype(int), axis=0) if len(canopy) else []
    cover = len(covered) / float(grid[0] * grid[1])
    top = float(np.percentile(canopy[:, 2], 95) - ground) if len(canopy) else 0.0
    share = {k: round(float(c) / n, 3) for k, c in zip(["terrain", "trunk", "canopy", "understory"], counts)}
    facts = {
        "tool": TOOL, "version": VERSION, "scene_id": scene_id, "points": int(n),
        "trunk_count": len(trunks), "trunks": trunks[:50], "canopy_cover_fraction": round(cover, 3),
        "canopy_top_height_p95": round(top, 2), "point_share": share,
        "extent": [round(float(v), 2) for v in (hi - lo)],
        "notes": "Counts are clusters of trunk points; touching or occluded trunks can merge or be missed.",
    }
    facts["sentences"] = sentences(facts)
    return facts


def sentences(f):
    s = f["point_share"]
    return [
        f"Scene {f['scene_id']} contains {f['points']} points.",
        f"Trunk clusters detected: {f['trunk_count']}.",
        f"Canopy cover fraction: {f['canopy_cover_fraction']}.",
        f"Canopy top height (95th percentile above ground): {f['canopy_top_height_p95']} units.",
        f"Point shares: terrain {s['terrain']}, trunk {s['trunk']}, canopy {s['canopy']}, understory {s['understory']}.",
        f"Scene extent: {f['extent'][0]} by {f['extent'][1]} units.",
        "Trunk counts are clusters of trunk points; touching or occluded trunks can merge or be missed.",
    ]


if __name__ == "__main__":
    a = np.loadtxt(sys.argv[1], ndmin=2)
    axis = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    order = [i for i in range(3) if i != axis] + [axis]
    print(json.dumps(summarize(a[:, order], a[:, 3].astype(int), sys.argv[1]), indent=1))
