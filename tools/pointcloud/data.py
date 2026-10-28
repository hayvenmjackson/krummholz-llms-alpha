"""Load Carl's forest point clouds (x y z category per line, ShapeNetPart-style splits).

Each extracted dataset folder holds 12345678/<seed>_<a>_<b>.txt plus
train_test_split/shuffled_{train,val,test}_file_list.json. Files are cached as
float32/int64 .npz on first read.
"""
import json
from pathlib import Path

import numpy as np


def split_files(root, split):
    root = Path(root)
    names = json.loads((root / "train_test_split" / f"shuffled_{split}_file_list.json").read_text())
    files = [root / "12345678" / (Path(n).name + ".txt") for n in names]
    return [f for f in files if f.exists()], len(names)


def load(path):
    cache = Path(str(path) + ".npz")
    if cache.exists():
        z = np.load(cache)
        return z["xyz"], z["label"]
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    xyz, label = a[:, :3].astype(np.float32), a[:, 3].astype(np.int64)
    np.savez(cache, xyz=xyz, label=label)
    return xyz, label


def up_axis(files, n=20):
    """The axis along which terrain points vary least is vertical."""
    stds = []
    for f in files[:n]:
        xyz, lab = load(f)
        t = xyz[lab == 0]
        if len(t) > 50:
            stds.append(t.std(0))
    return int(np.argmin(np.mean(stds, 0)))


def to_z_up(xyz, axis):
    order = [i for i in range(3) if i != axis] + [axis]
    return xyz[:, order]


def normalize(xyz):
    """Center on the xy centroid and ground, scale by max horizontal radius. Returns (xyz, height feature, scale)."""
    c = xyz.mean(0)
    c[2] = xyz[:, 2].min()
    p = xyz - c
    s = float(np.sqrt((p[:, :2] ** 2).sum(1)).max()) or 1.0
    p = p / s
    h = (xyz[:, 2] - xyz[:, 2].min()) / max(float(np.ptp(xyz[:, 2])), 1e-6)
    return p.astype(np.float32), h[:, None].astype(np.float32), s


class CloudSet:
    def __init__(self, files, axis, npoints=8192, augment=False, seed=0):
        self.files, self.axis, self.n, self.augment = files, axis, npoints, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        xyz, lab = load(self.files[i])
        xyz = to_z_up(xyz, self.axis)
        pick = self.rng.choice(len(xyz), self.n, replace=len(xyz) < self.n)
        xyz, lab = xyz[pick], lab[pick]
        p, h, _ = normalize(xyz)
        if self.augment:
            t = self.rng.uniform(0, 2 * np.pi)
            r = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]], np.float32)
            p = (p @ r.T) * self.rng.uniform(0.9, 1.1) + self.rng.normal(0, 0.002, p.shape).astype(np.float32)
        return p.astype(np.float32), h, lab
