"""PointNet++ (single-scale grouping) for per-point forest segmentation, pure PyTorch.

No custom CUDA ops, so it trains on any PyTorch GPU pod and exports with torch.jit.
Classes follow Carl's datasets: 0 terrain, 1 trunk, 2 canopy, 3 understory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["terrain", "trunk", "canopy", "understory"]


def fps(xyz, n):
    """Farthest point sampling. xyz (B, N, 3) -> indices (B, n)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, n, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(n):
        idx[:, i] = far
        d = ((xyz - xyz[batch, far][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


def gather(x, idx):
    """x (B, N, C), idx (B, ...) -> (B, ..., C)."""
    B = x.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(x, 1, flat[..., None].expand(-1, -1, x.shape[-1]))
    return out.reshape(*idx.shape, x.shape[-1])


def ball_query(xyz, centers, radius, k):
    """Up to k neighbors within radius; pads with the nearest neighbor. -> (B, S, k)."""
    d = torch.cdist(centers, xyz)
    d_k, idx = d.topk(k, dim=-1, largest=False)
    nearest = idx[..., :1].expand_as(idx)
    return torch.where(d_k <= radius, idx, nearest)


class SA(nn.Module):
    def __init__(self, npoint, radius, k, cin, mlp):
        super().__init__()
        self.npoint, self.radius, self.k = npoint, radius, k
        layers, c = [], cin + 3
        for m in mlp:
            layers += [nn.Conv2d(c, m, 1), nn.BatchNorm2d(m), nn.ReLU(inplace=True)]
            c = m
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat):
        centers = gather(xyz, fps(xyz.detach(), self.npoint))
        nb = ball_query(xyz, centers, self.radius, self.k)
        g = gather(xyz, nb) - centers[:, :, None]
        if feat is not None:
            g = torch.cat([g, gather(feat, nb)], -1)
        g = self.mlp(g.permute(0, 3, 1, 2))  # (B, C, S, k)
        return centers, g.max(-1).values.transpose(1, 2)


class FP(nn.Module):
    def __init__(self, cin, mlp):
        super().__init__()
        layers, c = [], cin
        for m in mlp:
            layers += [nn.Conv1d(c, m, 1), nn.BatchNorm1d(m), nn.ReLU(inplace=True)]
            c = m
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, feat1, feat2):
        d, idx = torch.cdist(xyz1, xyz2).topk(3, dim=-1, largest=False)
        w = 1.0 / (d + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (gather(feat2, idx) * w[..., None]).sum(2)
        x = interp if feat1 is None else torch.cat([feat1, interp], -1)
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=4, in_feat=1):
        super().__init__()
        self.sa1 = SA(2048, 0.05, 32, in_feat, [32, 32, 64])
        self.sa2 = SA(512, 0.1, 32, 64, [64, 64, 128])
        self.sa3 = SA(128, 0.2, 32, 128, [128, 128, 256])
        self.sa4 = SA(32, 0.4, 32, 256, [256, 256, 512])
        self.fp4 = FP(768, [256, 256])
        self.fp3 = FP(384, [256, 256])
        self.fp2 = FP(320, [256, 128])
        self.fp1 = FP(128 + in_feat, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Conv1d(128, num_classes, 1))

    def forward(self, xyz, feat):
        """xyz (B, N, 3) normalized; feat (B, N, in_feat). Returns logits (B, N, classes)."""
        l1, f1 = self.sa1(xyz, feat)
        l2, f2 = self.sa2(l1, f1)
        l3, f3 = self.sa3(l2, f2)
        l4, f4 = self.sa4(l3, f3)
        f3 = self.fp4(l3, l4, f3, f4)
        f2 = self.fp3(l2, l3, f2, f3)
        f1 = self.fp2(l1, l2, f1, f2)
        f0 = self.fp1(xyz, l1, feat, f1)
        return self.head(f0.transpose(1, 2)).transpose(1, 2)
