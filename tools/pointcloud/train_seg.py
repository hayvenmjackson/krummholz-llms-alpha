"""Train PointNet++ on one or more of Carl's forest datasets; report per-dataset test IoU."""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import data
from pointnet2 import CLASSES, PointNet2Seg


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=8, drop_last=shuffle,
                      persistent_workers=True, worker_init_fn=lambda w: np.random.seed(w + int(time.time())))


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    inter, union, correct, total = np.zeros(4), np.zeros(4), 0, 0
    for p, h, lab in dl:
        pred = model(p.cuda(), h.cuda()).argmax(-1).cpu()
        correct += (pred == lab).sum().item(); total += lab.numel()
        for c in range(4):
            inter[c] += ((pred == c) & (lab == c)).sum().item()
            union[c] += ((pred == c) | (lab == c)).sum().item()
    model.train()
    iou = inter / np.maximum(union, 1)
    return {"accuracy": round(correct / max(total, 1), 4), "miou": round(float(iou.mean()), 4),
            "iou": {CLASSES[c]: round(float(iou[c]), 4) for c in range(4)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="name=path for each extracted dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--npoints", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hours", type=float, default=4.0, help="stop training after this wall time")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    sets, info = {}, {}
    for spec in a.data:
        name, root = spec.split("=", 1)
        files = {s: data.split_files(root, s) for s in ("train", "val", "test")}
        axis = data.up_axis(files["train"][0])
        sets[name] = {s: data.CloudSet(f, axis, a.npoints, augment=(s == "train"), seed=i)
                      for i, (s, (f, _)) in enumerate(files.items())}
        info[name] = {"root": root, "up_axis": axis,
                      **{s: {"present": len(f), "listed": n} for s, (f, n) in files.items()}}
        log(f"{name}: {info[name]}")

    counts = np.zeros(4)
    for ds in sets.values():
        for f in ds["train"].files[:200]:
            counts += np.bincount(data.load(f)[1], minlength=4)[:4]
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    weights = (weights / weights.mean()).cuda()
    log(f"class point counts (sample) {counts.astype(int).tolist()} weights {weights.tolist()}")

    train = loader(ConcatDataset([d["train"] for d in sets.values()]), a.batch, True)
    vals = {n: loader(d["val"], a.batch, False) for n, d in sets.items()}
    model = PointNet2Seg().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(train))
    started, best, history = time.time(), -1.0, []
    for epoch in range(1, a.epochs + 1):
        losses = []
        for step, (p, h, lab) in enumerate(train, 1):
            loss = F.cross_entropy(model(p.cuda(), h.cuda()).reshape(-1, 4), lab.cuda().reshape(-1), weight=weights)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
            if step % 100 == 0:
                log(f"epoch {epoch} step {step}/{len(train)} loss {np.mean(losses[-100:]):.4f}")
        val = {n: evaluate(model, dl) for n, dl in vals.items()}
        mean_miou = float(np.mean([v["miou"] for v in val.values()]))
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "val": val})
        log(f"epoch {epoch} loss {np.mean(losses):.4f} val {json.dumps(val)}")
        if mean_miou > best:
            best = mean_miou
            torch.save(model.state_dict(), out / "pointnet2-forest.pt")
        if time.time() - started > a.hours * 3600:
            log("wall-time cap reached"); break

    model.load_state_dict(torch.load(out / "pointnet2-forest.pt"))
    test = {n: evaluate(model, loader(d["test"], a.batch, False)) for n, d in sets.items()}
    log(f"TEST {json.dumps(test)}")
    (out / "pointnet2-results.json").write_text(json.dumps({
        "model": "PointNet++ SSG, pure PyTorch (tools/pointcloud/pointnet2.py)", "classes": CLASSES,
        "npoints": a.npoints, "epochs_run": len(history), "batch": a.batch, "lr": a.lr,
        "seconds": round(time.time() - started), "datasets": info, "best_val_mean_miou": best,
        "test": test, "history": history}, indent=1))


if __name__ == "__main__":
    main()
