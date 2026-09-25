"""Model 1 training loop (written by us; runs on Kaggle GPU).  Everything is initialised from scratch."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.classifier.architecture import build_classifier
from models.classifier.dataset import ClsDataset
from models.classifier.evaluate import evaluate, plot_confusion, plot_curves
from models.tagger.labels import DOC_CLASSES


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_classifier(data_dir, out_dir, epochs=30, batch_size=64, lr=2e-3, weight_decay=1e-2, arch="resnet",
                     augment=True, seed=42, num_workers=2, resume=True, tag="classifier", max_train=None):
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    stats = json.load(open(os.path.join(data_dir, "cls_stats.json")))
    mean, std, n_cls = stats["mean"], stats["std"], len(DOC_CLASSES)

    dl = lambda ds, shuffle: DataLoader(ds, batch_size, shuffle=shuffle, num_workers=num_workers,
                                        pin_memory=use_amp, drop_last=shuffle, persistent_workers=num_workers > 0)
    tr_ds = ClsDataset(data_dir, "train", mean, std, augment=augment, max_items=max_train)
    train_loader = dl(tr_ds, True)
    val_loader = dl(ClsDataset(data_dir, "val", mean, std), False)
    test_loader = dl(ClsDataset(data_dir, "test", mean, std), False)

    counts = np.bincount(tr_ds.y[: len(tr_ds)], minlength=n_cls).astype(np.float64)
    w = np.where(counts > 0, (counts.sum() / (n_cls * np.maximum(counts, 1))) ** 0.5, 1.0)    # sqrt-inverse frequency
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=0.05)

    model = build_classifier(arch, n_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] arch={arch} params={n_params/1e6:.2f}M device={device} train={len(tr_ds)} class_counts={counts.astype(int).tolist()}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    last_p, best_p = os.path.join(out_dir, f"{tag}_last.pth"), os.path.join(out_dir, f"{tag}_best.pth")
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    start, best = 0, -1.0
    if resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, start, best = ck["hist"], ck["epoch"] + 1, ck["best"]
        print(f"[{tag}] resumed from epoch {start}")

    for epoch in range(start, epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item() * len(y); seen += len(y)
        val = evaluate(model, val_loader, device, n_cls, criterion)
        hist["train_loss"].append(run / seen); hist["val_loss"].append(val["loss"]); hist["val_acc"].append(val["acc"])
        print(f"[{tag}] epoch {epoch+1:02d}/{epochs} train_loss={run/seen:.4f} val_loss={val['loss']:.4f} "
              f"val_acc={val['acc']:.4f} ({time.time()-t0:.0f}s)")
        if val["acc"] > best:
            best = val["acc"]
            torch.save(model.state_dict(), best_p)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "epoch": epoch, "best": best}, last_p)          # per-epoch checkpoint (resume)

    model.load_state_dict(torch.load(best_p, map_location=device))
    test = evaluate(model, test_loader, device, n_cls, criterion)
    metrics = {"val_acc_best": best, "test_acc": test["acc"], "params": n_params, "arch": arch, "augment": augment,
               "test_recall": {c: float(r) for c, r in zip(DOC_CLASSES, test["recall"])}}
    print(f"[{tag}] TEST acc={test['acc']:.4f} recall={metrics['test_recall']}")
    try:
        plot_curves(hist, os.path.join(out_dir, f"{tag}_curves.png"))
        plot_confusion(test["confusion"], DOC_CLASSES, os.path.join(out_dir, f"{tag}_confusion.png"))
    except Exception as e:                                                          # plots are nice-to-have
        print("plotting skipped:", e)
    torch.save({"state_dict": model.state_dict(), "config": {"arch": arch, "n_cls": n_cls}, "classes": DOC_CLASSES,
                "mean": mean, "std": std, "metrics": metrics}, os.path.join(out_dir, f"{tag}_v1.pth"))
    json.dump({"metrics": metrics, "history": hist, "confusion": test["confusion"].tolist()},
              open(os.path.join(out_dir, f"{tag}_report.json"), "w"), indent=1)
    return metrics


def sanity_overfit(data_dir, n=64, steps=120, arch="resnet", seed=0):
    """Code-correctness check: the model must be able to memorise 64 images (accuracy -> ~100%)."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = json.load(open(os.path.join(data_dir, "cls_stats.json")))
    ds = ClsDataset(data_dir, "train", st["mean"], st["std"], augment=False)
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    x = torch.stack([ds[i][0] for i in idx]).to(device)
    y = torch.tensor([ds[i][1] for i in idx]).to(device)
    m = build_classifier(arch, len(DOC_CLASSES)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for s in range(steps):
        m.train(); loss = nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0:
            print(f"initial loss {loss.item():.3f} (expect roughly ln 5 = 1.61)")
    m.eval()
    with torch.no_grad():
        acc = (m(x).argmax(1) == y).float().mean().item()
    print(f"overfit-{n} accuracy: {acc:.3f} (should be ~1.0)")
    return acc
