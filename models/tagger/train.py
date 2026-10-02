"""Model 2 training loop (written by us; runs on Kaggle GPU). Everything is initialised from scratch."""
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.tagger.architecture import build_tagger, count_params
from models.tagger.dataset import TaggerDataset, collate, label_counts, load_records
from models.tagger.evaluate import field_accuracy, meter_from_preds, predict_records
from models.tagger.labels import LABELS, N_CLS
from models.tagger.loss import class_weights, partial_ce
from models.tagger.vocab import build_vocab, save_vocab, vocab_size

DEFAULTS = dict(
    epochs=80, batch_size=8, lr=1e-3, weight_decay=1e-2, warmup_steps=100, patience=12, grad_clip=1.0,
    # model
    dropout=0.3, d=256, hid=192, lstm_layers=2, use_text=True, use_look=True, use_pos=True, use_lstm=True,
    # data
    train_sets=("cord_train", "sroie_train", "synth_train"), val_sets=("cord_val", "sroie_val"),
    test_sets=("cord_test", "sroie_test"), synth_frac=0.3, augment=True, max_docs=None,
    # loss
    partial_loss=True, class_weights=True,
    seed=42, num_workers=2, tag="tagger", resume=True)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _exists(data_dir, names):
    return [n for n in names if os.path.exists(os.path.join(data_dir, f"{n}.jsonl"))]


def _to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _loss(model, b, w, partial):
    logits = model(b["chars"], b["crops"], b["pos"], b["pad"])
    real = ~b["pad"]
    un = b["unannot"][:, None, :].expand(-1, logits.shape[1], -1)[real]
    if not partial:
        un = torch.zeros_like(un)                       # naive cross-entropy (ablation)
    return partial_ce(logits[real], b["labels"][real], un, w)


def train_tagger(data_dir, out_dir, **overrides):
    cfg = {**DEFAULTS, **overrides}
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = cfg["tag"]

    # ---- data (synthetic invoices are capped to synth_frac of the training set) ----
    tr_names, va_names, te_names = (_exists(data_dir, cfg[k]) for k in ("train_sets", "val_sets", "test_sets"))
    real_names = [n for n in tr_names if not n.startswith("synth")]
    n_real = sum(len(load_records(data_dir, n)[0]) for n in real_names)
    limits = {n: int(n_real * cfg["synth_frac"] / (1 - cfg["synth_frac"])) for n in tr_names if n.startswith("synth")} \
        if n_real else {}                                   # no real data (smoke tests): use all synthetic docs
    if cfg["max_docs"]:
        limits = {**limits, **{n: cfg["max_docs"] for n in real_names}}

    probe = TaggerDataset(data_dir, tr_names, {}, augment=False, limits=limits)
    chars = build_vocab((w["text"] for r in probe.records for w in r["words"]), min_count=2)
    save_vocab(chars, os.path.join(out_dir, f"{tag}_vocab.json"))
    vocab = {ch: i + 2 for i, ch in enumerate(chars)}
    train_ds = TaggerDataset(data_dir, tr_names, vocab, augment=cfg["augment"], limits=limits)
    val_ds = TaggerDataset(data_dir, va_names, vocab)
    dl = lambda ds, bs, sh: DataLoader(ds, bs, shuffle=sh, collate_fn=collate, num_workers=cfg["num_workers"],
                                       persistent_workers=cfg["num_workers"] > 0)
    train_loader = dl(train_ds, cfg["batch_size"], True)

    counts = label_counts(train_ds.records)
    w = torch.tensor(class_weights(counts) if cfg["class_weights"] else np.ones(N_CLS, np.float32), device=device)

    mcfg = dict(vocab=vocab_size(vocab), n_cls=N_CLS, p=cfg["dropout"], d=cfg["d"], hid=cfg["hid"],
                use_text=cfg["use_text"], use_look=cfg["use_look"], use_pos=cfg["use_pos"], use_lstm=cfg["use_lstm"],
                lstm_layers=cfg["lstm_layers"])
    model = build_tagger(mcfg).to(device)
    n_params = count_params(model)
    print(f"[{tag}] params={n_params/1e6:.2f}M device={device} train_docs={len(train_ds)} val_docs={len(val_ds)} "
          f"vocab={len(vocab)+2} label_counts={dict(zip(LABELS, counts.tolist()))}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_loader)
    lr_at = lambda s: (s + 1) / cfg["warmup_steps"] if s < cfg["warmup_steps"] else \
        0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    last_p, best_p = os.path.join(out_dir, f"{tag}_last.pth"), os.path.join(out_dir, f"{tag}_best.pth")
    hist = {"train_loss": [], "val_macro_f1": []}
    start, best, bad = 0, -1.0, 0
    if cfg["resume"] and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, start, best, bad = ck["hist"], ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"[{tag}] resumed from epoch {start}")

    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for batch in train_loader:
            loss = _loss(model, _to(batch, device), w, cfg["partial_loss"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step(); sched.step()
            run += loss.item(); nb += 1
        rep = meter_from_preds(predict_records(model, val_ds, device)).report()
        f1 = rep["macro_f1"]
        hist["train_loss"].append(run / max(nb, 1)); hist["val_macro_f1"].append(f1)
        per = {k: v["f1"] for k, v in rep.items() if k != "macro_f1"}
        print(f"[{tag}] epoch {epoch+1:02d}/{cfg['epochs']} loss={run/max(nb,1):.4f} val_macroF1={f1:.4f} "
              f"per-class={per} ({time.time()-t0:.0f}s)")
        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), best_p)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "epoch": epoch, "best": best, "bad": bad}, last_p)     # per-epoch checkpoint
        if bad >= cfg["patience"]:
            print(f"[{tag}] early stop (no improvement for {cfg['patience']} epochs)")
            break

    # ---- final evaluation on the held-out test sets with the best weights ----
    model.load_state_dict(torch.load(best_p, map_location=device, weights_only=False))
    results, pooled = {}, []
    for name in te_names:
        preds = predict_records(model, TaggerDataset(data_dir, [name], vocab), device)
        results[name] = {"tokens": meter_from_preds(preds).report(), "fields": field_accuracy(preds)}
        pooled += preds
    results["ALL_TEST"] = {"tokens": meter_from_preds(pooled).report(), "fields": field_accuracy(pooled)} if pooled else {}
    metrics = {"val_macro_f1_best": best, "params": n_params, "config": {k: v for k, v in cfg.items() if k != "resume"},
               "test": results}
    print(f"[{tag}] TEST ALL macro-F1={results.get('ALL_TEST', {}).get('tokens', {}).get('macro_f1')} "
          f"fields={results.get('ALL_TEST', {}).get('fields')}")
    torch.save({"state_dict": model.state_dict(), "config": mcfg, "chars": chars, "labels": LABELS, "metrics": metrics},
               os.path.join(out_dir, f"{tag}_v1.pth"))
    json.dump({"metrics": metrics, "history": hist}, open(os.path.join(out_dir, f"{tag}_report.json"), "w"), indent=1)
    try:
        from models.classifier.evaluate import plot_curves
        plot_curves(hist, os.path.join(out_dir, f"{tag}_curves.png"), keys=(("train_loss",), ("val_macro_f1",)))
    except Exception as e:
        print("plotting skipped:", e)
    return metrics


def sanity_overfit(data_dir, n_docs=8, steps=250, seed=0):
    """Code-correctness check: the model must memorise 8 documents (token accuracy -> ~100%)."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = _exists(data_dir, ("cord_train", "sroie_train"))
    probe = TaggerDataset(data_dir, names[:1], {}, limits={names[0]: n_docs})
    chars = build_vocab((w["text"] for r in probe.records for w in r["words"]), min_count=1)
    vocab = {ch: i + 2 for i, ch in enumerate(chars)}
    ds = TaggerDataset(data_dir, names[:1], vocab, limits={names[0]: n_docs})
    model = build_tagger(dict(vocab=vocab_size(vocab), p=0.0)).to(device)
    print(f"params={count_params(model)/1e6:.2f}M")
    b = _to(collate([ds[i] for i in range(len(ds))]), device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    w = torch.ones(N_CLS, device=device)
    for s in range(steps):
        model.train()
        loss = _loss(model, b, w, partial=False)
        if s == 0:
            print(f"initial loss {loss.item():.3f} (expect ~ln {N_CLS} = {math.log(N_CLS):.2f})")
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(b["chars"], b["crops"], b["pos"], b["pad"]).argmax(-1)
        real = ~b["pad"]
        acc = (pred[real] == b["labels"][real]).float().mean().item()
    print(f"overfit-{n_docs}-docs token accuracy: {acc:.3f} (should be ~1.0)")
    return acc
