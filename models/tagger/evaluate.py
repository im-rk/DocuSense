"""Model 2 evaluation: per-class token P/R/F1 (pooled over the sources that annotate each class) + field-level accuracy."""
import re

import numpy as np
import torch

from models.tagger.dataset import collate
from models.tagger.labels import LABELS, N_CLS, annotated_classes, unannot_mask
from models.tagger.postprocess import extract


class Meter:
    """Token-level counts. Per source, only the classes that source annotates are scored, and predictions of a class the
    source does not annotate count as OTHER (we cannot know they are wrong)."""

    def __init__(self):
        self.tp, self.fp, self.fn = (np.zeros(N_CLS, dtype=np.int64) for _ in range(3))

    def update(self, pred: np.ndarray, gold: np.ndarray, source: str):
        pred = np.where(np.array(unannot_mask(source))[pred], 0, pred)
        for c in annotated_classes(source):
            self.tp[c] += int(((pred == c) & (gold == c)).sum())
            self.fp[c] += int(((pred == c) & (gold != c)).sum())
            self.fn[c] += int(((pred != c) & (gold == c)).sum())

    def report(self) -> dict:
        out, f1s = {}, []
        for c in range(1, N_CLS):
            sup = int(self.tp[c] + self.fn[c])
            if sup == 0 and self.fp[c] == 0:
                continue
            p = self.tp[c] / max(self.tp[c] + self.fp[c], 1)
            r = self.tp[c] / max(sup, 1)
            f = 2 * p * r / max(p + r, 1e-9)
            out[LABELS[c]] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
                              "support": sup}
            if sup > 0:
                f1s.append(f)
        out["macro_f1"] = round(float(np.mean(f1s)) if f1s else 0.0, 4)
        return out


@torch.no_grad()
def predict_records(model, dataset, device, batch_size=16):
    """-> list of (record, pred_ids (n,), pred_probs (n,)) for every doc in `dataset` (no augmentation)."""
    model.eval()
    out, i = [], 0
    for s in range(0, len(dataset), batch_size):
        batch = collate([dataset[j] for j in range(s, min(s + batch_size, len(dataset)))])
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        prob = model(b["chars"], b["crops"], b["pos"], b["pad"]).softmax(-1).cpu().numpy()
        for k in range(prob.shape[0]):
            n = int((~batch["pad"][k]).sum())
            out.append((dataset.docs[i][0], prob[k, :n].argmax(-1), prob[k, :n].max(-1)))
            i += 1
    return out


def meter_from_preds(preds) -> Meter:
    m = Meter()
    for rec, pl, _ in preds:
        m.update(np.asarray(pl), np.array([w["label"] for w in rec["words"]]), rec["source"])
    return m


def _norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def field_accuracy(preds) -> dict:
    """Field-level exact match after post-processing. Gold fields = the same post-processor applied to the gold labels
    (so it scores the model, not the rules). Only fields the gold actually has are scored."""
    hit, tot = {}, {}
    for rec, pl, pp in preds:
        gw = [dict(text=w["text"], box=w["box"], line=w["line"], label=w["label"], prob=1.0) for w in rec["words"]]
        pw = [dict(g, label=int(l), prob=float(p)) for g, l, p in zip(gw, pl, pp)]
        gold, pred = extract(gw)["fields"], extract(pw)["fields"]
        for f in ("vendor", "date", "total", "subtotal", "tax"):
            if f in gold:
                tot[f] = tot.get(f, 0) + 1
                ok = (_norm(gold[f]) == _norm(pred.get(f, ""))) if f in ("vendor", "date") \
                    else (f in pred and abs(gold[f] - pred[f]) < 0.006)
                hit[f] = hit.get(f, 0) + int(ok)
    return {f: {"acc": round(hit.get(f, 0) / tot[f], 4), "n": tot[f]} for f in tot}
