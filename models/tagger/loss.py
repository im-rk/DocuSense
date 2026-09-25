"""Partial-label loss + class weights for Model 2."""
import numpy as np
import torch


def partial_ce(logits, labels, unannot, w):
    """Cross-entropy that copes with datasets annotating DIFFERENT class subsets.

    logits (T,C) | labels (T,) | unannot (T,C) bool: classes the token's source dataset never annotates | w (C,)
    A token labelled OTHER in such a source may really belong to one of the unannotated classes (a CORD vendor line is
    labelled OTHER only because CORD has no VENDOR class), so its loss is  -log( p(OTHER) + sum p(unannotated) ).
    Tokens with a real label use ordinary cross-entropy.
    """
    logp = logits.log_softmax(-1)
    ce = -logp.gather(1, labels[:, None]).squeeze(1)
    allowed = unannot.clone()
    allowed[:, 0] = True                                                     # OTHER is always allowed
    marg = -torch.logsumexp(logp.masked_fill(~allowed, float("-inf")), dim=-1)
    loss = torch.where(labels == 0, marg, ce)
    wt = w[labels]
    return (loss * wt).sum() / wt.sum()


def class_weights(label_counts, max_w: float = 10.0) -> np.ndarray:
    """sqrt-inverse-frequency weights; classes never seen get 1."""
    c = np.asarray(label_counts, dtype=np.float64)
    seen = c > 0
    w = np.ones_like(c)
    w[seen] = np.sqrt(c.sum() / (seen.sum() * c[seen]))
    return np.clip(w, 0.05, max_w).astype(np.float32)
