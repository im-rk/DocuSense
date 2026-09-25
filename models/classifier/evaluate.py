"""Model 1 evaluation + plots."""
import numpy as np
import torch


@torch.no_grad()
def evaluate(model, loader, device, n_cls, criterion=None):
    """-> dict(loss, acc, confusion (true rows x pred cols), recall per class)."""
    model.eval()
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        for t, p in zip(y.cpu().tolist(), logits.argmax(1).cpu().tolist()):
            cm[t, p] += 1
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return {"loss": total_loss / max(n, 1), "acc": float(cm.diagonal().sum() / max(cm.sum(), 1)),
            "confusion": cm, "recall": recall}


def plot_confusion(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_curves(history: dict, path: str, keys=(("train_loss", "val_loss"), ("val_acc",))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6))
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, keys):
        for k in group:
            if k in history and history[k]:
                ax.plot(range(1, len(history[k]) + 1), history[k], label=k)
        ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
