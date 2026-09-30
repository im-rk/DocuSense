"""Load both trained models ONCE per process (thread-safe)."""
import os
import threading
from dataclasses import dataclass

from config.settings import settings


class ModelsNotFound(RuntimeError):
    pass


@dataclass
class Models:
    classifier: object
    cls_meta: dict
    tagger: object
    vocab: dict
    tag_meta: dict
    versions: dict


_lock = threading.Lock()
_models = None


def models_status() -> dict:
    missing = [p for p in (settings.classifier_ckpt, settings.tagger_ckpt) if not os.path.exists(p)]
    return {"loaded": _models is not None, "missing": missing}


def get_models() -> Models:
    global _models
    with _lock:
        if _models is None:
            missing = models_status()["missing"]
            if missing:
                raise ModelsNotFound(
                    "Trained model files not found: " + ", ".join(missing) +
                    ". Train on Kaggle (see KAGGLE_TRAINING.md) and copy the .pth files into models/checkpoints/.")
            from models.classifier.predict import load_classifier
            from models.tagger.predict import load_tagger
            clf, cls_meta = load_classifier(settings.classifier_ckpt, settings.device)
            tag, vocab, tag_meta = load_tagger(settings.tagger_ckpt, settings.device)
            _models = Models(clf, cls_meta, tag, vocab, tag_meta, {
                "classifier": os.path.basename(settings.classifier_ckpt), "tagger": os.path.basename(settings.tagger_ckpt)})
        return _models
