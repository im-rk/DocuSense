"""Whole pipeline (file -> JSON) with tiny freshly-trained checkpoints and OCR replaced by the synthetic ground-truth words."""
import pytest

torch = pytest.importorskip("torch")

from data.synth_invoices import build_synth, make_invoice


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    from models.classifier.architecture import build_classifier
    from models.tagger.labels import DOC_CLASSES
    from models.tagger.train import train_tagger
    d = tmp_path_factory.mktemp("pipe")
    build_synth(str(d), n_train=12, n_val=4, audit=0)
    train_tagger(str(d), str(d), epochs=1, batch_size=4, num_workers=0, train_sets=("synth_train",), val_sets=("synth_val",),
                 test_sets=(), tag="tagger", patience=3)
    torch.save({"state_dict": build_classifier("resnet", 5).state_dict(), "config": {"arch": "resnet", "n_cls": 5},
                "classes": DOC_CLASSES, "mean": 0.9, "std": 0.2, "metrics": {}}, d / "classifier_v1.pth")
    return d


def test_process_end_to_end(ckpts, tmp_path, monkeypatch):
    from config.settings import settings
    import pipeline.loader as loader
    import pipeline.run as run
    monkeypatch.setattr(settings, "classifier_ckpt", str(ckpts / "classifier_v1.pth"))
    monkeypatch.setattr(settings, "tagger_ckpt", str(ckpts / "tagger_v1.pth"))
    monkeypatch.setattr(loader, "_models", None)

    im, words = make_invoice(seed=7)
    path = tmp_path / "inv.png"
    im.save(path)
    monkeypatch.setattr(run, "get_words", lambda *a, **k: [{"text": w["text"], "box": w["box"], "conf": 90} for w in words])

    res = run.process(str(path))                                    # uses the real loader -> checks checkpoint loading too
    assert res["document_type"] in ("RECEIPT", "INVOICE", "FORM", "LETTER", "OTHER")
    assert len(res["words"]) == len(words) and res["image_size"] == {"width": im.width, "height": im.height}
    assert {"fields", "line_items", "field_confidences", "validation", "inference_ms"} <= set(res)
    assert res["validation"]["checks"].keys() == {"subtotal_plus_tax_equals_total", "items_sum_equals_subtotal"}
    assert all(w["label"] for w in res["words"])

    monkeypatch.setattr(run, "get_words", lambda *a, **k: [])       # blank page -> flagged, no crash
    blank = run.process(str(path))
    assert blank["validation"]["needs_review"] and blank["fields"] == {}


def test_missing_checkpoints_message(monkeypatch, tmp_path):
    from config.settings import settings
    import pipeline.loader as loader
    monkeypatch.setattr(settings, "classifier_ckpt", str(tmp_path / "nope1.pth"))
    monkeypatch.setattr(settings, "tagger_ckpt", str(tmp_path / "nope2.pth"))
    monkeypatch.setattr(loader, "_models", None)
    with pytest.raises(loader.ModelsNotFound, match="KAGGLE_TRAINING"):
        loader.get_models()


def test_settings_ignore_unprefixed_keys_in_env_file(tmp_path, monkeypatch):
    """A stale v1 .env (DATABASE_URL=postgres...) must not override our settings; DOCUSENSE_* keys still work."""
    from config.settings import Settings
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://x@localhost/db\nDOCUSENSE_TESSERACT_CMD=/opt/tess\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCUSENSE_DATABASE_URL", raising=False)
    s = Settings()
    assert s.database_url == "sqlite:///docusense.db" and s.tesseract_cmd == "/opt/tess"
