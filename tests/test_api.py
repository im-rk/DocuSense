"""API tests with the ML pipeline mocked (no checkpoints / Tesseract needed)."""
import io
import time

import pytest

pytest.importorskip("torch")                       # api.tasks imports the pipeline, which imports torch
from fastapi.testclient import TestClient
from PIL import Image

FAKE = {
    "document_type": "RECEIPT", "type_confidence": 0.97, "type_probabilities": {"RECEIPT": 0.97},
    "fields": {"vendor": "ACME", "date": "2018-12-25", "subtotal": 15.0, "tax": 0.9, "total": 15.9},
    "line_items": [{"name": "Nasi", "qty": 1, "price": 15.0}],
    "field_confidences": {"vendor": 0.95, "date": 0.9, "total": 0.99},
    "field_sources": {"vendor": "model"}, "words": [{"text": "ACME", "box": [1, 1, 50, 20], "label": "VENDOR", "prob": 0.9, "line": 0}],
    "image_size": {"width": 100, "height": 100}, "ocr_engine": "tesseract", "model_versions": {},
    "inference_ms": 42,
    "validation": {"passed": True, "needs_review": False, "errors": [], "warnings": [], "checks": {},
                   "missing_fields": [], "low_confidence_fields": []},
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "up"))
    import db.session as dbs
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(dbs, "engine", eng)
    monkeypatch.setattr(dbs, "SessionLocal", sessionmaker(bind=eng, autoflush=False, expire_on_commit=False))
    import api.tasks as tasks
    monkeypatch.setattr(tasks, "SessionLocal", dbs.SessionLocal)
    monkeypatch.setattr(tasks, "process", lambda path: dict(FAKE))
    from api.main import app
    dbs.Base.metadata.create_all(eng)
    def override_db():
        db = dbs.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[dbs.get_db] = override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _png():
    b = io.BytesIO(); Image.new("RGB", (100, 100), "white").save(b, "PNG"); return b.getvalue()


def _upload(c):
    r = c.post("/documents", files={"file": ("r.png", _png(), "image/png")})
    assert r.status_code == 202, r.text
    return r.json()["document_id"]


def test_upload_process_result_export_and_corrections(client):
    assert client.get("/health").json()["status"] == "ok"
    doc_id = _upload(client)
    for _ in range(50):                                            # background task runs right after the response
        body = client.get(f"/documents/{doc_id}").json()
        if body["document"]["status"] not in ("PENDING", "PROCESSING"):
            break
        time.sleep(0.05)
    assert body["document"]["status"] == "COMPLETE" and body["result"]["fields"]["total"] == 15.9
    assert client.get(f"/documents/{doc_id}/image").status_code == 200
    assert [d["id"] for d in client.get("/documents").json()] == [doc_id]

    # correction: wrong total -> arithmetic check now fails -> NEEDS_REVIEW, and the edit is audited
    r = client.put(f"/documents/{doc_id}/fields", json={"fields": {"total": "20,00", "vendor": "ACME LTD"}})
    assert r.status_code == 200
    out = r.json()
    assert out["result"]["fields"]["total"] == 20.0 and out["result"]["fields"]["vendor"] == "ACME LTD"
    assert out["document"]["status"] == "NEEDS_REVIEW"
    assert out["result"]["validation"]["checks"]["subtotal_plus_tax_equals_total"] == "FAIL"
    assert out["result"]["field_confidences"]["total"] == 1.0
    assert client.put(f"/documents/{doc_id}/fields", json={"fields": {"date": "nonsense"}}).status_code == 422
    assert client.put(f"/documents/{doc_id}/fields", json={"fields": {"colour": "red"}}).status_code == 422

    assert client.get(f"/documents/{doc_id}/export?fmt=json").json()["fields"]["total"] == 20.0
    csv = client.get(f"/documents/{doc_id}/export?fmt=csv").text
    assert "vendor,date" in csv and "ACME LTD" in csv and "Nasi" in csv
    x = client.get(f"/documents/{doc_id}/export?fmt=xlsx")
    assert x.status_code == 200 and x.content[:2] == b"PK"

    s = client.get("/analytics/summary").json()
    assert s["total_documents"] == 1 and s["total_corrections"] == 2 and s["correction_rate"] == 1.0
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_upload_validation(client):
    assert client.post("/documents", files={"file": ("x.exe", b"abc", "application/octet-stream")}).status_code == 415
    assert client.post("/documents", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    assert client.get("/documents/nope").status_code == 404


def test_failed_processing_is_recorded(client, monkeypatch):
    import api.tasks as tasks
    from pipeline.loader import ModelsNotFound
    def boom(path): raise ModelsNotFound("checkpoints missing")
    monkeypatch.setattr(tasks, "process", boom)
    doc_id = _upload(client)
    for _ in range(50):
        d = client.get(f"/documents/{doc_id}").json()["document"]
        if d["status"] != "PENDING":
            break
        time.sleep(0.05)
    assert d["status"] == "FAILED" and "checkpoints missing" in d["error"]
