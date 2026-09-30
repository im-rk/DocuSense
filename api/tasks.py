"""Background processing of one uploaded document (runs in FastAPI's thread pool)."""
import logging
from datetime import datetime, timezone

from db.models import Document, Extraction
from db.session import SessionLocal
from pipeline.loader import ModelsNotFound
from pipeline.run import process

log = logging.getLogger("docusense")


def save_result(db, doc: Document, res: dict):
    ex = doc.extraction or Extraction(document_id=doc.id)
    ex.document_type, ex.type_confidence = res["document_type"], res["type_confidence"]
    ex.fields, ex.line_items, ex.confidences = res["fields"], res["line_items"], res["field_confidences"]
    ex.validation, ex.words, ex.image_size = res["validation"], res["words"], res["image_size"]
    ex.extra = {k: res.get(k) for k in ("type_probabilities", "field_sources", "model_versions", "ocr_engine")}
    ex.inference_ms = res["inference_ms"]
    db.add(ex)
    doc.status = "NEEDS_REVIEW" if res["validation"]["needs_review"] else "COMPLETE"


def process_document(doc_id: str):
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            return
        try:
            doc.status = "PROCESSING"
            db.commit()
            res = process(doc.file_path)
            save_result(db, doc, res)
            doc.error = None
        except ModelsNotFound as e:
            doc.status, doc.error = "FAILED", str(e)
        except Exception as e:                                       # never leave a document stuck in PROCESSING
            log.exception("processing failed for %s", doc_id)
            doc.status, doc.error = "FAILED", f"{type(e).__name__}: {e}"[:1000]
        doc.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
