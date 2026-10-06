from workers.celery_app import celery_app

@celery_app.task(bind=True)
def process_document(self, document_id: str, file_path: str):
    # STUB: will be filled in Phase 7 after models are trained
    # For now: just update status to show pipeline works
    from db.session import SessionLocal
    from db.models import Document

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "COMPLETE"
            db.commit()
    finally:
        db.close()
