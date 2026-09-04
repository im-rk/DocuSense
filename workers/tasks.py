import time
from workers.celery_app import celery_app
from db.session import SessionLocal
from db.models import Document, Extraction
from pipeline.pdf_processor import pdf_to_images
from pipeline.ocr_processor import get_words_and_boxes
from pipeline.model_loader import get_classifier, get_extractor
from models.classifier.predict import classify_document
from models.extractor.predict import extract_words
from models.extractor.postprocess import words_to_fields
from validation.rules import get_validator
from pipeline.confidence import compute_field_confidences
import datetime

@celery_app.task(bind=True, max_retries=3)
def process_document(self, document_id: str, file_path: str):
    db = SessionLocal()
    start_ms = int(time.time() * 1000)

    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "PROCESSING"
        doc.processing_started_at = datetime.datetime.utcnow()
        db.commit()

        images = pdf_to_images(file_path)
        page_one = images[0]

        classifier = get_classifier()
        doc_type, type_conf = classify_document(classifier, page_one)

        words = get_words_and_boxes(page_one)

        extractor = get_extractor()
        classified_words = extract_words(extractor, page_one, words)

        fields, line_items = words_to_fields(classified_words)

        validator = get_validator(doc_type)
        validation = validator.validate(fields, line_items)

        confidences = compute_field_confidences(classified_words)

        inference_ms = int(time.time() * 1000) - start_ms
        extraction = Extraction(
            document_id=document_id,
            document_type=doc_type,
            type_confidence=type_conf,
            raw_extraction=fields,
            field_confidences=confidences,
            line_items=line_items,
            validation_passed=validation.is_valid,
            validation_errors=validation.errors,
            validation_warnings=validation.warnings,
            inference_ms=inference_ms
        )
        db.add(extraction)
        doc.status = "NEEDS_REVIEW" if not validation.is_valid else "COMPLETE"
        doc.processing_finished_at = datetime.datetime.utcnow()
        db.commit()

    except Exception as exc:
        doc.status = "FAILED"
        doc.error_message = str(exc)
        db.commit()
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
    finally:
        db.close()
