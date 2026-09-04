import os, shutil, time
from uuid import UUID
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from io import BytesIO
from sqlalchemy.orm import Session
from db.session import get_db
from db.models import Document, Extraction, Correction
from api.dependencies import get_current_user
from workers.tasks import process_document
from config.settings import settings

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Validate file type
    allowed = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Unsupported file type")

    # Save file to disk
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = f"{settings.upload_dir}/{int(time.time())}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(file_path)

    # Create document record
    doc = Document(
        original_name=file.filename,
        file_path=file_path,
        file_size_bytes=file_size,
        mime_type=file.content_type,
        uploaded_by=current_user.id,
        status="PENDING"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Queue Celery task
    task = process_document.delay(str(doc.id), file_path)
    doc.celery_task_id = task.id
    db.commit()

    return {
        "document_id": str(doc.id),
        "status": "PENDING",
        "message": "Document queued for processing"
    }

@router.get("/{document_id}/status")
def get_status(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    return {"document_id": str(doc.id), "status": doc.status,
            "error": doc.error_message}

@router.get("/{document_id}/result")
def get_result(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    extraction = db.query(Extraction)\
        .filter(Extraction.document_id == document_id).first()
    if not extraction:
        raise HTTPException(404, "Extraction not found or processing incomplete")
    return {
        "document_type": extraction.document_type,
        "type_confidence": extraction.type_confidence,
        "fields": extraction.raw_extraction,
        "field_confidences": extraction.field_confidences,
        "line_items": extraction.line_items,
        "validation_passed": extraction.validation_passed,
        "validation_errors": extraction.validation_errors,
    }

@router.put("/{document_id}/fields")
def save_correction(
    document_id: UUID,
    body: dict,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    extraction = db.query(Extraction)\
        .filter(Extraction.document_id == document_id).first()
    if not extraction:
        raise HTTPException(404, "Extraction not found")

    for field_name, corrected_value in body.get("corrections", {}).items():
        original = extraction.raw_extraction.get(field_name)
        correction = Correction(
            extraction_id=extraction.id,
            corrected_by=current_user.id,
            field_name=field_name,
            original_value=str(original) if original else None,
            corrected_value=str(corrected_value)
        )
        db.add(correction)
        extraction.raw_extraction[field_name] = corrected_value

    db.commit()
    return {"message": "Corrections saved"}

@router.get("/{document_id}/export")
def export_document(
    document_id: UUID,
    format: str = "json",
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    extraction = db.query(Extraction)\
        .filter(Extraction.document_id == document_id).first()
    if not extraction:
        raise HTTPException(404, "Not found")

    if format == "json":
        return extraction.raw_extraction

    flat = {**extraction.raw_extraction, "document_type": extraction.document_type}

    if format == "csv":
        df = pd.DataFrame([flat])
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=extraction.csv"})

    if format == "excel":
        df = pd.DataFrame([flat])
        buf = BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=extraction.xlsx"})
