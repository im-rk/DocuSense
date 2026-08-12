import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from db.session import Base

class User(Base):
    __tablename__ = "users"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email         = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255))
    full_name     = Column(String(255))
    created_at    = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_name   = Column(String(500), nullable=False)
    file_path       = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type       = Column(String(100), nullable=False)
    page_count      = Column(Integer)
    uploaded_by     = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    uploaded_at     = Column(DateTime, default=datetime.utcnow)
    status          = Column(String(20), default="PENDING")
    celery_task_id  = Column(String(200))
    processing_started_at  = Column(DateTime)
    processing_finished_at = Column(DateTime)
    error_message   = Column(Text)

class Extraction(Base):
    __tablename__ = "extractions"
    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id       = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    document_type     = Column(String(50))
    type_confidence   = Column(Float)
    raw_extraction    = Column(JSONB, default=dict)
    field_confidences = Column(JSONB, default=dict)
    line_items        = Column(JSONB, default=list)
    validation_passed = Column(Boolean)
    validation_errors = Column(JSONB, default=list)
    validation_warnings = Column(JSONB, default=list)
    inference_ms      = Column(Integer)
    extracted_at      = Column(DateTime, default=datetime.utcnow)

class Correction(Base):
    __tablename__ = "corrections"
    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extraction_id   = Column(UUID(as_uuid=True), ForeignKey("extractions.id"), nullable=False)
    corrected_by    = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    field_name      = Column(String(100), nullable=False)
    original_value  = Column(Text)
    corrected_value = Column(Text, nullable=False)
    corrected_at    = Column(DateTime, default=datetime.utcnow)
