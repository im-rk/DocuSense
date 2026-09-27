import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now():
    return datetime.now(timezone.utc)


def _id():
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    original_name: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str] = mapped_column(String(1000))
    mime_type: Mapped[str] = mapped_column(String(100), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)   # PENDING|PROCESSING|COMPLETE|NEEDS_REVIEW|FAILED
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extraction: Mapped["Extraction | None"] = relationship(back_populates="document", uselist=False,
                                                           cascade="all, delete-orphan")


class Extraction(Base):
    __tablename__ = "extractions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), unique=True, index=True)
    document_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    type_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    fields: Mapped[dict] = mapped_column(JSON, default=dict)
    line_items: Mapped[list] = mapped_column(JSON, default=list)
    confidences: Mapped[dict] = mapped_column(JSON, default=dict)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    words: Mapped[list] = mapped_column(JSON, default=list)
    image_size: Mapped[dict] = mapped_column(JSON, default=dict)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)          # type_probabilities, field_sources, versions, ocr engine
    inference_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    document: Mapped[Document] = relationship(back_populates="extraction")
    corrections: Mapped[list["Correction"]] = relationship(back_populates="extraction", cascade="all, delete-orphan")


class Correction(Base):
    """Human edits: audit trail and future training data."""
    __tablename__ = "corrections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    extraction_id: Mapped[str] = mapped_column(ForeignKey("extractions.id"), index=True)
    field: Mapped[str] = mapped_column(String(100))
    original: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected: Mapped[str] = mapped_column(Text)
    corrected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    extraction: Mapped[Extraction] = relationship(back_populates="corrections")
