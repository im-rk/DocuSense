from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Correction, Document, Extraction
from db.session import get_db

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    docs = db.scalars(select(Document)).all()
    by_status = Counter(d.status for d in docs)
    since = datetime.now(timezone.utc) - timedelta(days=13)
    per_day = Counter(d.created_at.date().isoformat() for d in docs if d.created_at.replace(tzinfo=timezone.utc) >= since)
    days = [(since + timedelta(days=i)).date().isoformat() for i in range(14)]

    exts = db.scalars(select(Extraction)).all()
    conf = defaultdict(list)
    for e in exts:
        for k, v in (e.confidences or {}).items():
            conf[k].append(v)
    n_corr = db.scalar(select(func.count(Correction.id))) or 0
    corrected_docs = db.scalar(select(func.count(func.distinct(Correction.extraction_id)))) or 0
    done = sum(by_status.get(s, 0) for s in ("COMPLETE", "NEEDS_REVIEW"))
    ms = [e.inference_ms for e in exts if e.inference_ms]
    return {
        "total_documents": len(docs), "by_status": dict(by_status),
        "per_day": [{"date": d, "count": per_day.get(d, 0)} for d in days],
        "needs_review_rate": round(by_status.get("NEEDS_REVIEW", 0) / done, 3) if done else 0.0,
        "correction_rate": round(corrected_docs / done, 3) if done else 0.0, "total_corrections": n_corr,
        "avg_inference_ms": int(sum(ms) / len(ms)) if ms else None,
        "avg_confidence": {k: round(sum(v) / len(v), 3) for k, v in sorted(conf.items())},
    }
