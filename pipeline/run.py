"""End-to-end: file -> structured JSON.   CLI:  python -m pipeline.run path\\to\\receipt.jpg"""
import json
import sys
import time

from config.settings import settings
from models.classifier.predict import classify
from models.tagger.postprocess import extract
from models.tagger.predict import tag_words
from pipeline.loader import get_models
from pipeline.ocr import get_words
from pipeline.preprocess import load_pages, to_gray
from validation.rules import validate


def process(path: str, models=None) -> dict:
    t0 = time.time()
    m = models or get_models()
    img = load_pages(path, dpi=settings.pdf_dpi, poppler_path=settings.poppler_path)[0]
    gray = to_gray(img)
    H, W = gray.shape

    doc_type, type_conf, type_probs = classify(m.classifier, m.cls_meta, img)          # Model 1
    ocr_words = get_words(img, settings.ocr_min_conf, settings.ocr_psm, settings.tesseract_cmd)
    tagged = tag_words(m.tagger, m.vocab, gray, ocr_words, settings.device) if ocr_words else []   # Model 2
    ex = extract(tagged)
    val = validate(ex["fields"], ex["line_items"], ex["field_confidences"], settings.review_conf_threshold,
                   doc_type, type_conf, settings.type_warn_conf)
    if not ocr_words:
        val["errors"].append("no text found on the page (blank or unreadable image)")
        val["passed"], val["needs_review"] = False, True

    return {
        "document_type": doc_type, "type_confidence": round(type_conf, 4),
        "type_probabilities": {k: round(v, 4) for k, v in type_probs.items()},
        "fields": ex["fields"], "line_items": ex["line_items"],
        "field_confidences": {k: round(v, 4) for k, v in ex["field_confidences"].items()},
        "field_sources": ex["field_sources"], "validation": val,
        "words": [{"text": w["text"], "box": [round(float(v), 1) for v in w["box"]], "label": w["label_name"],
                   "prob": round(w["prob"], 3), "line": w["line"]} for w in tagged],
        "image_size": {"width": W, "height": H}, "ocr_engine": "tesseract",
        "model_versions": m.versions, "inference_ms": int((time.time() - t0) * 1000),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pipeline.run <image-or-pdf>")
    out = process(sys.argv[1])
    out["words"] = f"<{len(out['words'])} words omitted>"
    print(json.dumps(out, indent=2, ensure_ascii=False))
