"""Rule-based baseline (regex + keywords). Lower bound for the ablation table - no learning involved."""
import re

import numpy as np

from models.tagger.labels import LABEL_ID as L
from models.tagger.postprocess import normalise_date, parse_amount

_AMT = re.compile(r"^\D{0,3}\d[\d.,]*[.,]\d{2}$")


def rule_tag(rec: dict) -> np.ndarray:
    words = rec["words"]
    pred = np.zeros(len(words), dtype=np.int64)
    lines = {}
    for i, w in enumerate(words):
        lines.setdefault(w["line"], []).append(i)
    first_line = min(lines) if lines else 0
    for ln, idx in lines.items():
        text = " ".join(words[i]["text"] for i in idx).upper()
        amts = [i for i in idx if _AMT.match(words[i]["text"]) and parse_amount(words[i]["text"]) is not None]
        if ln == first_line:
            for i in idx:
                pred[i] = L["VENDOR"]
        for i in idx:
            if normalise_date(words[i]["text"]) and re.search(r"\d", words[i]["text"]) and len(words[i]["text"]) >= 6:
                pred[i] = L["DATE"]
        if not amts:
            continue
        last = amts[-1]
        if "TOTAL" in text and "SUB" in text:
            pred[last] = L["SUBTOTAL"]
        elif "TOTAL" in text:
            pred[last] = L["TOTAL"]
        elif re.search(r"\b(TAX|GST|SST|VAT|PPN)\b", text):
            pred[last] = L["TAX"]
        elif not re.search(r"\b(CASH|CHANGE|ROUND|PAID|CARD|AMOUNT|TENDER)\b", text):
            pred[last] = L["ITEM_PRICE"]
            for i in idx:
                if i < last and re.search(r"[A-Za-z]{2,}", words[i]["text"]):
                    pred[i] = L["ITEM_NAME"]
    return pred


def baseline_preds(records) -> list:
    """Same shape as evaluate.predict_records output."""
    return [(r, rule_tag(r), np.ones(len(r["words"]))) for r in records]
