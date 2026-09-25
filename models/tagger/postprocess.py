"""Tagged words -> structured fields. Deterministic rules on top of Model 2 output (no learning here).

Input words: list of dicts in reading order with keys
    text, box [x0,y0,x1,y1], line (int), label (int id or name), prob (float 0-1)
"""
import re
from datetime import date

from models.tagger.labels import LABEL_ID as L

# ---------------------------------------------------------------- amounts
_CUR = re.compile(r"(?i)(us\$|rm|rp|idr|usd|inr|myr|sgd|eur|gbp|\$|₹|€|£|/-)")


def parse_amount(text):
    """'RM 68.90' -> 68.9 ; '60.000' -> 60000.0 ; '1,234.50' -> 1234.5 ; '1.234,50' -> 1234.5 ; junk -> None."""
    if text is None:
        return None
    t = str(text).strip()
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")"))
    t = re.sub(r"[^\d.,]", "", _CUR.sub("", t)).strip(".,")
    if not re.search(r"\d", t):
        return None
    last_dot, last_com = t.rfind("."), t.rfind(",")
    if last_dot >= 0 and last_com >= 0:                       # both present: the later one is the decimal mark
        dec = max(last_dot, last_com)
        s = re.sub(r"[.,]", "", t[:dec]) + "." + t[dec + 1:]
    elif last_dot >= 0 or last_com >= 0:                      # only one kind of separator
        sep = "." if last_dot >= 0 else ","
        pos = t.rfind(sep)
        head, frac = t[:pos], t[pos + 1:]
        if t.count(sep) > 1:                                  # 1.234.567 -> thousands
            s = t.replace(sep, "")
        elif len(frac) == 3 and head not in ("", "0"):        # 60.000 / 1,234 -> thousands
            s = head + frac
        else:                                                 # 68.90 / 5.5 / 0.125 -> decimal
            s = (head or "0") + "." + frac
    else:
        s = t
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


# ---------------------------------------------------------------- dates
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_D_NUM = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})")
_D_ISO = re.compile(r"(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})")
_D_DMY = re.compile(r"(\d{1,2})\s*[-/. ]?\s*([A-Za-z]{3,9})\s*[-/.,]?\s*(\d{2,4})")
_D_MDY = re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2})\s*,?\s*(\d{2,4})")


def _mk(y, m, d):
    y = int(y)
    y = y + 2000 if y < 100 else y
    try:
        return date(y, int(m), int(d)).isoformat()
    except ValueError:
        return None


def normalise_date(text):
    """Any common receipt date -> 'YYYY-MM-DD' (day-first unless the numbers force month-first) or None."""
    if not text:
        return None
    t = str(text)
    m = _D_ISO.search(t)
    if m:
        return _mk(m.group(1), m.group(2), m.group(3))
    m = _D_NUM.search(t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        day, month = (b, a) if (b > 12 and a <= 12) else (a, b)      # day-first unless numbers force month-first
        return _mk(y, month, day)
    m = _D_DMY.search(t)
    if m and m.group(2)[:3].lower() in _MONTHS:
        return _mk(m.group(3), _MONTHS[m.group(2)[:3].lower()], m.group(1))
    m = _D_MDY.search(t)
    if m and m.group(1)[:3].lower() in _MONTHS:
        return _mk(m.group(3), _MONTHS[m.group(1)[:3].lower()], m.group(2))
    return None


# ---------------------------------------------------------------- extraction
_INV_RE = re.compile(r"(?i)(?:invoice|inv|receipt|bill|doc)\s*(?:no|number|num|#)?\s*[:.#\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})")
_ONE_LINE_AMOUNT = {L["DATE"], L["SUBTOTAL"], L["TAX"], L["TOTAL"], L["ITEM_PRICE"], L["ITEM_QTY"]}


def _lab(w):
    v = w["label"]
    return L[v] if isinstance(v, str) else int(v)


def _spans(words):
    spans, cur = [], None
    for w in words:
        lab = _lab(w)
        if lab == 0:
            cur = None
            continue
        line = w.get("line", 0)
        if cur and cur["label"] == lab and (
                line == cur["line"] or (lab not in _ONE_LINE_AMOUNT and lab != L["ITEM_NAME"] and line - cur["line"] <= 1)):
            cur["words"].append(w)
            cur["line"] = line
        else:
            cur = {"label": lab, "words": [w], "line": line}
            spans.append(cur)
    for s in spans:
        s["text"] = " ".join(w["text"] for w in s["words"])
        s["prob"] = sum(w.get("prob", 1.0) for w in s["words"]) / len(s["words"])
        s["y"] = s["words"][0]["box"][1]
    return spans


def _best(spans, label, parse=None, lowest=False):
    cands = []
    for s in spans:
        if s["label"] != label:
            continue
        val = parse(s["text"]) if parse else s["text"].strip()
        if val is None or val == "":
            continue
        cands.append((round(s["prob"], 3), s["y"] if lowest else -s["y"], val, s["prob"]))
    if not cands:
        return None, None
    c = max(cands, key=lambda x: (x[0], x[1]))
    return c[2], c[3]


def _lines_text(words):
    lines = {}
    for w in words:
        lines.setdefault(w.get("line", 0), []).append(w["text"])
    return [" ".join(v) for _, v in sorted(lines.items())]


def _line_items(words):
    by_line = {}
    for w in words:
        lab = _lab(w)
        if lab in (L["ITEM_NAME"], L["ITEM_QTY"], L["ITEM_PRICE"]):
            by_line.setdefault(w.get("line", 0), []).append((lab, w))
    items, pending = [], None
    for _, ws in sorted(by_line.items()):
        name = " ".join(w["text"] for lab, w in ws if lab == L["ITEM_NAME"]).strip()
        prices = [parse_amount(w["text"]) for lab, w in ws if lab == L["ITEM_PRICE"]]
        prices = [p for p in prices if p is not None]
        qtys = [parse_amount(w["text"]) for lab, w in ws if lab == L["ITEM_QTY"]]
        qtys = [q for q in qtys if q is not None]
        qty = qtys[0] if qtys else None
        if qty is not None and float(qty).is_integer():
            qty = int(qty)
        if name and prices:
            items.append({"name": (pending + " " + name).strip() if pending else name,
                          "qty": qty, "price": prices[-1]})
            pending = None
        elif name:                                   # wrapped name: price comes on a later line
            pending = (pending + " " + name).strip() if pending else name
        elif prices and pending:
            items.append({"name": pending, "qty": qty, "price": prices[-1]})
            pending = None
    return items


def extract(words):
    """-> {'fields':{vendor,date,invoice_number,subtotal,tax,total}, 'line_items':[...],
           'field_confidences':{...}, 'field_sources':{field: 'model'|'regex'}}"""
    spans = _spans(words)
    fields, conf, src = {}, {}, {}

    # vendor: the confident (longest-weighted) VENDOR span, top-most on ties
    vs = [s for s in spans if s["label"] == L["VENDOR"] and s["text"].strip()]
    if vs:
        v = max(vs, key=lambda s: (round(sum(w.get("prob", 1.0) for w in s["words"]), 2), -s["y"]))
        fields["vendor"], conf["vendor"], src["vendor"] = v["text"].strip(), v["prob"], "model"

    val, p = _best(spans, L["DATE"], normalise_date)
    if val:
        fields["date"], conf["date"], src["date"] = val, p, "model"
    else:                                            # regex fallback over the OCR text
        for line in _lines_text(words):
            d = normalise_date(line)
            if d:
                fields["date"], src["date"] = d, "regex"
                break

    for name, lab, lowest in (("subtotal", L["SUBTOTAL"], True), ("tax", L["TAX"], True), ("total", L["TOTAL"], True)):
        val, p = _best(spans, lab, parse_amount, lowest=lowest)
        if val is not None:
            fields[name], conf[name], src[name] = val, p, "model"

    val, p = _best(spans, L["INVOICE_NO"])
    if val:
        fields["invoice_number"], conf["invoice_number"], src["invoice_number"] = val.strip(" :#."), p, "model"
    else:
        for line in _lines_text(words):
            m = _INV_RE.search(line)
            if m and re.search(r"\d", m.group(1)):
                fields["invoice_number"], src["invoice_number"] = m.group(1), "regex"
                break

    return {"fields": fields, "line_items": _line_items(words),
            "field_confidences": conf, "field_sources": src}
