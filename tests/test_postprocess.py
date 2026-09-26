from models.tagger.labels import LABEL_ID as L
from models.tagger.postprocess import extract, normalise_date, parse_amount
from validation.rules import validate


def test_parse_amount():
    assert parse_amount("RM 68.90") == 68.9
    assert parse_amount("68,90") == 68.9
    assert parse_amount("60.000") == 60000.0          # Indonesian thousands
    assert parse_amount("1,234") == 1234.0
    assert parse_amount("1,234.50") == 1234.5
    assert parse_amount("1.234,50") == 1234.5
    assert parse_amount("1.234.567") == 1234567.0
    assert parse_amount("0.125") == 0.125
    assert parse_amount("₹ 76,700") == 76700.0
    assert parse_amount("(5.00)") == -5.0
    assert parse_amount("abc") is None and parse_amount("") is None and parse_amount(None) is None


def test_normalise_date():
    assert normalise_date("25/12/2018") == "2018-12-25"
    assert normalise_date("2018-03-15") == "2018-03-15"
    assert normalise_date("12/25/2018") == "2018-12-25"        # 25 can only be a day
    assert normalise_date("05/03/19") == "2019-03-05"          # day-first, 2-digit year
    assert normalise_date("25 DEC 2018") == "2018-12-25"
    assert normalise_date("Dec 25, 2018") == "2018-12-25"
    assert normalise_date("DATE: 15-Mar-2024 10:30") == "2024-03-15"
    assert normalise_date("31/02/2020") is None
    assert normalise_date("hello") is None


def W(text, label, line, y=0, prob=0.9):
    return {"text": text, "label": label, "line": line, "box": [0, y, 10, y + 10], "prob": prob}


def test_extract_receipt():
    words = [
        W("ACME", L["VENDOR"], 0, 0), W("STORE", L["VENDOR"], 0, 0), W("SDN", L["VENDOR"], 0, 0),
        W("Date:", 0, 1, 20), W("25/12/2018", L["DATE"], 1, 20),
        W("Nasi", L["ITEM_NAME"], 2, 40), W("Goreng", L["ITEM_NAME"], 2, 40),
        W("2", L["ITEM_QTY"], 2, 40), W("10.00", L["ITEM_PRICE"], 2, 40),
        W("Teh", L["ITEM_NAME"], 3, 60), W("Tarik", L["ITEM_NAME"], 3, 60),      # wrapped: price on next line
        W("5.00", L["ITEM_PRICE"], 4, 80),
        W("Subtotal", 0, 5, 100), W("15.00", L["SUBTOTAL"], 5, 100),
        W("Tax", 0, 6, 120), W("0.90", L["TAX"], 6, 120),
        W("Total", 0, 7, 140), W("15.90", L["TOTAL"], 7, 140),
        W("Cash", 0, 8, 160), W("20.00", 0, 8, 160),
    ]
    r = extract(words)
    f = r["fields"]
    assert f["vendor"] == "ACME STORE SDN" and f["date"] == "2018-12-25"
    assert (f["subtotal"], f["tax"], f["total"]) == (15.0, 0.9, 15.9)
    assert r["line_items"] == [{"name": "Nasi Goreng", "qty": 2, "price": 10.0},
                               {"name": "Teh Tarik", "qty": None, "price": 5.0}]
    v = validate(f, r["line_items"], r["field_confidences"])
    assert v["checks"] == {"subtotal_plus_tax_equals_total": "PASS", "items_sum_equals_subtotal": "PASS"}
    assert v["passed"] and not v["needs_review"]


def test_extract_fallbacks_and_validation():
    words = [W("Invoice", 0, 0), W("No:", 0, 0), W("INV-2024-0847", 0, 0),
             W("12/03/2024", 0, 1), W("TOTAL", 0, 2), W("100.00", L["TOTAL"], 2, prob=0.4),
             W("Sub", 0, 3), W("50.00", L["SUBTOTAL"], 3), W("tax", 0, 4), W("10.00", L["TAX"], 4)]
    r = extract(words)
    assert r["fields"]["invoice_number"] == "INV-2024-0847" and r["field_sources"]["invoice_number"] == "regex"
    assert r["fields"]["date"] == "2024-03-12" and r["field_sources"]["date"] == "regex"
    v = validate(r["fields"], r["line_items"], r["field_confidences"])
    assert v["checks"]["subtotal_plus_tax_equals_total"] == "FAIL" and not v["passed"] and v["needs_review"]
    assert "total" in v["low_confidence_fields"] and "vendor" in v["missing_fields"]


def test_total_prefers_confident_then_lowest():
    words = [W("Total", 0, 0), W("10.00", L["TOTAL"], 0, y=0, prob=0.9),
             W("Total", 0, 1), W("12.00", L["TOTAL"], 1, y=50, prob=0.9)]
    assert extract(words)["fields"]["total"] == 12.0
