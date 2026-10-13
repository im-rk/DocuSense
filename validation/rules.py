"""Arithmetic + completeness checks on extracted fields."""

REQUIRED = ("vendor", "date", "total")
NOT_FINANCIAL = ("FORM", "LETTER", "OTHER")      # Model 1 classes that are not a receipt or invoice


def validate(fields: dict, line_items: list, confidences: dict, low_conf: float = 0.70,
             doc_type: str = None, type_conf: float = None, type_warn_conf: float = 0.50) -> dict:
    errors, warnings, checks = [], [], {}
    sub, tax, tot = fields.get("subtotal"), fields.get("tax"), fields.get("total")

    if sub is not None and tax is not None and tot is not None:
        ok = abs(sub + tax - tot) <= max(0.02, 0.01 * abs(tot))
        checks["subtotal_plus_tax_equals_total"] = "PASS" if ok else "FAIL"
        if not ok:
            errors.append(f"subtotal + tax ({sub + tax:.2f}) != total ({tot:.2f})")
    else:
        checks["subtotal_plus_tax_equals_total"] = "SKIP"

    prices = [i["price"] for i in line_items if i.get("price") is not None]
    if sub is not None and prices:
        ok = abs(sum(prices) - sub) <= max(0.02, 0.02 * abs(sub))
        checks["items_sum_equals_subtotal"] = "PASS" if ok else "FAIL"
        if not ok:                                   # discounts / service charge can cause this -> warning only
            warnings.append(f"sum of items ({sum(prices):.2f}) != subtotal ({sub:.2f})")
    else:
        checks["items_sum_equals_subtotal"] = "SKIP"

    missing = [f for f in REQUIRED if fields.get(f) in (None, "")]
    if "total" in missing:
        errors.append("total amount not found")
    warnings += [f"missing field: {f}" for f in missing if f != "total"]

    low = sorted(f for f, c in confidences.items() if c is not None and c < low_conf)
    if low:
        warnings.append("low confidence: " + ", ".join(low))

    # Model 2 only knows receipts and invoices, so on a letter or form it still returns plausible-looking fields.
    # Model 1 is the only part that can notice; it warns (never blocks) because it is only ~90 % accurate.
    wrong_type = doc_type in NOT_FINANCIAL and (type_conf or 0) >= type_warn_conf
    if wrong_type:
        warnings.insert(0, f"this page was classified as '{doc_type.lower()}' ({type_conf:.0%} sure), not a receipt or invoice - "
                           "the extracted fields may be meaningless")

    return {"passed": not errors, "needs_review": bool(errors or low or missing or wrong_type),
            "errors": errors, "warnings": warnings, "checks": checks,
            "missing_fields": missing, "low_confidence_fields": low}
