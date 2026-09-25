"""Arithmetic + completeness checks on extracted fields."""

REQUIRED = ("vendor", "date", "total")


def validate(fields: dict, line_items: list, confidences: dict, low_conf: float = 0.70) -> dict:
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

    return {"passed": not errors, "needs_review": bool(errors or low or missing),
            "errors": errors, "warnings": warnings, "checks": checks,
            "missing_fields": missing, "low_confidence_fields": low}
