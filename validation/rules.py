from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

class InvoiceValidator:
    REQUIRED = ["vendor_name", "invoice_date", "total_amount"]
    TOLERANCE = 1

    def validate(self, fields: dict, line_items: list) -> ValidationResult:
        errors, warnings = [], []

        missing = [f for f in self.REQUIRED if not fields.get(f)]
        if missing:
            warnings.append(f"Missing required fields: {', '.join(missing)}")

        if line_items and fields.get("subtotal"):
            computed = sum(item.get("amount", 0) for item in line_items)
            stated   = fields["subtotal"]
            if abs(computed - stated) > self.TOLERANCE:
                errors.append(f"Line items sum ({computed}) != stated subtotal ({stated})")

        if all(fields.get(k) for k in ["subtotal", "tax_amount", "total_amount"]):
            computed = fields["subtotal"] + fields["tax_amount"]
            stated   = fields["total_amount"]
            if abs(computed - stated) > self.TOLERANCE:
                errors.append(f"Subtotal + tax ({computed}) != stated total ({stated})")

        for key in ["subtotal", "tax_amount", "total_amount"]:
            val = fields.get(key, 0)
            if isinstance(val, (int, float)) and val < 0:
                errors.append(f"{key} cannot be negative: {val}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

def get_validator(doc_type: str):
    return InvoiceValidator()
