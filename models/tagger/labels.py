"""Label schema for Model 2 (word tagger) and class names for Model 1 (page classifier)."""

LABELS = ["OTHER", "VENDOR", "DATE", "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE",
          "SUBTOTAL", "TAX", "TOTAL", "INVOICE_NO"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
N_CLS = len(LABELS)

DOC_CLASSES = ["RECEIPT", "INVOICE", "FORM", "LETTER", "OTHER"]

# Classes a dataset never annotates. A word labelled OTHER in that dataset may really belong to one of
# them, so the partial-label loss lets the model put probability mass there (see loss.py).
_UNANNOTATED = {
    "cord":  ["VENDOR", "DATE", "INVOICE_NO"],
    "sroie": ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "SUBTOTAL", "TAX", "INVOICE_NO"],
    "synth": [],
}


def unannot_mask(source: str) -> list:
    """bool list of length N_CLS: True = this source never annotates that class."""
    names = _UNANNOTATED.get(source, [])
    return [LABELS[i] in names for i in range(N_CLS)]


def annotated_classes(source: str) -> list:
    """class ids (excluding OTHER) that the source annotates."""
    m = unannot_mask(source)
    return [i for i in range(1, N_CLS) if not m[i]]
