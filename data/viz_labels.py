"""Draw labelled word boxes on a page - used to eyeball the auto-labels (label-quality gate)."""
from PIL import Image, ImageDraw

from models.tagger.labels import LABELS

COLORS = {"OTHER": None, "VENDOR": (220, 20, 60), "DATE": (30, 144, 255), "ITEM_NAME": (34, 139, 34),
          "ITEM_QTY": (255, 140, 0), "ITEM_PRICE": (148, 0, 211), "SUBTOTAL": (0, 139, 139),
          "TAX": (139, 69, 19), "TOTAL": (255, 0, 0), "INVOICE_NO": (0, 0, 205)}


def draw_words(img: Image.Image, words: list, max_side: int = 1200) -> Image.Image:
    """words: dicts with box [x0,y0,x1,y1] in the pixel space of `img` and label (int or name)."""
    im = img.convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)))
    d = ImageDraw.Draw(im)
    for w in words:
        lab = w["label"] if isinstance(w["label"], str) else LABELS[int(w["label"])]
        col = COLORS.get(lab)
        if col:
            d.rectangle([v * s for v in w["box"]], outline=col, width=2)
            d.text((w["box"][0] * s, max(0, w["box"][1] * s - 10)), lab, fill=col)
    return im
