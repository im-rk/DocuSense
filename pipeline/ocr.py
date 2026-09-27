"""OCR front-end: page image -> words + boxes.

NOTE (project rule): Tesseract ships a pretrained recogniser, so this is the ONE external, non-from-scratch component.
It is isolated here so it can be swapped for our own recogniser later. Models 1 and 2 never use pretrained weights.
"""
from PIL import Image


def get_words(img: Image.Image, min_conf: int = 30, psm: int = 6, tesseract_cmd: str = "") -> list:
    """-> [{text, box:[x0,y0,x1,y1] in the ORIGINAL image pixels, conf}]"""
    import pytesseract
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    im = img.convert("L")
    long_side = max(im.size)
    s = min(2.5, 1600 / long_side) if long_side < 1600 else 1.0        # small photos OCR better when upscaled
    if s > 1.05:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    else:
        s = 1.0
    try:
        d = pytesseract.image_to_data(im, config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError("Tesseract is not installed or not on PATH. Install it and/or set DOCUSENSE_TESSERACT_CMD.") from e
    words = []
    for i, text in enumerate(d["text"]):
        text = str(text).strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not text or conf < min_conf:
            continue
        l, t, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append({"text": text, "box": [l / s, t / s, (l + w) / s, (t + h) / s], "conf": conf})
    return words
