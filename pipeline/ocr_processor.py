import pytesseract
from PIL import Image

def get_words_and_boxes(image: Image.Image) -> list[dict]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    W, H = image.size
    words = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text and conf > 30:  # filter low-confidence OCR
            x, y, w, h = (data['left'][i], data['top'][i],
                          data['width'][i], data['height'][i])
            words.append({
                "text": text,
                "x1": x, "y1": y,
                "x2": x + w, "y2": y + h,
                "x1_norm": x / W, "y1_norm": y / H,
                "x2_norm": (x + w) / W, "y2_norm": (y + h) / H,
            })
    return words
