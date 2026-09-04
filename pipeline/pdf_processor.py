from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images(file_path: str, dpi: int = 200) -> list[Image.Image]:
    if file_path.lower().endswith('.pdf'):
        return convert_from_path(file_path, dpi=dpi)
    else:
        return [Image.open(file_path).convert('RGB')]
