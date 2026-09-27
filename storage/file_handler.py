import os
import re
import uuid

from fastapi import HTTPException, UploadFile

from config.settings import settings
from pipeline.preprocess import ALLOWED_EXT


def save_upload(file: UploadFile) -> tuple:
    """Stream the upload to disk with an extension check and a size limit. -> (path, size_bytes)"""
    name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}")
    os.makedirs(settings.upload_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80]
    path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}_{safe}")
    limit, size = settings.max_file_size_mb * 1024 * 1024, 0
    with open(path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close(); os.remove(path)
                raise HTTPException(413, f"File larger than {settings.max_file_size_mb} MB")
            out.write(chunk)
    if size == 0:
        os.remove(path)
        raise HTTPException(400, "Empty file")
    return path, size
