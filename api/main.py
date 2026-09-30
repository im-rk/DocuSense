import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analytics, documents
from config.settings import settings
from db.session import init_db
from pipeline.loader import models_status

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    init_db()
    st = models_status()
    if st["missing"]:
        logging.getLogger("docusense").warning(
            "Model checkpoints missing: %s - uploads will FAIL until you copy the trained .pth files there "
            "(see KAGGLE_TRAINING.md).", st["missing"])
    yield


app = FastAPI(title="DocuSense", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(documents.router)
app.include_router(analytics.router)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "models": models_status()}
