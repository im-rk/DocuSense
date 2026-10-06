from fastapi import FastAPI
from api.routers import auth, documents

app = FastAPI(title="DocuSense API")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}
