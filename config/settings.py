from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings can be overridden with DOCUSENSE_* environment variables or a .env file
    (prefix avoids clashing with unrelated variables such as an old DATABASE_URL)."""
    model_config = SettingsConfigDict(env_prefix="DOCUSENSE_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///docusense.db"
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20

    classifier_ckpt: str = "models/checkpoints/classifier_v1.pth"
    tagger_ckpt: str = "models/checkpoints/tagger_v1.pth"
    device: str = "cpu"

    tesseract_cmd: str = ""            # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe (blank = use PATH)
    poppler_path: str = ""             # folder containing pdftoppm (blank = use PATH)
    ocr_min_conf: int = 10
    ocr_psm: int = 6
    pdf_dpi: int = 200

    review_conf_threshold: float = 0.70
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


settings = Settings()
