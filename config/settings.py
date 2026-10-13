from pydantic_settings import BaseSettings, DotEnvSettingsSource, SettingsConfigDict


class _PrefixedDotEnv(DotEnvSettingsSource):
    """Only DOCUSENSE_* keys of a .env file count. Without this, an old .env line such as DATABASE_URL=postgres://...
    silently overrides our settings (pydantic-settings also matches un-prefixed names inside .env files)."""

    def _load_env_vars(self):
        pre = self.env_prefix.lower()
        return {k: v for k, v in super()._load_env_vars().items() if k.lower().startswith(pre)}


class Settings(BaseSettings):
    """All settings can be overridden with DOCUSENSE_* environment variables or a .env file
    (prefix avoids clashing with unrelated variables such as an old DATABASE_URL)."""
    model_config = SettingsConfigDict(env_prefix="DOCUSENSE_", env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        dotenv = _PrefixedDotEnv(settings_cls, env_file=cls.model_config.get("env_file"), env_file_encoding="utf-8")
        return init_settings, env_settings, dotenv, file_secret_settings

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
    type_warn_conf: float = 0.50       # Model 1 confidence above which a non-receipt page type triggers a warning
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


settings = Settings()
