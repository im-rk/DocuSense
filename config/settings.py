from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6380"
    jwt_secret: str
    jwt_expiry_hours: int = 24
    upload_dir: str = "uploads"
    classifier_checkpoint: str = "models/checkpoints/classifier_v1.pth"
    extractor_checkpoint: str = "models/checkpoints/extractor_v1.pth"
    device: str = "cpu"
    classifier_confidence_threshold: float = 0.70

    class Config:
        env_file = ".env"

settings = Settings()
