from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # MinIO Configuration
    MINIO_URL: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False

    # Model Configuration
    MODEL_BUCKET: str = "models"
    DATA_BUCKET: str = "data"

    class Config:
        env_file = ".env.dev"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
