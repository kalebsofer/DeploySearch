from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os


class Settings(BaseSettings):
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    MINIO_HOST: str
    MINIO_PORT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str

    ENVIRONMENT: str = "dev"

    model_config = dict(
        env_file=(
            ".env.dev" if os.getenv("ENVIRONMENT", "dev") == "dev" else ".env.prod"
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings():
    return Settings()
