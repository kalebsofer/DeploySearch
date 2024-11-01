from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    MINIO_URL: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False

    ENVIRONMENT: str = "dev"

    model_config = dict(
        env_file=(
            ".env.dev"
            if os.getenv("ENVIRONMENT", "dev") == "development"
            else ".env.prod"
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
