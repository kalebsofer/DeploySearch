from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    BACKEND_URL: str
    TIMEOUT_SECONDS: int = 30
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    ENVIRONMENT: str = "dev"

    model_config = dict(
        env_file=(
            ".env.dev" if os.getenv("ENVIRONMENT", "dev") == "dev" else ".env.prod"
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
