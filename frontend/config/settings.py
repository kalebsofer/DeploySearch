from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os


class Settings(BaseSettings):
    BACKEND_URL: str = "http://localhost:8051"
    API_VERSION: str = "v1"

    ENV: str = "development"

    TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 3

    class Config:
        env_file = os.getenv(
            "ENV_FILE", Path(__file__).parent.parent.parent / "env" / ".env.dev"
        )
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
