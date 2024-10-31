from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    MINIO_HOST: str
    MINIO_PORT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()
