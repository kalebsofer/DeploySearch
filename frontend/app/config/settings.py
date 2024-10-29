from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BACKEND_URL: str = "http://backend:8051"
    TIMEOUT_SECONDS: int = 30

    class Config:
        env_file = ".env.dev"


def get_settings() -> Settings:
    return Settings()
