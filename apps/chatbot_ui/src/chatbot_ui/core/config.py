from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    API_URL: str = "http://api:8000"  # default for Docker Compose; override via env
    model_config = SettingsConfigDict(env_file=".env")


config = Config()
