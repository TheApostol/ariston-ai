from pathlib import Path
from dotenv import load_dotenv

# Load .env before pydantic-settings reads it — absolute path so it works
# regardless of working directory
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENROUTER_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    FHIR_BASE_URL: str | None = None
    FHIR_AUTH_TOKEN: str | None = None
    DEBUG: bool = False
    DEFAULT_MODEL: str = "anthropic"
    JWT_SECRET: str = "ariston-ai-jwt-secret-change-in-production-2026"
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_WEBHOOK_SECRET: str | None = None
    GOOGLE_PROJECT_ID: str | None = None
    ANTHROPIC_ADMIN_KEY: str | None = None

    @property
    def anthropic_api_key(self) -> str | None:
        return self.ANTHROPIC_API_KEY

    @property
    def openai_api_key(self) -> str | None:
        return self.OPENAI_API_KEY


settings = Settings()
