from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    FHIR_BASE_URL: str | None = None
    FHIR_AUTH_TOKEN: str | None = None
    DEBUG: bool = False
    DEFAULT_MODEL: str = "openrouter"
    JWT_SECRET: str = "ariston-ai-jwt-secret-change-in-production-2026"
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_WEBHOOK_SECRET: str | None = None

    # alias so anthropic_model.py (settings.anthropic_api_key) works
    @property
    def anthropic_api_key(self) -> str | None:
        return self.ANTHROPIC_API_KEY

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
