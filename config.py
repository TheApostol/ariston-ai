from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Ariston AI"
    debug: bool = False

    # AI model endpoints
    default_model: str = "gpt-4o"
    openai_api_key: str
    anthropic_api_key: str
    GEMINI_API_KEY: str
    OPENROUTER_API_KEY: str

    # Future FHIR config
    fhir_base_url: str = ""
    fhir_auth_token: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
