from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = "Ariston AI"
    debug: bool = False

    # AI model endpoints
    default_model: str = "gpt-4o"
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API Key")
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API Key")

    # Future FHIR config
    fhir_base_url: str = ""
    fhir_auth_token: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
