from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = "Ariston AI LifeScience OS"
    version: str = "1.0.0-phase1-audit"
    debug: bool = False

    # AI model endpoints (Production Hardened)
    default_model: str = Field(default="gpt-4o", pattern="^(gpt|claude|gemini|openrouter).*")
    
    # Mandate API Keys for clinical pathways
    openai_api_key: str = Field(default="", min_length=0)
    anthropic_api_key: str = Field(default="", min_length=0)
    GEMINI_API_KEY: str = Field(default="", min_length=0)
    OPENROUTER_API_KEY: str = Field(default="", min_length=0)

    # Compliance Settings
    audit_db_path: str = "memory.db"
    enable_clinical_grounding: bool = True
    
    # Future FHIR config
    fhir_base_url: str = ""
    fhir_auth_token: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()
