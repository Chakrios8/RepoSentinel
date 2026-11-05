from functools import lru_cache
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """RepoSentinel Application Configuration."""
    
    APP_NAME: str = "RepoSentinel"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Google Gemini API configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # ML Triage configuration
    TRIAGE_THRESHOLD: float = 0.50
    MODEL_PATH: str = str(ROOT_DIR / "backend" / "ml" / "ast_triage_model.joblib")
    FEATURE_COLUMNS_PATH: str = str(ROOT_DIR / "backend" / "ml" / "feature_columns.json")

    # RAG Vector Store configuration
    CHROMA_PERSIST_DIR: str = str(ROOT_DIR / "backend" / "rag" / "chroma_db")
    SEED_RULES_PATH: str = str(ROOT_DIR / "backend" / "rag" / "seed_rules.json")

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def is_api_key_configured(self) -> bool:
        """Check if a real Gemini API key is configured."""
        key = self.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            return False
        clean = key.strip()
        return bool(clean and clean != "your_gemini_api_key_here" and len(clean) > 8)


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
