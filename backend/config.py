"""
Application Configuration

Loads settings from .env file and provides typed configuration
for the entire backend application.

All configuration is centralized here — no other module reads
environment variables directly.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from the backend directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "ollama"  # "ollama" | "cloud"

    # Ollama local settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Cloud (OpenAI-compatible) settings
    cloud_api_key: str = ""
    cloud_base_url: str = "https://api.openai.com/v1"
    cloud_model: str = "gpt-4o-mini"


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
    ])


def _load_llm_config() -> LLMConfig:
    """Load LLM configuration from environment variables."""
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        cloud_api_key=os.getenv("CLOUD_API_KEY", ""),
        cloud_base_url=os.getenv("CLOUD_BASE_URL", "https://api.openai.com/v1"),
        cloud_model=os.getenv("CLOUD_MODEL", "gpt-4o-mini"),
    )


def _load_app_config() -> AppConfig:
    """Load application configuration from environment variables."""
    cors_origins_str = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    )
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

    return AppConfig(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        debug=os.getenv("APP_DEBUG", "true").lower() == "true",
        cors_origins=cors_origins,
    )


# Singleton instances — import these directly
llm_config = _load_llm_config()
app_config = _load_app_config()
