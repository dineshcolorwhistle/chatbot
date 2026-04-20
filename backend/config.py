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
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)


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
    max_user_messages: int = 5
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
    ])
    admin_emails: list[str] = field(default_factory=lambda: [
        "admin@colorwhistle.com"
    ])
    
    # SMTP Settings for sending real emails
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@colorwhistle.com"


@dataclass(frozen=True)
class PineconeConfig:
    """Pinecone vector database configuration for RAG."""

    api_key: str = ""
    index_name: str = "colorwhistle-kb"
    cloud: str = "aws"
    region: str = "us-east-1"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "nomic-embed-text"
    dimension: int = 768
    ollama_base_url: str = "http://localhost:11434"


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

    admin_emails_str = os.getenv(
        "ADMIN_EMAILS", "admin@colorwhistle.com"
    )
    admin_emails = [email.strip() for email in admin_emails_str.split(",")]

    return AppConfig(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        debug=os.getenv("APP_DEBUG", "true").lower() == "true",
        max_user_messages=int(os.getenv("MAX_USER_MESSAGES", "5")),
        cors_origins=cors_origins,
        admin_emails=admin_emails,
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        smtp_from=os.getenv("SMTP_FROM", "noreply@colorwhistle.com"),
    )


def _load_pinecone_config() -> PineconeConfig:
    """Load Pinecone configuration from environment variables."""
    return PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY", ""),
        index_name=os.getenv("PINECONE_INDEX_NAME", "colorwhistle-kb"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
    )


def _load_embedding_config() -> EmbeddingConfig:
    """Load embedding model configuration from environment variables."""
    return EmbeddingConfig(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


# Singleton instances — import these directly
llm_config = _load_llm_config()
app_config = _load_app_config()
pinecone_config = _load_pinecone_config()
embedding_config = _load_embedding_config()
