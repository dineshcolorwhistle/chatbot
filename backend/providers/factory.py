"""
LLM Provider Factory

Creates the appropriate LLM provider instance based on configuration.
This is the single entry point for obtaining an LLM provider — agents
and services should never instantiate providers directly.

Usage:
    from providers.factory import create_llm_provider
    provider = create_llm_provider()
    response = await provider.generate(messages)
"""

import logging

from config import llm_config
from providers.base import LLMProvider
from providers.ollama_provider import OllamaProvider
from providers.cloud_provider import CloudProvider

logger = logging.getLogger(__name__)


def create_llm_provider() -> LLMProvider:
    """Create and return the configured LLM provider.

    Reads the LLM_PROVIDER setting from config and instantiates
    the appropriate provider with its configuration.

    Returns:
        An instance of LLMProvider (either OllamaProvider or CloudProvider).

    Raises:
        ValueError: If the configured provider name is not recognized.
    """
    provider_name = llm_config.provider.lower()

    if provider_name == "ollama":
        logger.info(
            "Initializing Ollama provider (url: %s, model: %s)",
            llm_config.ollama_base_url,
            llm_config.ollama_model,
        )
        return OllamaProvider(
            base_url=llm_config.ollama_base_url,
            model=llm_config.ollama_model,
        )

    elif provider_name == "cloud":
        logger.info(
            "Initializing Cloud provider (url: %s, model: %s)",
            llm_config.cloud_base_url,
            llm_config.cloud_model,
        )
        return CloudProvider(
            api_key=llm_config.cloud_api_key,
            base_url=llm_config.cloud_base_url,
            model=llm_config.cloud_model,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported providers: 'ollama', 'cloud'. "
            f"Set LLM_PROVIDER in your .env file."
        )
