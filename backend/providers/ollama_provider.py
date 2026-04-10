"""
Ollama LLM Provider

Implements the LLMProvider interface for Ollama, supporting:
  - Local Ollama instance (http://localhost:11434)
  - Remote Ollama-compatible endpoints

Uses Ollama's native REST API (/api/chat) for optimal local performance.

Requirements:
  - Ollama must be installed and running locally for local usage
  - At least one model must be pulled (e.g., `ollama pull llama3.2`)
"""

import logging
import httpx

from providers.base import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider for Ollama (local and remote instances).

    Attributes:
        base_url: The Ollama server base URL.
        model: The model identifier to use (e.g., "llama3.2").
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 120.0,
    ):
        """Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL. Default is localhost.
            model: Model name to use. Must be pulled in Ollama first.
            timeout: HTTP request timeout in seconds. LLMs can be slow,
                     so default is 120s.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response using the Ollama /api/chat endpoint.

        Args:
            messages: Conversation history as LLMMessage list.
            temperature: Controls randomness (0.0 - 1.0).
            max_tokens: Maximum response tokens. None uses model default.

        Returns:
            LLMResponse with the generated content.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError: If the API returns an error.
        """
        self._validate_messages(messages)

        # Convert to Ollama's message format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Build request payload
        payload: dict = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,  # We want the full response at once
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(
                        "Ollama API error (status %d): %s",
                        response.status_code,
                        error_text,
                    )
                    raise RuntimeError(
                        f"Ollama API returned status {response.status_code}: "
                        f"{error_text}"
                    )

                data = response.json()

                # Extract the assistant's reply
                content = data.get("message", {}).get("content", "")
                model_used = data.get("model", self.model)

                if not content:
                    logger.warning("Ollama returned an empty response.")
                    raise RuntimeError("Ollama returned an empty response.")

                logger.info(
                    "Ollama response generated (model: %s, length: %d chars)",
                    model_used,
                    len(content),
                )

                return LLMResponse(
                    content=content,
                    model=model_used,
                    provider="ollama",
                )

        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self.base_url, e)
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Start it with: ollama serve"
            ) from e

        except httpx.TimeoutException as e:
            logger.error("Ollama request timed out after %ss", self.timeout)
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                f"The model may be loading or the request is too complex."
            ) from e

    async def health_check(self) -> bool:
        """Check if Ollama is running and the model is available.

        Verifies:
          1. Ollama server is reachable
          2. The configured model is available locally

        Returns:
            True if Ollama is healthy and model is ready.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check server is running
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code != 200:
                    logger.warning("Ollama health check failed: status %d", response.status_code)
                    return False

                # Check if configured model is available
                data = response.json()
                available_models = [
                    model.get("name", "").split(":")[0]
                    for model in data.get("models", [])
                ]

                model_base_name = self.model.split(":")[0]

                if model_base_name not in available_models:
                    logger.warning(
                        "Model '%s' not found in Ollama. Available: %s. "
                        "Pull it with: ollama pull %s",
                        self.model,
                        available_models,
                        self.model,
                    )
                    return False

                logger.info(
                    "Ollama health check passed (model: %s)",
                    self.model,
                )
                return True

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning("Ollama health check failed: %s", e)
            return False
