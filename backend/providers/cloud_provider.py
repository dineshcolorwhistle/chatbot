"""
Cloud LLM Provider (OpenAI-Compatible)

Implements the LLMProvider interface for any OpenAI-compatible cloud API.
Works with: OpenAI, Groq, Together AI, Mistral, and others.

This provider is used when you need cloud-based LLM inference
(e.g., for better quality models or when no local GPU is available).

Requirements:
  - A valid API key from the cloud provider
  - Network access to the provider's API endpoint
"""

import logging
import httpx

from providers.base import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class CloudProvider(LLMProvider):
    """LLM provider for OpenAI-compatible cloud APIs.

    Attributes:
        base_url: The API base URL (e.g., https://api.openai.com/v1).
        api_key: Authentication API key.
        model: The model identifier to use.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ):
        """Initialize the cloud LLM provider.

        Args:
            api_key: API key for authentication. Required.
            base_url: API base URL. Default is OpenAI's endpoint.
            model: Model identifier. Default is gpt-4o-mini.
            timeout: HTTP request timeout in seconds.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError(
                "Cloud API key is required. Set CLOUD_API_KEY in .env file."
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response using the OpenAI-compatible chat completions API.

        Args:
            messages: Conversation history as LLMMessage list.
            temperature: Controls randomness (0.0 - 1.0).
            max_tokens: Maximum response tokens. None uses model default.

        Returns:
            LLMResponse with the generated content.

        Raises:
            ConnectionError: If the API endpoint is unreachable.
            RuntimeError: If the API returns an error.
        """
        self._validate_messages(messages)

        # Convert to OpenAI message format
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Build request payload
        payload: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(
                        "Cloud API error (status %d): %s",
                        response.status_code,
                        error_text,
                    )
                    raise RuntimeError(
                        f"Cloud API returned status {response.status_code}: "
                        f"{error_text}"
                    )

                data = response.json()

                # Extract response content
                choices = data.get("choices", [])
                if not choices:
                    logger.warning("Cloud API returned no choices.")
                    raise RuntimeError("Cloud API returned no choices.")

                content = choices[0].get("message", {}).get("content", "")
                model_used = data.get("model", self.model)

                if not content:
                    logger.warning("Cloud API returned an empty response.")
                    raise RuntimeError("Cloud API returned an empty response.")

                logger.info(
                    "Cloud response generated (model: %s, length: %d chars)",
                    model_used,
                    len(content),
                )

                return LLMResponse(
                    content=content,
                    model=model_used,
                    provider="cloud",
                )

        except httpx.ConnectError as e:
            logger.error("Cannot connect to cloud API at %s: %s", self.base_url, e)
            raise ConnectionError(
                f"Cannot connect to cloud API at {self.base_url}. "
                f"Check your internet connection and API endpoint."
            ) from e

        except httpx.TimeoutException as e:
            logger.error("Cloud API request timed out after %ss", self.timeout)
            raise RuntimeError(
                f"Cloud API request timed out after {self.timeout}s."
            ) from e

    async def health_check(self) -> bool:
        """Check if the cloud API is reachable and the key is valid.

        Sends a minimal request to verify connectivity and authentication.

        Returns:
            True if the API is reachable and key is valid.
        """
        try:
            test_messages = [
                LLMMessage(role="user", content="Hello")
            ]

            response = await self.generate(
                messages=test_messages,
                temperature=0.0,
                max_tokens=5,
            )

            logger.info(
                "Cloud API health check passed (model: %s)",
                response.model,
            )
            return True

        except Exception as e:
            logger.warning("Cloud API health check failed: %s", e)
            return False
