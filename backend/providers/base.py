"""
Abstract Base LLM Provider

Defines the interface that all LLM providers must implement.
This abstraction allows swapping between Ollama local, cloud APIs,
or any future provider without changing agent code.

Usage:
    All agents interact with LLMs exclusively through this interface.
    They never call Ollama or cloud APIs directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """A single message in a conversation.

    Attributes:
        role: One of "system", "user", or "assistant".
        content: The text content of the message.
    """

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The generated text content.
        model: The model identifier that generated the response.
        provider: The provider name (e.g., "ollama", "cloud").
    """

    content: str
    model: str
    provider: str


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    All LLM integrations must implement this interface to ensure
    consistent behavior across different providers.

    Methods:
        generate: Generate a response from a list of messages.
        health_check: Verify the provider is available and responsive.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history as a list of LLMMessage objects.
                      Must include at least one message. Typically starts
                      with a system message followed by user/assistant turns.
            temperature: Controls randomness. 0.0 = deterministic, 1.0 = creative.
                         Default is 0.7 for a balance of consistency and variety.
            max_tokens: Maximum tokens in the response. None = provider default.

        Returns:
            LLMResponse containing the generated text, model name, and provider.

        Raises:
            ConnectionError: If the provider is unreachable.
            ValueError: If the messages list is empty or malformed.
            RuntimeError: If the LLM returns an error response.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is available and responsive.

        Returns:
            True if the provider is reachable and ready,
            False otherwise.
        """
        ...

    def _validate_messages(self, messages: list[LLMMessage]) -> None:
        """Validate the messages list before sending to the provider.

        Args:
            messages: The messages to validate.

        Raises:
            ValueError: If messages are empty or contain invalid roles.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        valid_roles = {"system", "user", "assistant"}
        for msg in messages:
            if msg.role not in valid_roles:
                raise ValueError(
                    f"Invalid message role: '{msg.role}'. "
                    f"Must be one of: {valid_roles}"
                )
