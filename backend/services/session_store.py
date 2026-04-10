"""
Abstract Session Store

Defines the interface for session persistence.
Currently implemented as in-memory storage, but designed
for easy migration to Redis, PostgreSQL, or any other backend.

Usage:
    from services.session_store import session_store
    session = await session_store.get("user-001")
"""

from abc import ABC, abstractmethod

from models.schemas import Session


class BaseSessionStore(ABC):
    """Abstract base class for session storage backends.

    All session persistence implementations must follow this interface.
    This ensures the orchestrator and agents are decoupled from the
    storage mechanism.
    """

    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Args:
            session_id: The unique session identifier.

        Returns:
            The Session object if found, None otherwise.
        """
        ...

    @abstractmethod
    async def create(self, session_id: str) -> Session:
        """Create a new session with the given ID.

        Args:
            session_id: The unique session identifier.

        Returns:
            The newly created Session object.

        Raises:
            ValueError: If a session with this ID already exists.
        """
        ...

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Save/update an existing session.

        Args:
            session: The Session object to persist.
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The session to delete.

        Returns:
            True if the session was found and deleted,
            False if no session existed with that ID.
        """
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: The session to check.

        Returns:
            True if the session exists.
        """
        ...

    async def get_or_create(self, session_id: str) -> tuple[Session, bool]:
        """Get an existing session or create a new one.

        Convenience method used by the orchestrator to ensure
        a session always exists before processing a message.

        Args:
            session_id: The session identifier.

        Returns:
            A tuple of (session, is_new) where is_new indicates
            whether the session was just created.
        """
        existing = await self.get(session_id)
        if existing:
            return existing, False
        new_session = await self.create(session_id)
        return new_session, True
