"""
In-Memory Session Store

Concrete implementation of BaseSessionStore using a Python dictionary.
Sessions persist only while the application is running.

This is the MVP storage backend. For production, swap to:
  - RedisSessionStore (for distributed caching)
  - PostgresSessionStore (for durable persistence)

Thread Safety:
  Since FastAPI uses asyncio (single-threaded event loop),
  dictionary operations are inherently safe. No locks needed.
"""

import logging
from datetime import datetime

from models.schemas import Session
from services.session_store import BaseSessionStore

logger = logging.getLogger(__name__)


class InMemorySessionStore(BaseSessionStore):
    """In-memory session storage using a Python dictionary.

    Attributes:
        _sessions: Internal dictionary mapping session_id to Session objects.
    """

    def __init__(self) -> None:
        """Initialize an empty session store."""
        self._sessions: dict[str, Session] = {}
        logger.info("In-memory session store initialized")

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Args:
            session_id: The unique session identifier.

        Returns:
            The Session object if found, None otherwise.
        """
        session = self._sessions.get(session_id)
        if session:
            logger.debug("Session found: %s (stage: %s)", session_id, session.stage)
        else:
            logger.debug("Session not found: %s", session_id)
        return session

    async def create(self, session_id: str) -> Session:
        """Create a new session.

        Args:
            session_id: The unique session identifier.

        Returns:
            The newly created Session object.

        Raises:
            ValueError: If the session ID already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session already exists: {session_id}")

        session = Session(session_id=session_id)
        self._sessions[session_id] = session

        logger.info(
            "New session created: %s (total active: %d)",
            session_id,
            len(self._sessions),
        )
        return session

    async def save(self, session: Session) -> None:
        """Save/update a session.

        Args:
            session: The Session object to persist.
        """
        session.updated_at = datetime.utcnow()
        self._sessions[session.session_id] = session

        logger.debug(
            "Session saved: %s (stage: %s)",
            session.session_id,
            session.stage,
        )

    async def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted, False if not found.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(
                "Session deleted: %s (total active: %d)",
                session_id,
                len(self._sessions),
            )
            return True

        logger.debug("Cannot delete — session not found: %s", session_id)
        return False

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: The session to check.

        Returns:
            True if the session exists.
        """
        return session_id in self._sessions

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)

    async def list_sessions(self) -> list[str]:
        """List all active session IDs.

        Returns:
            List of session ID strings. Useful for debugging.
        """
        return list(self._sessions.keys())


# Singleton instance — import this directly
session_store = InMemorySessionStore()
