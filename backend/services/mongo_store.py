"""
MongoDB Session Store

Concrete implementation of BaseSessionStore using MongoDB (via Motor).
Provides durable, scalable session persistence.
"""

import logging
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient

from models.schemas import Session
from services.session_store import BaseSessionStore
from config import mongo_config

logger = logging.getLogger(__name__)


class MongoSessionStore(BaseSessionStore):
    """MongoDB session storage.

    Attributes:
        client: Motor Async IO MongoDB Client.
        db: MongoDB database instance.
        collection: MongoDB collection instance.
    """

    def __init__(self) -> None:
        """Initialize the MongoDB connection and collection."""
        if not mongo_config.uri:
            logger.warning("MONGODB_URI is not set. MongoSessionStore will fail connecting.")
            
        self._client = AsyncIOMotorClient(mongo_config.uri)
        self._db = self._client[mongo_config.db_name]
        self._collection = self._db[mongo_config.collection_name]
        
        logger.info(
            "MongoDB session store initialized. Database: %s, Collection: %s",
            mongo_config.db_name,
            mongo_config.collection_name,
        )

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        document = await self._collection.find_one({"session_id": session_id})
        
        if document:
            # Remove the '_id' added by MongoDB before passing to Pydantic
            document.pop("_id", None)
            session = Session(**document)
            logger.debug("MongoDB: Session found: %s (stage: %s)", session_id, session.stage)
            return session
            
        logger.debug("MongoDB: Session not found: %s", session_id)
        return None

    async def create(self, session_id: str) -> Session:
        """Create a new session."""
        existing = await self.exists(session_id)
        if existing:
            raise ValueError(f"Session already exists: {session_id}")

        session = Session(session_id=session_id)
        document = session.model_dump()
        
        await self._collection.insert_one(document)

        logger.info(
            "MongoDB: New session created: %s",
            session_id,
        )
        return session

    async def save(self, session: Session) -> None:
        """Save/update a session."""
        session.updated_at = datetime.utcnow()
        document = session.model_dump()
        
        # Replace the existing document with the updated session state
        result = await self._collection.replace_one(
            {"session_id": session.session_id},
            document,
            upsert=True
        )

        logger.debug(
            "MongoDB: Session saved: %s (stage: %s, modified count: %d)",
            session.session_id,
            session.stage,
            result.modified_count,
        )

    async def delete(self, session_id: str) -> bool:
        """Delete a session by ID."""
        result = await self._collection.delete_one({"session_id": session_id})
        
        if result.deleted_count > 0:
            logger.info("MongoDB: Session deleted: %s", session_id)
            return True

        logger.debug("MongoDB: Cannot delete — session not found: %s", session_id)
        return False

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        count = await self._collection.count_documents({"session_id": session_id}, limit=1)
        return count > 0

    async def list_sessions(self) -> list[str]:
        """List all active session IDs. Mostly for debugging."""
        cursor = self._collection.find({}, {"session_id": 1, "_id": 0})
        session_ids = [doc["session_id"] for doc in await cursor.to_list(length=1000)]
        return session_ids


# Singleton instance — import this directly
session_store = MongoSessionStore()
