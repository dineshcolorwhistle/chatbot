import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from config import mongo_config

logger = logging.getLogger(__name__)

class MongoAdminStore:
    """MongoDB admin storage."""

    def __init__(self) -> None:
        if not mongo_config.uri:
            logger.warning("MONGODB_URI is not set. MongoAdminStore will fail connecting.")
            
        self._client = AsyncIOMotorClient(mongo_config.uri)
        self._db = self._client[mongo_config.db_name]
        self._collection = self._db["admins"]

    async def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        document = await self._collection.find_one({"email": email})
        if document:
            document["_id"] = str(document["_id"])
        return document

    async def create_admin(self, name: str, email: str, password_hash: str) -> Dict[str, Any]:
        existing = await self.get_by_email(email)
        if existing:
            raise ValueError(f"Admin already exists with email: {email}")

        admin_doc = {
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await self._collection.insert_one(admin_doc)
        admin_doc["_id"] = str(result.inserted_id)
        return admin_doc

    async def update_password(self, email: str, password_hash: str) -> bool:
        result = await self._collection.update_one(
            {"email": email},
            {"$set": {"password_hash": password_hash, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    async def list_admins(self) -> List[Dict[str, Any]]:
        cursor = self._collection.find({})
        admins = []
        for doc in await cursor.to_list(length=1000):
            doc["_id"] = str(doc["_id"])
            # Do not return hashes
            doc.pop("password_hash", None)
            admins.append(doc)
        return admins

admin_store = MongoAdminStore()
