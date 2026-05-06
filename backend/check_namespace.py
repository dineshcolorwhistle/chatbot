import asyncio
from config import mongo_config
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient(mongo_config.uri)
    db = client[mongo_config.db_name]
    coll = db[mongo_config.collection_name]
    
    docs = await coll.find().sort('created_at', -1).limit(5).to_list(5)
    for d in docs:
        print(f"Session: {d.get('session_id')}, Namespace: {d.get('namespace')}, Created: {d.get('created_at')}")

if __name__ == "__main__":
    asyncio.run(check())
