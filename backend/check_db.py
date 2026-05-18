import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    uri = os.getenv("MONGODB_URI")
    print(f"URI: {uri[:30]}...")
    client = AsyncIOMotorClient(uri)
    db = client["chatbot_sessions"]
    collection = db["admins"]
    
    admins = await collection.find({}).to_list(100)
    print(f"Found {len(admins)} admins:")
    for admin in admins:
        print(f"- {admin.get('email')} : {admin.get('name')}")

if __name__ == "__main__":
    asyncio.run(main())
