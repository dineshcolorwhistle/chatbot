import asyncio
import sys

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')

from services.mongo_store import session_store

async def main():
    sessions = await session_store.list_sessions()
    print(f"Total sessions: {len(sessions)}")
    
    for session_id in sessions:
        session = await session_store.get(session_id)
        if session:
            print(f"\n--- Session: {session_id} ---")
            print(session.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
