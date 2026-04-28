import asyncio
import sys
from backend.services.mongo_store import session_store

async def main():
    sessions = await session_store.list_sessions()
    print(f"Total sessions: {len(sessions)}")
    
    for session_id in sessions:
        session = await session_store.get(session_id)
        print(f"\n--- Session: {session_id} ---")
        if session and session.history:
            messages = session.history.messages
            print(f"Message Count: {len(messages)}")
            for msg in messages:
                print(f"[{msg.role}]: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
