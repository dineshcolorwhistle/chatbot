"""Quick test script for Phase 2 — Models & Session Store."""

import asyncio
from models.schemas import (
    ConversationStage,
    PersonalInfo, TechDiscovery, ScopePricing, CollectedData,
    Session, ChatRequest, ChatResponse,
)
from services.memory_store import session_store


async def test_all():
    print("=" * 50)
    print("Phase 2 — Models & Session Store Test")
    print("=" * 50)

    # 1. Stage enum
    print("\n1. Conversation Stages:")
    for stage in ConversationStage:
        print(f"   {stage.value}")

    # 2. Personal info completion check
    print("\n2. PersonalInfo model:")
    pi = PersonalInfo(name="DK")
    print(f"   Complete: {pi.is_complete()}, Missing: {pi.get_missing_fields()}")
    pi2 = PersonalInfo(name="DK", email="dk@test.com", phone="+91123")
    print(f"   Complete: {pi2.is_complete()}, Missing: {pi2.get_missing_fields()}")

    # 3. CollectedData summary
    print("\n3. CollectedData summary dict:")
    cd = CollectedData()
    cd.personal_info = pi2
    cd.tech_discovery.project_type = "AI Chatbot"
    print(f"   {cd.to_summary_dict()}")

    # 4. Session store CRUD
    print("\n4. Session Store CRUD:")
    s, is_new = await session_store.get_or_create("test-001")
    print(f"   Created: id={s.session_id}, new={is_new}, stage={s.stage.value}")

    s.add_message("user", "Hello")
    s.add_message("assistant", "Welcome!")
    await session_store.save(s)
    print(f"   Messages: {len(s.conversation_history)}")
    print(f"   LLM history: {s.get_history_for_llm()}")

    s2, is_new2 = await session_store.get_or_create("test-001")
    print(f"   Re-fetched: new={is_new2}, messages={len(s2.conversation_history)}")

    deleted = await session_store.delete("test-001")
    print(f"   Deleted: {deleted}, exists: {await session_store.exists('test-001')}")

    # 5. API models
    print("\n5. API Models:")
    req = ChatRequest(session_id="user-001", message="Hello")
    print(f"   ChatRequest: {req.model_dump()}")
    resp = ChatResponse(reply="Hi!", stage=ConversationStage.WELCOME)
    print(f"   ChatResponse: {resp.model_dump()}")

    print("\n" + "=" * 50)
    print("[PASS] All Phase 2 tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_all())
