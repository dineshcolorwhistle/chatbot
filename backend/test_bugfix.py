"""
Bug Fix Verification Tests

Tests for two specific bugs:
  1. "Welcome to ColorWhistle!" greeting repeating in later responses
  2. "We noticed you haven't shared your name" showing when name was provided

These tests simulate a full conversation flow through the orchestrator
with a mock LLM provider.
"""

import asyncio
import json
import sys
import os
import io
import re

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure backend root is in path
sys.path.insert(0, os.path.dirname(__file__))

from models.schemas import Session, ConversationStage
from providers.base import LLMProvider, LLMMessage, LLMResponse
from services.conversation_agent import ConversationAgent, ConversationResult
from services.orchestrator import Orchestrator
from services.session_store import BaseSessionStore


# ============================================
# Mock Session Store
# ============================================

class MockSessionStore(BaseSessionStore):
    """In-memory mock session store for testing."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str):
        return self._sessions.get(session_id)

    async def create(self, session_id: str):
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    async def exists(self, session_id: str):
        return session_id in self._sessions

    async def get_or_create(self, session_id: str):
        if session_id in self._sessions:
            return self._sessions[session_id], False
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session, True

    async def save(self, session: Session):
        self._sessions[session.session_id] = session

    async def delete(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# ============================================
# Mock LLM Provider
# ============================================

class MockLLMProvider(LLMProvider):
    """Mock LLM that returns controlled responses.
    
    Simulates a small model that sometimes produces greetings
    and sometimes fails to extract data as JSON.
    """

    def __init__(self, force_greeting: bool = False):
        self.call_count = 0
        self.last_messages = None
        self.force_greeting = force_greeting

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages

        user_msg = messages[-1].content if len(messages) > 1 else ""
        system_msg = messages[0].content if messages else ""

        # For question-answering prompts (no JSON required)
        if "Just write a plain text answer" in system_msg or "Do NOT output JSON" in system_msg:
            if self.force_greeting:
                # Simulate the bug: LLM adds a greeting
                return LLMResponse(
                    content="Welcome to ColorWhistle! We offer comprehensive web development services including React, Python, and full-stack solutions.",
                    model="mock",
                    provider="mock",
                )
            return LLMResponse(
                content="We offer comprehensive web development services including React, Python, and full-stack solutions.",
                model="mock",
                provider="mock",
            )

        # For data-extraction prompts (JSON expected)
        if "my name is" in user_msg.lower() or "i'm " in user_msg.lower() or "i am " in user_msg.lower():
            # Extract name from the message
            name_match = re.search(
                r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})",
                user_msg, re.IGNORECASE,
            )
            name = name_match.group(1).strip() if name_match else "Unknown"
            # Clean name — stop at certain words
            name = re.split(r'\b(?:and|from|at|in|with)\b', name, flags=re.IGNORECASE)[0].strip()
            
            if self.force_greeting:
                response = json.dumps({
                    "response": f"Welcome to ColorWhistle! Nice to meet you, {name}! How can I help you today?",
                    "extracted_data": {"name": name},
                })
            else:
                response = json.dumps({
                    "response": f"Nice to meet you, {name}! How can I help you with your project today?",
                    "extracted_data": {"name": name},
                })
        elif "@" in user_msg:
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', user_msg)
            email = email_match.group(0) if email_match else ""
            response = json.dumps({
                "response": "Thanks for sharing your email! How can I assist you with your project?",
                "extracted_data": {"email": email} if email else {},
            })
        elif "website" in user_msg.lower() or "web" in user_msg.lower() or "app" in user_msg.lower():
            response = json.dumps({
                "response": "That sounds like a great project! Could you tell me more about the features you're looking for?",
                "extracted_data": {"project_type": "Web Application"},
            })
        else:
            if self.force_greeting:
                response = json.dumps({
                    "response": "Welcome to ColorWhistle! I'd be happy to help! Can you tell me more about your project?",
                    "extracted_data": {},
                })
            else:
                response = json.dumps({
                    "response": "I'd be happy to help! Can you tell me more about your project?",
                    "extracted_data": {},
                })

        return LLMResponse(content=response, model="mock", provider="mock")

    async def health_check(self) -> bool:
        return True


class FailingExtractionLLMProvider(LLMProvider):
    """Mock LLM that FAILS to extract data as JSON.
    
    Simulates the bug where the LLM returns plain text instead
    of JSON, causing data extraction to fail.
    """

    def __init__(self):
        self.call_count = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        user_msg = messages[-1].content if len(messages) > 1 else ""
        system_msg = messages[0].content if messages else ""

        # For question-answering prompts
        if "Just write a plain text answer" in system_msg or "Do NOT output JSON" in system_msg:
            return LLMResponse(
                content="We provide full-stack development services.",
                model="mock",
                provider="mock",
            )

        # DELIBERATELY return bad JSON / plain text to simulate extraction failure
        if "my name is" in user_msg.lower() or "i'm " in user_msg.lower():
            # Return plain text instead of JSON — this simulates the bug
            return LLMResponse(
                content="Nice to meet you! How can I help you with your project today?",
                model="mock",
                provider="mock",
            )
        else:
            # Return valid JSON for other messages
            return LLMResponse(
                content=json.dumps({
                    "response": "I'd be happy to help with your project!",
                    "extracted_data": {},
                }),
                model="mock",
                provider="mock",
            )

    async def health_check(self) -> bool:
        return True


# ============================================
# Test Functions
# ============================================

async def test_bug1_no_repeated_greeting():
    """Test Bug 1: Greeting should NOT repeat in later responses.
    
    The orchestrator sends "Welcome to ColorWhistle!" as the first
    message. The conversation agent should NEVER add another greeting.
    """
    print("\n--- Test Bug 1: No Repeated Greeting ---")

    provider = MockLLMProvider(force_greeting=False)
    store = MockSessionStore()
    orchestrator = Orchestrator(provider, store)

    session_id = "bug1-test-001"

    # Message 1: First user message
    response1 = await orchestrator.process_message(session_id, "Hello! I need a website.")
    print(f"  Response 1: {response1.reply[:100]}...")

    # Message 2
    response2 = await orchestrator.process_message(session_id, "I need an e-commerce platform")
    print(f"  Response 2: {response2.reply[:100]}...")

    # Message 3
    response3 = await orchestrator.process_message(session_id, "What services do you offer?")
    print(f"  Response 3: {response3.reply[:100]}...")

    # Check: No response after the 1st should contain "Welcome to ColorWhistle"
    welcome_pattern = re.compile(r"welcome to colorwhistle", re.IGNORECASE)

    assert not welcome_pattern.search(response2.reply), \
        f"Response 2 should NOT contain greeting. Got: {response2.reply}"
    assert not welcome_pattern.search(response3.reply), \
        f"Response 3 should NOT contain greeting. Got: {response3.reply}"

    print("  ✅ PASSED — No repeated greetings in responses 2 and 3")


async def test_bug1_greeting_stripped_from_llm():
    """Test Bug 1: Even if the LLM produces a greeting, it should be stripped."""
    print("\n--- Test Bug 1: Greeting Stripped from LLM Output ---")

    # This provider FORCES the LLM to produce greetings
    provider = MockLLMProvider(force_greeting=True)
    store = MockSessionStore()
    orchestrator = Orchestrator(provider, store)

    session_id = "bug1-test-002"

    # Message 1: First message — welcome is from orchestrator, not LLM
    response1 = await orchestrator.process_message(session_id, "Hello!")
    print(f"  Response 1: {response1.reply[:100]}...")

    # Message 2: LLM will try to add greeting, should be stripped
    response2 = await orchestrator.process_message(session_id, "I'm John and I need a website")
    print(f"  Response 2: {response2.reply[:100]}...")

    # Message 3: LLM will try to add greeting again, should be stripped
    response3 = await orchestrator.process_message(session_id, "What services do you offer?")
    print(f"  Response 3: {response3.reply[:100]}...")

    welcome_pattern = re.compile(r"welcome to colorwhistle", re.IGNORECASE)

    assert not welcome_pattern.search(response2.reply), \
        f"Response 2 should have greeting stripped. Got: {response2.reply}"
    assert not welcome_pattern.search(response3.reply), \
        f"Response 3 should have greeting stripped. Got: {response3.reply}"

    print("  ✅ PASSED — Greetings stripped from LLM output")


async def test_bug2_name_not_missed_with_good_extraction():
    """Test Bug 2: When name IS extracted properly, limit warning should NOT say it's missing."""
    print("\n--- Test Bug 2: Name Properly Extracted → No False Warning ---")

    provider = MockLLMProvider(force_greeting=False)
    store = MockSessionStore()
    orchestrator = Orchestrator(provider, store)

    session_id = "bug2-test-001"

    # Message 1: Provide name
    r1 = await orchestrator.process_message(session_id, "Hi, I'm John Doe")
    print(f"  Msg 1 reply: {r1.reply[:80]}...")
    print(f"  Msg 1 data: {r1.data_collected}")

    # Message 2
    r2 = await orchestrator.process_message(session_id, "I need a website for my business")
    print(f"  Msg 2 reply: {r2.reply[:80]}...")

    # Message 3
    r3 = await orchestrator.process_message(session_id, "My email is john@example.com")
    print(f"  Msg 3 reply: {r3.reply[:80]}...")
    print(f"  Msg 3 data: {r3.data_collected}")

    # Message 4 (penultimate)
    r4 = await orchestrator.process_message(session_id, "I want e-commerce features")
    print(f"  Msg 4 reply: {r4.reply[:80]}...")

    # Message 5 (limit reached)
    r5 = await orchestrator.process_message(session_id, "Also need payment integration")
    print(f"  Msg 5 reply: {r5.reply[:120]}...")
    print(f"  Msg 5 stage: {r5.stage}")

    # The limit warning should NOT say "haven't shared your name"
    assert "haven't shared your name" not in r5.reply, \
        f"Should NOT say name is missing when it was provided. Got: {r5.reply}"
    assert "haven't shared your email" not in r5.reply, \
        f"Should NOT say email is missing when it was provided. Got: {r5.reply}"

    print("  ✅ PASSED — No false 'haven't shared your name' warning")


async def test_bug2_retroactive_scan_catches_missed_name():
    """Test Bug 2: When LLM fails to extract name, retroactive scan catches it."""
    print("\n--- Test Bug 2: Retroactive Scan Catches Missed Name ---")

    # This provider deliberately fails to extract names from JSON
    provider = FailingExtractionLLMProvider()
    store = MockSessionStore()
    orchestrator = Orchestrator(provider, store)

    session_id = "bug2-test-002"

    # Message 1: Provide name — but LLM will fail to extract as JSON
    # The regex fallback in conversation_agent should still catch it
    r1 = await orchestrator.process_message(session_id, "Hi, I'm Dinesh Kumar")
    print(f"  Msg 1 reply: {r1.reply[:80]}...")
    print(f"  Msg 1 data: {r1.data_collected}")

    # Check if regex fallback caught the name
    session = await store.get(session_id)
    name_after_msg1 = session.collected_data.personal_info.name
    print(f"  Name after msg 1 (regex fallback): {name_after_msg1}")

    # Message 2
    r2 = await orchestrator.process_message(session_id, "I need a website")
    
    # Message 3
    r3 = await orchestrator.process_message(session_id, "My email is dinesh@test.com")

    # Message 4
    r4 = await orchestrator.process_message(session_id, "E-commerce platform")

    # Message 5 (limit reached)
    r5 = await orchestrator.process_message(session_id, "With payment gateway")
    print(f"  Msg 5 reply: {r5.reply[:150]}...")
    print(f"  Msg 5 stage: {r5.stage}")
    print(f"  Msg 5 data: {r5.data_collected}")

    # Even if initial extraction failed, retroactive scan should have caught the name
    session_final = await store.get(session_id)
    final_name = session_final.collected_data.personal_info.name
    print(f"  Final name in session: {final_name}")

    assert final_name is not None, \
        "Name should have been caught by regex fallback or retroactive scan"

    # The limit warning should NOT falsely claim name is missing
    assert "haven't shared your name" not in r5.reply, \
        f"Retroactive scan should have caught the name. Got: {r5.reply}"

    print("  ✅ PASSED — Retroactive scan caught the name missed by LLM extraction")


async def test_bug2_truly_missing_name_shows_warning():
    """Test: When name is ACTUALLY missing, the warning should still appear."""
    print("\n--- Test: Truly Missing Name Shows Correct Warning ---")

    provider = FailingExtractionLLMProvider()
    store = MockSessionStore()
    orchestrator = Orchestrator(provider, store)

    session_id = "bug2-test-003"

    # Message 1-5: Never provide name or email
    r1 = await orchestrator.process_message(session_id, "Hello")
    r2 = await orchestrator.process_message(session_id, "I need a website")
    r3 = await orchestrator.process_message(session_id, "E-commerce platform")
    r4 = await orchestrator.process_message(session_id, "With payment gateway")
    r5 = await orchestrator.process_message(session_id, "And shipping integration")

    print(f"  Msg 5 reply: {r5.reply[:150]}...")
    print(f"  Msg 5 stage: {r5.stage}")

    # Since name and email were never provided, warning should appear
    assert "name" in r5.reply.lower() and "email" in r5.reply.lower(), \
        f"Should mention missing name and email when truly missing. Got: {r5.reply}"

    print("  ✅ PASSED — Correct warning shown when name/email truly missing")


async def test_conversation_agent_no_greeting():
    """Test: ConversationAgent strips greeting from its replies."""
    print("\n--- Test: ConversationAgent Strips Greeting ---")

    provider = MockLLMProvider(force_greeting=True)
    agent = ConversationAgent(provider)
    session = Session(session_id="strip-test-001", stage=ConversationStage.CONVERSATION)

    # Add some history so it's not the first message
    session.add_message("assistant", "Welcome! How can I help?")
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there!")

    # Process a data message — LLM will return with greeting
    result = await agent.process_message(session, "I'm John and I need help")
    print(f"  Reply: {result.reply[:100]}...")

    welcome_pattern = re.compile(r"welcome to colorwhistle", re.IGNORECASE)
    assert not welcome_pattern.search(result.reply), \
        f"Greeting should be stripped. Got: {result.reply}"

    print("  ✅ PASSED — Greeting stripped from conversation agent reply")


async def test_conversation_agent_strips_question_greeting():
    """Test: ConversationAgent strips greeting from question answers."""
    print("\n--- Test: ConversationAgent Strips Greeting from Questions ---")

    provider = MockLLMProvider(force_greeting=True)
    agent = ConversationAgent(provider)
    session = Session(session_id="strip-test-002", stage=ConversationStage.CONVERSATION)
    session.add_message("assistant", "Welcome! How can I help?")

    # Process a question — LLM will return with greeting
    result = await agent.process_message(session, "What services do you offer?")
    print(f"  Reply: {result.reply[:100]}...")

    welcome_pattern = re.compile(r"welcome to colorwhistle", re.IGNORECASE)
    assert not welcome_pattern.search(result.reply), \
        f"Greeting should be stripped from question answer. Got: {result.reply}"

    print("  ✅ PASSED — Greeting stripped from question answer")


# ============================================
# Run All Tests
# ============================================

async def run_all_tests():
    """Run all bug fix verification tests."""
    print("=" * 60)
    print("  Bug Fix Verification Tests")
    print("=" * 60)

    tests = [
        test_bug1_no_repeated_greeting,
        test_bug1_greeting_stripped_from_llm,
        test_bug2_name_not_missed_with_good_extraction,
        test_bug2_retroactive_scan_catches_missed_name,
        test_bug2_truly_missing_name_shows_warning,
        test_conversation_agent_no_greeting,
        test_conversation_agent_strips_question_greeting,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
