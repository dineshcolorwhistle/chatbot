"""
Test Suite for Agent Phases (3, 4, 5)

Tests the Conversation Agent, Summarization Agent, and Email Agent
using a mock LLM provider to verify agent logic independently
of actual LLM connectivity.
"""

import asyncio
import json
import sys
import os
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure backend root is in path
sys.path.insert(0, os.path.dirname(__file__))

from models.schemas import Session, ConversationStage
from providers.base import LLMProvider, LLMMessage, LLMResponse
from services.conversation_agent import ConversationAgent, ConversationResult
from services.summarization_agent import SummarizationAgent
from services.email_agent import EmailAgent, EmailResult


# ============================================
# Mock LLM Provider
# ============================================

class MockLLMProvider(LLMProvider):
    """Mock LLM that returns predefined responses for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_messages = None

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages

        # Determine context from system prompt
        system_msg = messages[0].content if messages else ""
        # Also check user message for additional context
        user_msg = messages[-1].content if len(messages) > 1 else ""

        # Return appropriate mock response based on context
        # Order matters — more specific matches first
        if "Welcome" in system_msg or "WELCOME" in system_msg:
            response = json.dumps({
                "response": "Welcome! I'm excited to help you scope out your project. Could you start by telling me your name?",
                "extracted_data": {},
                "stage_complete": False
            })
        elif "Personal Information" in system_msg or "PERSONAL_INFO" in system_msg:
            response = json.dumps({
                "response": "Thanks, John! Could you share your email address?",
                "extracted_data": {"name": "John Doe"},
                "stage_complete": False
            })
        elif "Technical Discovery" in system_msg or "TECH_DISCOVERY" in system_msg:
            response = json.dumps({
                "response": "That sounds like a great project! What key features would you need?",
                "extracted_data": {"project_type": "E-commerce Web App"},
                "stage_complete": False
            })
        elif "Scope & Pricing" in system_msg or "SCOPE_PRICING" in system_msg:
            response = json.dumps({
                "response": "Great, that gives me a clear picture. Let me prepare a summary for you.",
                "extracted_data": {"budget": "$10K-$15K", "timeline": "3 months"},
                "stage_complete": True
            })
        elif "project analyst" in system_msg.lower():
            # Summarization Agent — identified by its unique role description
            response = (
                "Lead Summary\n\n"
                "Client Information:\n"
                "- Name: John Doe\n"
                "- Email: john@example.com\n"
                "- Phone: +1-555-0123\n\n"
                "Project Overview:\n"
                "- Type: E-commerce Web App\n"
                "- Features: Product catalog, cart, checkout\n"
                "- Budget: $10K-$15K\n"
                "- Timeline: 3 months\n\n"
                "Priority Level: High"
            )
        elif "thank-you email" in system_msg.lower() or "thank you email" in system_msg.lower():
            # Email Agent — user thank-you email
            response = (
                "Dear John,\n\n"
                "Thank you for discussing your project with us at ColorWhistle.\n\n"
                "We've noted your requirements for an E-commerce Web App and our team "
                "will review them promptly. Expect to hear from us within 1-2 business days.\n\n"
                "Best regards,\nThe ColorWhistle Team"
            )
        elif "lead notification" in system_msg.lower() or "internal" in system_msg.lower():
            # Email Agent — admin notification email
            response = (
                "New Lead Notification\n\n"
                "Client: John Doe (john@example.com)\n"
                "Project: E-commerce Web App\n"
                "Budget: $10K-$15K | Timeline: 3 months\n\n"
                "Please follow up within 24 hours."
            )
        else:
            response = json.dumps({
                "response": "I'd be happy to help! Can you tell me more?",
                "extracted_data": {},
                "stage_complete": False
            })

        return LLMResponse(content=response, model="mock", provider="mock")

    async def health_check(self) -> bool:
        return True


# ============================================
# Test Functions
# ============================================

async def test_conversation_agent_welcome():
    """Test: Conversation Agent handles welcome stage."""
    print("\n--- Test: Conversation Agent — Welcome Stage ---")

    provider = MockLLMProvider()
    agent = ConversationAgent(provider)
    session = Session(session_id="test-001", stage=ConversationStage.WELCOME)

    result = await agent.process_message(session, "Hello!")

    assert isinstance(result, ConversationResult), "Should return ConversationResult"
    assert result.reply, "Should have a reply"
    assert isinstance(result.extracted_data, dict), "Should have extracted_data dict"
    assert result.should_advance is True, "Welcome stage should advance on any message"

    print(f"  Reply: {result.reply[:80]}...")
    print(f"  Extracted: {result.extracted_data}")
    print(f"  Should advance: {result.should_advance}")
    print("  ✅ PASSED")


async def test_conversation_agent_personal_info():
    """Test: Conversation Agent extracts personal data."""
    print("\n--- Test: Conversation Agent — Personal Info Stage ---")

    provider = MockLLMProvider()
    agent = ConversationAgent(provider)
    session = Session(
        session_id="test-002",
        stage=ConversationStage.PERSONAL_INFO,
    )

    result = await agent.process_message(session, "My name is John Doe")

    assert isinstance(result, ConversationResult)
    assert result.reply, "Should have a reply"
    assert "name" in result.extracted_data, "Should extract name"
    assert result.extracted_data["name"] == "John Doe"

    print(f"  Reply: {result.reply[:80]}...")
    print(f"  Extracted: {result.extracted_data}")
    print(f"  Should advance: {result.should_advance}")
    print("  ✅ PASSED")


async def test_conversation_agent_scope_complete():
    """Test: Conversation Agent detects stage completion."""
    print("\n--- Test: Conversation Agent — Scope Stage (Complete) ---")

    provider = MockLLMProvider()
    agent = ConversationAgent(provider)
    session = Session(
        session_id="test-003",
        stage=ConversationStage.SCOPE_PRICING,
    )

    result = await agent.process_message(session, "Budget is $10K-15K, timeline 3 months")

    assert isinstance(result, ConversationResult)
    assert result.should_advance is True, "Should advance when data is complete"

    print(f"  Reply: {result.reply[:80]}...")
    print(f"  Extracted: {result.extracted_data}")
    print(f"  Should advance: {result.should_advance}")
    print("  ✅ PASSED")


async def test_conversation_agent_with_context():
    """Test: System prompt includes already-collected data."""
    print("\n--- Test: Conversation Agent — Context Awareness ---")

    provider = MockLLMProvider()
    agent = ConversationAgent(provider)
    session = Session(
        session_id="test-004",
        stage=ConversationStage.PERSONAL_INFO,
    )
    # Pre-fill some data
    session.collected_data.personal_info.name = "John Doe"

    result = await agent.process_message(session, "My email is john@example.com")

    # Check that the system prompt included the already-collected name
    system_prompt = provider.last_messages[0].content
    assert "John Doe" in system_prompt, "System prompt should include already-collected name"

    print(f"  System prompt includes collected data: ✅")
    print(f"  Reply: {result.reply[:80]}...")
    print("  ✅ PASSED")


async def test_summarization_agent():
    """Test: Summarization Agent generates a summary."""
    print("\n--- Test: Summarization Agent ---")

    provider = MockLLMProvider()
    agent = SummarizationAgent(provider)

    session = Session(session_id="test-005", stage=ConversationStage.SUMMARY)
    session.collected_data.personal_info.name = "John Doe"
    session.collected_data.personal_info.email = "john@example.com"
    session.collected_data.personal_info.phone = "+1-555-0123"
    session.collected_data.tech_discovery.project_type = "E-commerce Web App"
    session.collected_data.tech_discovery.features = "Product catalog, cart, checkout"
    session.collected_data.scope_pricing.budget = "$10K-$15K"
    session.collected_data.scope_pricing.timeline = "3 months"

    summary = await agent.generate_summary(session)

    assert isinstance(summary, str), "Summary should be a string"
    assert len(summary) > 50, "Summary should be substantial"
    assert "John Doe" in summary, "Summary should reference client name"

    print(f"  Summary length: {len(summary)} chars")
    print(f"  Summary preview: {summary[:120]}...")
    print(f"  LLM calls: {provider.call_count}")
    print("  ✅ PASSED")


async def test_summarization_agent_fallback():
    """Test: Summarization Agent fallback when LLM fails."""
    print("\n--- Test: Summarization Agent — Fallback ---")

    # Create a provider that always fails
    class FailingProvider(MockLLMProvider):
        async def generate(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    provider = FailingProvider()
    agent = SummarizationAgent(provider)

    session = Session(session_id="test-006", stage=ConversationStage.SUMMARY)
    session.collected_data.personal_info.name = "Jane Smith"
    session.collected_data.personal_info.email = "jane@test.com"

    summary = await agent.generate_summary(session)

    assert isinstance(summary, str), "Should return fallback summary"
    assert "Jane Smith" in summary, "Fallback should include client name"
    assert "fallback" in summary.lower(), "Should indicate fallback mode"

    print(f"  Fallback summary generated: ✅")
    print(f"  Length: {len(summary)} chars")
    print("  ✅ PASSED")


async def test_email_agent():
    """Test: Email Agent composes and mock-sends emails."""
    print("\n--- Test: Email Agent ---")

    provider = MockLLMProvider()
    agent = EmailAgent(provider)

    session = Session(session_id="test-007", stage=ConversationStage.EMAIL)
    session.collected_data.personal_info.name = "John Doe"
    session.collected_data.personal_info.email = "john@example.com"
    session.collected_data.personal_info.phone = "+1-555-0123"
    session.collected_data.tech_discovery.project_type = "E-commerce Web App"
    session.collected_data.scope_pricing.budget = "$10K-$15K"
    session.collected_data.scope_pricing.timeline = "3 months"
    session.summary = "Lead Summary: John Doe wants an e-commerce app..."

    result = await agent.compose_and_send(session)

    assert isinstance(result, EmailResult), "Should return EmailResult"
    assert result.success is True, "Should succeed"
    assert result.user_email.to == "john@example.com", "User email should go to client"
    assert result.admin_email.to == "admin@colorwhistle.com", "Admin email to admin"
    assert result.user_email.body, "User email body should not be empty"
    assert result.admin_email.body, "Admin email body should not be empty"
    assert result.user_email.email_type == "user_thankyou"
    assert result.admin_email.email_type == "admin_notification"

    print(f"  User email to: {result.user_email.to}")
    print(f"  Admin email to: {result.admin_email.to}")
    print(f"  User email subject: {result.user_email.subject}")
    print(f"  Admin email subject: {result.admin_email.subject}")
    print(f"  LLM calls: {provider.call_count}")
    print(f"  Status: {result.message}")
    print("  ✅ PASSED")


async def test_email_agent_fallback():
    """Test: Email Agent fallback when LLM fails."""
    print("\n--- Test: Email Agent — Fallback ---")

    class FailingProvider(MockLLMProvider):
        async def generate(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    provider = FailingProvider()
    agent = EmailAgent(provider)

    session = Session(session_id="test-008", stage=ConversationStage.EMAIL)
    session.collected_data.personal_info.name = "Jane Smith"
    session.collected_data.personal_info.email = "jane@test.com"

    result = await agent.compose_and_send(session)

    assert result.success is True, "Should succeed with fallback"
    assert "Jane Smith" in result.user_email.body, "Fallback should include name"
    assert result.user_email.to == "jane@test.com"

    print(f"  Fallback emails composed: ✅")
    print(f"  User email to: {result.user_email.to}")
    print("  ✅ PASSED")


# ============================================
# Run All Tests
# ============================================

async def run_all_tests():
    """Run all agent tests."""
    print("=" * 60)
    print("  Agent Tests — Phases 3, 4, 5")
    print("=" * 60)

    tests = [
        test_conversation_agent_welcome,
        test_conversation_agent_personal_info,
        test_conversation_agent_scope_complete,
        test_conversation_agent_with_context,
        test_summarization_agent,
        test_summarization_agent_fallback,
        test_email_agent,
        test_email_agent_fallback,
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
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
