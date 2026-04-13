"""
Conversation Agent — 🗣️ LLM-Powered Dialogue Agent

Handles all user-facing conversation during stages 1–4:
  - Welcome (stage 1)
  - Personal Information (stage 2)
  - Technical Discovery (stage 3)
  - Scope & Pricing (stage 4)

The agent builds dynamic system prompts based on the current stage
and collected data, then uses the LLM to generate contextually
appropriate responses while extracting structured data from user input.

Design:
  - Receives LLMProvider via constructor (no global state)
  - Returns AgentResult with both reply text and extracted data
  - Uses JSON-structured LLM responses for reliable data extraction
  - Falls back to raw text response if JSON parsing fails
"""

import json
import logging
import re
from dataclasses import dataclass, field

from models.schemas import ConversationStage, Session
from providers.base import LLMProvider, LLMMessage

logger = logging.getLogger(__name__)


# ============================================
# Agent Result Type
# ============================================

@dataclass
class ConversationResult:
    """Result from the Conversation Agent.

    Attributes:
        reply: The assistant's response message to show the user.
        extracted_data: Dictionary of field-value pairs extracted from
                        the user's message. Keys match the field names
                        in the collected data models.
        should_advance: Whether the agent recommends advancing to
                        the next conversation stage.
    """

    reply: str
    extracted_data: dict = field(default_factory=dict)
    should_advance: bool = False


# ============================================
# System Prompts per Stage
# ============================================

PERSONA_PROMPT = """You are a friendly, professional project consultant AI assistant for a software development company called ColorWhistle.

Your personality:
- Warm and approachable, but professional
- Patient and helpful
- You ask one question at a time
- You acknowledge and validate user responses before asking the next question
- You keep responses concise (2-4 sentences max)
- You never break character or mention being an AI/LLM"""

STAGE_PROMPTS: dict[ConversationStage, str] = {
    ConversationStage.WELCOME: """
CURRENT STAGE: Welcome
GOAL: Greet the user warmly and introduce yourself as a project consultant.
Explain that you'll help them scope out their project requirements.
Ask them to share their name to get started.

Keep it brief and inviting. Make them feel comfortable.
""",

    ConversationStage.PERSONAL_INFO: """
CURRENT STAGE: Collecting Personal Information
GOAL: Collect the following personal details, one at a time:
- Full Name (required)
- Email Address (required) — validate it looks like a real email
- Contact Phone Number (required) — accept various formats
- Company Name (optional) — ask but accept if they skip

RULES:
- Ask for ONE piece of information at a time
- After getting each piece, acknowledge it and ask for the next
- If the user provides multiple pieces at once, extract all of them
- Validate email format naturally (e.g., "Hmm, that doesn't look like a valid email. Could you double-check?")
- For phone numbers, accept any reasonable format
- When you have name + email + phone, ask about company but mention it's optional
- Once you have at least name + email + phone, indicate collection is complete
""",

    ConversationStage.TECH_DISCOVERY: """
CURRENT STAGE: Technical Discovery
GOAL: Understand the user's project requirements:
- Project Type (required) — e.g., web app, mobile app, API, chatbot, e-commerce, etc.
- Technology Preferences (optional) — any preferred tech stack
- Key Features/Modules (required) — what the project needs to do
- Integrations (optional) — third-party services, APIs, payment gateways

RULES:
- Start by asking about their project type
- Ask follow-up questions to clarify scope
- If they're vague, offer examples to help them articulate needs
- Collect features as a comma-separated list if possible
- When you have at least project_type and features, move forward
""",

    ConversationStage.SCOPE_PRICING: """
CURRENT STAGE: Scope & Pricing Discussion
GOAL: Understand project scope and business constraints:
- Budget Range (required) — ask diplomatically, offer ranges if they're hesitant
- Timeline (required) — when do they need the project delivered
- MVP vs Full Production (optional) — are they looking for a quick MVP or production-ready
- Priority Features (optional) — what's most critical to launch

RULES:
- Be diplomatic about budget — it's a sensitive topic
- Offer budget ranges as options (e.g., "Are you thinking under $5K, $5K-$15K, or $15K+?")
- For timeline, get specific (weeks/months)
- When you have budget + timeline, indicate you have everything needed
""",
}

DATA_EXTRACTION_INSTRUCTION = """

RESPONSE FORMAT:
You MUST respond with valid JSON in this exact format:
```json
{
  "response": "Your conversational reply to the user",
  "extracted_data": {
    "field_name": "extracted_value"
  },
  "stage_complete": false
}
```

EXTRACTED_DATA FIELD NAMES by stage:
- Personal Info: "name", "email", "phone", "company"
- Tech Discovery: "project_type", "tech_stack", "features", "integrations"
- Scope & Pricing: "budget", "timeline", "mvp_or_production", "priority_features"

Only include fields that you can extract from the current user message.
Set "stage_complete" to true ONLY when all required fields for the current stage are collected.
If nothing new can be extracted, use an empty object: "extracted_data": {}
"""


class ConversationAgent:
    """LLM-powered agent that leads the conversation with users.

    Manages natural dialogue across stages 1-4, extracting structured
    data from free-form user responses while maintaining a friendly,
    professional tone.

    Optionally integrates with the KnowledgeBase for RAG-augmented
    responses about company details, services, and pricing.

    Attributes:
        _llm: The LLM provider instance used for generation.
        _knowledge_base: Optional knowledge base for RAG context retrieval.
    """

    def __init__(self, llm_provider: LLMProvider, knowledge_base=None) -> None:
        """Initialize the Conversation Agent.

        Args:
            llm_provider: The LLM provider to use for generating responses.
            knowledge_base: Optional KnowledgeBase instance for RAG retrieval.
        """
        self._llm = llm_provider
        self._knowledge_base = knowledge_base

    async def process_message(
        self,
        session: Session,
        user_message: str,
    ) -> ConversationResult:
        """Process a user message and generate a response.

        Builds a context-aware system prompt, sends the conversation
        to the LLM, and extracts any structured data from the response.

        Args:
            session: The current session with conversation history and data.
            user_message: The user's latest message.

        Returns:
            ConversationResult with the reply, extracted data, and
            stage advancement recommendation.
        """
        # Query knowledge base for relevant context (RAG)
        rag_context = ""
        if self._knowledge_base:
            try:
                results = await self._knowledge_base.query(
                    question=user_message,
                    top_k=5,
                    score_threshold=0.3,
                )
                rag_context = self._knowledge_base.format_context_for_llm(results)
            except Exception as e:
                logger.warning("RAG context retrieval failed: %s", e)

        system_prompt = self._build_system_prompt(session, rag_context=rag_context)

        # Build message list for LLM
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt)
        ]

        # Add conversation history (limited to last 20 messages for context)
        history = session.get_history_for_llm()
        recent_history = history[-20:] if len(history) > 20 else history
        for msg in recent_history:
            messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

        # Add the current user message
        messages.append(LLMMessage(role="user", content=user_message))

        # Generate LLM response
        try:
            llm_response = await self._llm.generate(
                messages=messages,
                temperature=0.7,
            )
            raw_content = llm_response.content

            logger.debug(
                "Conversation Agent raw LLM response (stage: %s): %s",
                session.stage.value,
                raw_content[:200],
            )

            # Parse the structured response
            reply, extracted_data, stage_complete = self._parse_llm_response(
                raw_content
            )

            # Determine if stage should advance
            should_advance = stage_complete or self._check_stage_completion(
                session, extracted_data
            )

            return ConversationResult(
                reply=reply,
                extracted_data=extracted_data,
                should_advance=should_advance,
            )

        except (ConnectionError, RuntimeError) as e:
            logger.error("Conversation Agent LLM error: %s", e)
            return ConversationResult(
                reply=(
                    "I apologize, but I'm having trouble processing your request "
                    "right now. Could you please try again?"
                ),
                extracted_data={},
                should_advance=False,
            )

    def _build_system_prompt(self, session: Session, rag_context: str = "") -> str:
        """Build a dynamic system prompt based on session context.

        Combines the persona, stage-specific instructions, collected
        data context, RAG knowledge base context, and data extraction
        format instructions.

        Args:
            session: The current session state.
            rag_context: Optional RAG context from the knowledge base.

        Returns:
            A complete system prompt string.
        """
        parts: list[str] = [PERSONA_PROMPT]

        # Add stage-specific prompt
        stage_prompt = STAGE_PROMPTS.get(session.stage, "")
        if stage_prompt:
            parts.append(stage_prompt)

        # Add RAG knowledge base context (if available)
        if rag_context:
            parts.append(rag_context)

        # Add context of already-collected data
        collected_context = self._build_collected_context(session)
        if collected_context:
            parts.append(collected_context)

        # Add JSON extraction instructions
        parts.append(DATA_EXTRACTION_INSTRUCTION)

        return "\n".join(parts)

    def _build_collected_context(self, session: Session) -> str:
        """Build context string of already-collected data.

        This tells the LLM what information has already been gathered
        so it doesn't re-ask questions.

        Args:
            session: The current session state.

        Returns:
            A formatted string of collected data, or empty string if nothing collected.
        """
        data = session.collected_data
        collected_items: list[str] = []

        # Personal info
        pi = data.personal_info
        if pi.name:
            collected_items.append(f"- Name: {pi.name}")
        if pi.email:
            collected_items.append(f"- Email: {pi.email}")
        if pi.phone:
            collected_items.append(f"- Phone: {pi.phone}")
        if pi.company:
            collected_items.append(f"- Company: {pi.company}")

        # Tech discovery
        td = data.tech_discovery
        if td.project_type:
            collected_items.append(f"- Project Type: {td.project_type}")
        if td.tech_stack:
            collected_items.append(f"- Tech Stack: {td.tech_stack}")
        if td.features:
            collected_items.append(f"- Features: {td.features}")
        if td.integrations:
            collected_items.append(f"- Integrations: {td.integrations}")

        # Scope & pricing
        sp = data.scope_pricing
        if sp.budget:
            collected_items.append(f"- Budget: {sp.budget}")
        if sp.timeline:
            collected_items.append(f"- Timeline: {sp.timeline}")
        if sp.mvp_or_production:
            collected_items.append(f"- Scope: {sp.mvp_or_production}")
        if sp.priority_features:
            collected_items.append(f"- Priority Features: {sp.priority_features}")

        if collected_items:
            return (
                "\nALREADY COLLECTED DATA (do NOT re-ask these):\n"
                + "\n".join(collected_items)
            )

        return ""

    def _parse_llm_response(
        self, raw_response: str
    ) -> tuple[str, dict, bool]:
        """Parse the LLM's JSON response into components.

        Attempts to extract structured JSON from the response.
        Falls back to using the raw response as plain text if
        JSON parsing fails.

        Args:
            raw_response: The raw text from the LLM.

        Returns:
            A tuple of (reply_text, extracted_data_dict, stage_complete_bool).
        """
        # Try to extract JSON from the response
        # LLMs sometimes wrap JSON in markdown code blocks
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_response,
            re.DOTALL,
        )

        json_str = None
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

        if json_str:
            try:
                parsed = json.loads(json_str)

                reply = parsed.get("response", "").strip()
                extracted = parsed.get("extracted_data", {})
                stage_complete = parsed.get("stage_complete", False)

                if reply:
                    # Clean extracted data — remove empty/null values
                    clean_extracted = {
                        k: v
                        for k, v in extracted.items()
                        if v is not None and str(v).strip()
                    }

                    logger.info(
                        "Parsed conversation response — "
                        "extracted fields: %s, stage_complete: %s",
                        list(clean_extracted.keys()),
                        stage_complete,
                    )

                    return reply, clean_extracted, stage_complete

            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning("Failed to parse LLM JSON response: %s", e)

        # Fallback: use the raw response as-is
        logger.info("Using raw LLM response (JSON parsing failed)")

        # Strip any leftover JSON artifacts from the response
        clean_response = raw_response.strip()

        # Remove markdown code blocks if present
        clean_response = re.sub(
            r"```(?:json)?.*?```", "", clean_response, flags=re.DOTALL
        ).strip()

        if not clean_response:
            clean_response = (
                "I'm here to help! Could you tell me a bit more about "
                "what you're looking for?"
            )

        return clean_response, {}, False

    def _check_stage_completion(
        self, session: Session, new_data: dict
    ) -> bool:
        """Check if the current stage's required data is complete.

        Considers both the already-collected data in the session and
        any newly extracted data from the current message.

        Args:
            session: The current session.
            new_data: Newly extracted data from the latest message.

        Returns:
            True if all required fields for the current stage are collected.
        """
        stage = session.stage

        if stage == ConversationStage.WELCOME:
            # Welcome stage advances when the user responds
            return True

        elif stage == ConversationStage.PERSONAL_INFO:
            pi = session.collected_data.personal_info
            name = pi.name or new_data.get("name")
            email = pi.email or new_data.get("email")
            phone = pi.phone or new_data.get("phone")
            return all([name, email, phone])

        elif stage == ConversationStage.TECH_DISCOVERY:
            td = session.collected_data.tech_discovery
            project_type = td.project_type or new_data.get("project_type")
            features = td.features or new_data.get("features")
            return all([project_type, features])

        elif stage == ConversationStage.SCOPE_PRICING:
            sp = session.collected_data.scope_pricing
            budget = sp.budget or new_data.get("budget")
            timeline = sp.timeline or new_data.get("timeline")
            return all([budget, timeline])

        return False
