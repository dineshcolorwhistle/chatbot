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
  - Includes intent detection for question vs data-provision messages
  - Validates extracted data to reject nonsensical extractions
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
# System Prompts — Simplified for small models
# ============================================

PERSONA_PROMPT = """You are a friendly, professional project consultant for ColorWhistle, a software development company.

STRICT RULES:
- Keep responses to 2-3 sentences maximum.
- Sound natural and human-like.
- Never mention being an AI or LLM.
- Ask only ONE question at a time.
- If the user asks a question about ColorWhistle services, answer it using ONLY the knowledge base context provided below (if any). If no relevant knowledge base context is available, say: "I don't have specific details about that right now, but I can have our team follow up with you on this. Let's continue with your project details."
- If the user asks something completely unrelated to web development or ColorWhistle, politely steer back to the project consultation.
- NEVER make up or guess information about ColorWhistle's services, pricing, or capabilities."""

STAGE_PROMPTS: dict[ConversationStage, str] = {
    ConversationStage.WELCOME: """
You are greeting a new visitor. Welcome them warmly and ask for their name to get started.
""",

    ConversationStage.PERSONAL_INFO: """
You are collecting the user's contact details. You need: name, email, and phone number. Company is optional.

WHAT TO COLLECT (one at a time):
1. Full Name
2. Email Address
3. Phone Number
4. Company Name (optional — mention it's optional)

IMPORTANT: If the user asks a question instead of providing their details, answer their question briefly, then ask again for the missing information.

Do NOT try to extract personal info from questions — only extract from clear answers like "My name is John" or "john@email.com".
""",

    ConversationStage.TECH_DISCOVERY: """
You are learning about the user's project. You need: project type and key features.

WHAT TO COLLECT:
1. Project Type (web app, mobile app, e-commerce, API, etc.)
2. Key Features they need
3. Tech stack preferences (optional)
4. Third-party integrations (optional)

Ask about their project type first, then features.
""",

    ConversationStage.SCOPE_PRICING: """
You are discussing project scope. You need: budget range and timeline.

WHAT TO COLLECT:
1. Budget Range — be diplomatic, offer ranges like "under $5K, $5K-$15K, or $15K+"
2. Timeline — when they need it delivered (weeks/months)
3. MVP or production-ready (optional)
4. Priority features (optional)

When you have budget and timeline, indicate you have everything needed.
""",
}


# ============================================
# JSON Extraction Instruction — kept as simple
# as possible for small models
# ============================================

DATA_EXTRACTION_INSTRUCTION = """
RESPOND IN THIS EXACT JSON FORMAT:
{"response": "your reply here", "extracted_data": {}, "stage_complete": false}

RULES FOR extracted_data:
- ONLY include data the user CLEARLY provided in their CURRENT message.
- If the user asked a question, set extracted_data to {} (empty).
- Do NOT guess or invent data.
"""

# Field names per stage for the JSON extraction
STAGE_FIELD_HINTS: dict[ConversationStage, str] = {
    ConversationStage.PERSONAL_INFO: 'Use field names: "name", "email", "phone", "company"',
    ConversationStage.TECH_DISCOVERY: 'Use field names: "project_type", "tech_stack", "features", "integrations"',
    ConversationStage.SCOPE_PRICING: 'Use field names: "budget", "timeline", "mvp_or_production", "priority_features"',
}


# ============================================
# Question Detection Patterns
# ============================================

QUESTION_PATTERNS = [
    r"\?$",                          # Ends with question mark
    r"^(is|are|do|does|can|could|would|will|what|how|why|where|when|which|who)\b",
    r"^(tell me|i want to know|i need to know|explain)\b",
    r"^(what's|how's|where's|who's|when's)\b",
]


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
        # Detect if the user is asking a question vs providing data
        is_question = self._is_question(user_message)

        if is_question:
            logger.info("Detected user question — will answer without extracting data")

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

        system_prompt = self._build_system_prompt(
            session, rag_context=rag_context, is_question=is_question
        )

        # Build message list for LLM
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt)
        ]

        # Add conversation history (limited to last 10 messages for small models)
        history = session.get_history_for_llm()
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

        # Add the current user message
        messages.append(LLMMessage(role="user", content=user_message))

        # Generate LLM response — use lower temperature for small models
        try:
            llm_response = await self._llm.generate(
                messages=messages,
                temperature=0.4,
            )
            raw_content = llm_response.content

            logger.debug(
                "Conversation Agent raw LLM response (stage: %s): %s",
                session.stage.value,
                raw_content[:300],
            )

            # Parse the structured response
            reply, extracted_data, stage_complete = self._parse_llm_response(
                raw_content
            )

            # If user asked a question, force-clear any extracted data
            # (the model may still hallucinate extractions)
            if is_question:
                extracted_data = {}
                stage_complete = False

            # Validate extracted data — reject garbage
            validated_data = self._validate_extracted_data(
                session.stage, extracted_data, user_message
            )

            # Determine if stage should advance
            should_advance = stage_complete or self._check_stage_completion(
                session, validated_data
            )

            return ConversationResult(
                reply=reply,
                extracted_data=validated_data,
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

    # ============================================
    # Intent Detection
    # ============================================

    def _is_question(self, message: str) -> bool:
        """Detect if the user message is a question rather than data provision.

        Uses pattern matching to determine if the user is asking
        something (which should be answered) vs providing personal
        data (which should be extracted).

        Args:
            message: The user's message text.

        Returns:
            True if the message appears to be a question.
        """
        msg_clean = message.strip().lower()

        # Check against question patterns
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                return True

        return False

    # ============================================
    # System Prompt Builder
    # ============================================

    def _build_system_prompt(
        self, session: Session, rag_context: str = "", is_question: bool = False
    ) -> str:
        """Build a dynamic system prompt based on session context.

        Combines the persona, stage-specific instructions, collected
        data context, RAG knowledge base context, and data extraction
        format instructions.

        Args:
            session: The current session state.
            rag_context: Optional RAG context from the knowledge base.
            is_question: Whether the user's message is a question.

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

        # If user is asking a question, add explicit instruction
        if is_question:
            parts.append(
                "\nThe user is asking a QUESTION. Answer it briefly using "
                "the knowledge base context above (if available). "
                "If no relevant info is in the knowledge base, say you don't have "
                "those details and offer to have the team follow up. "
                "Then gently remind them of what information you still need. "
                "Do NOT extract any data — set extracted_data to {}."
            )

        # Add context of already-collected data
        collected_context = self._build_collected_context(session)
        if collected_context:
            parts.append(collected_context)

        # Add JSON extraction instructions
        parts.append(DATA_EXTRACTION_INSTRUCTION)

        # Add field name hints for the current stage
        field_hint = STAGE_FIELD_HINTS.get(session.stage, "")
        if field_hint and not is_question:
            parts.append(field_hint)

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
                "\nALREADY COLLECTED (do NOT re-ask):\n"
                + "\n".join(collected_items)
            )

        return ""

    # ============================================
    # Response Parsing
    # ============================================

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
        # Strategy 1: Try to extract JSON from markdown code blocks
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_response,
            re.DOTALL,
        )

        json_str = None
        if json_match:
            json_str = json_match.group(1)
        else:
            # Strategy 2: Find a JSON object that contains "response" key
            json_match = re.search(
                r'\{\s*"response"\s*:.*\}',
                raw_response,
                re.DOTALL,
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                # Strategy 3: Try to find any JSON object
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)

        if json_str:
            try:
                # Sanitize common JSON issues from small models
                json_str = self._sanitize_json(json_str)
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

        # Remove any leftover JSON-like content
        clean_response = re.sub(
            r'\{\s*"response".*', "", clean_response, flags=re.DOTALL
        ).strip()

        if not clean_response:
            clean_response = (
                "I'm here to help! Could you tell me a bit more about "
                "what you're looking for?"
            )

        return clean_response, {}, False

    def _sanitize_json(self, json_str: str) -> str:
        """Sanitize common JSON issues from small LLM outputs.

        Small models often produce slightly malformed JSON.
        This attempts to fix common issues.

        Args:
            json_str: The raw JSON string to sanitize.

        Returns:
            A cleaned JSON string.
        """
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix unescaped newlines inside strings
        json_str = json_str.replace("\n", " ")

        # Remove any control characters
        json_str = re.sub(r"[\x00-\x1f\x7f]", " ", json_str)

        return json_str

    # ============================================
    # Data Validation
    # ============================================

    def _validate_extracted_data(
        self, stage: ConversationStage, data: dict, user_message: str
    ) -> dict:
        """Validate extracted data to reject nonsensical extractions.

        Small models often hallucinate data fields. This method
        ensures extracted values actually make sense.

        Args:
            stage: The current conversation stage.
            data: The extracted data dictionary.
            user_message: The original user message for cross-reference.

        Returns:
            A cleaned dictionary with only valid extractions.
        """
        if not data:
            return {}

        validated: dict = {}
        msg_lower = user_message.strip().lower()

        for field_name, value in data.items():
            value_str = str(value).strip()

            # Skip empty or very short values (likely garbage)
            if not value_str or len(value_str) < 2:
                logger.debug("Rejecting field '%s': value too short ('%s')", field_name, value_str)
                continue

            # Skip values that are obviously not real data
            garbage_values = {
                "no", "yes", "n/a", "none", "null", "undefined",
                "not provided", "not available", "unknown",
            }
            if value_str.lower() in garbage_values:
                logger.debug("Rejecting field '%s': garbage value ('%s')", field_name, value_str)
                continue

            # Stage-specific validation
            if stage == ConversationStage.PERSONAL_INFO:
                if field_name == "name":
                    # Name should be alphabetic words, at least 2 chars
                    if not re.match(r"^[a-zA-Z\s\.\-']{2,}$", value_str):
                        logger.debug("Rejecting name: doesn't look like a name ('%s')", value_str)
                        continue
                    # Name should appear in or relate to the user message
                    if not self._value_plausible_from_message(value_str, user_message):
                        logger.debug("Rejecting name: not found in user message ('%s')", value_str)
                        continue

                elif field_name == "email":
                    # Basic email validation
                    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value_str):
                        logger.debug("Rejecting email: invalid format ('%s')", value_str)
                        continue
                    # Email should appear in user message
                    if value_str.lower() not in msg_lower:
                        logger.debug("Rejecting email: not in user message ('%s')", value_str)
                        continue

                elif field_name == "phone":
                    # Phone should contain digits
                    digits = re.sub(r"[^\d]", "", value_str)
                    if len(digits) < 7:
                        logger.debug("Rejecting phone: too few digits ('%s')", value_str)
                        continue
                    # Phone digits should appear in user message
                    msg_digits = re.sub(r"[^\d]", "", user_message)
                    if digits not in msg_digits:
                        logger.debug("Rejecting phone: digits not in user message ('%s')", value_str)
                        continue

                elif field_name == "company":
                    if not self._value_plausible_from_message(value_str, user_message):
                        logger.debug("Rejecting company: not found in user message ('%s')", value_str)
                        continue

            elif stage == ConversationStage.TECH_DISCOVERY:
                if field_name in ("project_type", "features", "tech_stack", "integrations"):
                    if not self._value_plausible_from_message(value_str, user_message):
                        logger.debug(
                            "Rejecting %s: not plausible from message ('%s')",
                            field_name, value_str
                        )
                        continue

            elif stage == ConversationStage.SCOPE_PRICING:
                if field_name in ("budget", "timeline", "mvp_or_production", "priority_features"):
                    if not self._value_plausible_from_message(value_str, user_message):
                        logger.debug(
                            "Rejecting %s: not plausible from message ('%s')",
                            field_name, value_str
                        )
                        continue

            validated[field_name] = value_str

        if validated != data:
            logger.info(
                "Data validation — original: %s, validated: %s",
                list(data.keys()), list(validated.keys()),
            )

        return validated

    def _value_plausible_from_message(self, value: str, message: str) -> bool:
        """Check if an extracted value is plausibly derived from the user message.

        This prevents the model from hallucinating data that the user
        never actually said.

        Args:
            value: The extracted value to check.
            message: The user's original message.

        Returns:
            True if the value seems to come from the message.
        """
        msg_lower = message.lower()
        value_lower = value.lower()

        # Direct containment check
        if value_lower in msg_lower:
            return True

        # Check if individual significant words from the value appear in the message
        value_words = [w for w in value_lower.split() if len(w) > 2]
        if value_words:
            match_count = sum(1 for w in value_words if w in msg_lower)
            match_ratio = match_count / len(value_words)
            if match_ratio >= 0.5:
                return True

        return False

    # ============================================
    # Stage Completion Check
    # ============================================

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
