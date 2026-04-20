"""
Conversation Agent — 🗣️ LLM-Powered Dialogue Agent

Single-block conversational agent that handles all user interactions.
No stage-based routing — the agent naturally responds to user queries
and extracts structured data (name, email, project details) as the
conversation progresses.

Key behaviors:
  - Always answers the user's question FIRST
  - Extracts name, email, and project details from any message
  - Gently nudges for name/email before chat closes (without repeating)
  - Redirects budget/pricing questions to the human team
  - Filters irrelevant off-topic questions

Design:
  - Receives LLMProvider via constructor (no global state)
  - Returns ConversationResult with both reply text and extracted data
  - Uses JSON-structured LLM responses for reliable data extraction
  - Falls back to raw text response if JSON parsing fails
  - Includes intent detection for question vs data-provision messages
  - Validates extracted data to reject nonsensical extractions
"""

import json
import logging
import re
from dataclasses import dataclass, field

from models.schemas import Session
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
    """

    reply: str
    extracted_data: dict = field(default_factory=dict)


# ============================================
# System Prompt — Single unified prompt
# ============================================

PERSONA_PROMPT = """You are a friendly, professional project consultant for ColorWhistle, a software development company.

STRICT RULES:
- Keep responses to 2-3 sentences maximum.
- Sound natural and human-like. Act completely like a human.
- When greeting the user, you MUST start your message with the name "ColorWhistle" (e.g. "Welcome to ColorWhistle!", "Hello from ColorWhistle!").
- Wherever appropriate and needed, use the name "ColorWhistle".
- Never mention being an AI or LLM.
- ALWAYS answer the user's question or respond to their message FIRST.
- NEVER suggest any price or costing.
- If the user asks about budget or pricing, respond with: "Our team will reach out and discuss about the budget with you."
- Try to understand the technical details from the user's input, but do NOT forcefully ask for them.
- If the user asks a question about ColorWhistle services, answer it using ONLY the knowledge base context provided below (if any). If no relevant knowledge base context is available, say: "I don't have specific details about that right now, but I can have our team follow up with you on this."
- If the user asks something completely unrelated to web development or ColorWhistle, politely steer back to the project consultation.
- NEVER make up or guess information about ColorWhistle's services, pricing, or capabilities.
- IMPORTANT: Do NOT ask the user any questions like "What is your name?", "What is your email?", or "What are your features?". The automated system will append polite requests to the end of your response when needed.
- ONLY generate the consultant's immediate reply. NEVER simulate, predict, or write out the user's next response. Stop generating immediately after your own reply."""


# ============================================
# JSON Extraction Instruction
# ============================================

DATA_EXTRACTION_INSTRUCTION = """
RESPOND IN THIS EXACT JSON FORMAT:
{"response": "your reply here", "extracted_data": {}}

RULES FOR extracted_data:
- ONLY include data the user CLEARLY provided in their CURRENT message.
- If the user asked a question, set extracted_data to {} (empty).
- Do NOT guess or invent data.
- Do NOT simulate or predict the user's next response inside the "response" property. Limit the "response" string solely to the consultant's immediate reply.
- Use field names: "name", "email", "project_type", "tech_stack", "features", "integrations", "budget", "timeline"
"""


# ============================================
# Question Detection Patterns
# ============================================

QUESTION_PATTERNS = [
    r"\?$",                          # Ends with question mark
    r"^(is|are|do|does|can|could|would|will|what|how|why|where|when|which|who)\b",
    r"^(tell me|i want to know|i need to know|explain)\b",
    r"^(what's|how's|where's|who's|when's)\b",
]

# Patterns that indicate a data-provision message even if it looks like a question
DATA_PROVISION_PATTERNS = [
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi'm\b",
    r"\bcall me\b",
    r"\bname's\b",
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email pattern
]


# ============================================
# Irrelevant Question Patterns
# ============================================

IRRELEVANT_PATTERNS = [
    # Time & date
    r"\b(what|tell).*(time|date|day|month|year)\b",
    r"\b(current|today).*(time|date)\b",
    r"\bwhat time\b",
    r"\bwhat day\b",
    # Weather
    r"\b(weather|temperature|forecast|rain|sunny|cloudy|snow)\b",
    # Math & calculations
    r"\b(calculate|compute|solve|math|equation)\b",
    r"^\d+\s*[\+\-\*\/\%]\s*\d+",     # e.g. "5 + 3"
    # Sports & entertainment
    r"\b(score|match|game|cricket|football|soccer|basketball|baseball|tennis|movie|song|music|netflix)\b",
    # General knowledge / trivia
    r"\b(capital of|president of|population of|who invented|who discovered|who is the)\b",
    r"\b(meaning of life|tell me a joke|joke|funny|riddle)\b",
    # Personal / social
    r"\b(your name|who are you|are you real|are you human|are you ai|are you a bot)\b",
    r"\b(how old are you|where do you live|your age|your favorite)\b",
    # News & politics
    r"\b(latest news|breaking news|election|politics|stock market|crypto|bitcoin)\b",
    # Food & recipes
    r"\b(recipe|cook|food|restaurant|calories)\b",
    # Health (non-project)
    r"\b(headache|medicine|doctor|symptom|diet|exercise|workout)\b",
    # Navigation & travel
    r"\b(directions to|how to get to|nearest|flight|hotel|travel|vacation)\b",
    # Random tasks
    r"\b(translate|write a poem|write a story|sing|draw)\b",
    r"\b(lottery|horoscope|zodiac|astrology)\b",
]

# Keywords that indicate a RELEVANT question (about services, web dev, project)
RELEVANT_KEYWORDS = [
    "colorwhistle", "project", "website", "web app", "mobile app",
    "development", "design", "service", "pricing", "cost", "budget",
    "timeline", "deadline", "feature", "technology", "tech stack",
    "react", "python", "api", "database", "hosting", "deploy",
    "ecommerce", "e-commerce", "cms", "wordpress", "seo",
    "portfolio", "consultation", "team", "experience", "clients",
    "support", "maintenance", "integration", "payment",
    "frontend", "backend", "fullstack", "full-stack",
]

# Default redirect for irrelevant questions
DEFAULT_REDIRECT = (
    "I appreciate the question, but that's outside what I can help with! 😊 "
    "I'm your project consultant at ColorWhistle, and I'm here to understand "
    "your project needs. Let's continue with our consultation!"
)


class ConversationAgent:
    """LLM-powered agent that leads the conversation with users.

    Single-block conversational agent — no stage-based routing.
    Responds naturally to user messages, extracts structured data
    (name, email, project details) as available, and gently nudges
    for contact info before the chat limit is reached.

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

    def _clean_history_content(self, content: str) -> str:
        """Remove appended automated nudges from history so the LLM doesn't learn and parrot them."""
        nudges = [
            r"By the way, may I know your name\? 😊",
            r"By the way, when you're ready, could you please share your name\? 😊",
            r"Thanks, .*? Could you also share your email address so we can follow up\? 😊",
            r"Thanks, .*?! Could you also share your email address so we can follow up\? 😊",
            r"Before we wrap up, could you share your name so we can personalize your experience\? 😊",
            r"Before we wrap up, could you please share your email address so our team can follow up with you\? 😊",
        ]
        cleansed = content
        for nudge in nudges:
            cleansed = re.sub(nudge, "", cleansed, flags=re.IGNORECASE).strip()
        return cleansed

    async def process_message(
        self,
        session: Session,
        user_message: str,
    ) -> ConversationResult:
        """Process a user message and generate a response.

        Uses a two-pass approach for questions:
          Pass 1: Answer the user's question with a focused prompt
          Pass 2: Append a gentle data nudge via code (not LLM)

        For data-provision messages, uses the standard single-pass
        JSON extraction flow.

        Args:
            session: The current session with conversation history and data.
            user_message: The user's latest message.

        Returns:
            ConversationResult with the reply and extracted data.
        """
        # Detect if the user is asking a question vs providing data
        is_question = self._is_question(user_message)

        if is_question:
            logger.info("Detected user question — checking relevance")

            # Intercept irrelevant questions BEFORE they reach the LLM
            if self._is_irrelevant_question(user_message):
                logger.info("Irrelevant question detected — returning redirect")
                return ConversationResult(
                    reply=DEFAULT_REDIRECT,
                    extracted_data={},
                )

            logger.info("Relevant question — using answer-first flow")
            return await self._handle_question(session, user_message)

        # Standard data-collection flow for non-question messages
        return await self._handle_data_message(session, user_message)

    async def _handle_question(
        self,
        session: Session,
        user_message: str,
    ) -> ConversationResult:
        """Handle a user question with a focused answer-first approach.

        Uses a simplified prompt that tells the LLM to ONLY answer
        the question. Then appends a gentle data nudge via code
        so the small model doesn't have to juggle both tasks.

        Args:
            session: The current session.
            user_message: The user's question.

        Returns:
            ConversationResult with the answer + gentle nudge.
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

        # Build a SIMPLE, focused "answer only" prompt
        system_prompt = self._build_question_prompt(session, rag_context)

        # Build message list
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt)
        ]

        # Add limited history for context
        history = session.get_history_for_llm()
        recent_history = history[-6:] if len(history) > 6 else history
        for msg in recent_history:
            content = msg["content"]
            if msg["role"] == "assistant":
                content = self._clean_history_content(content)
            if content:
                messages.append(LLMMessage(role=msg["role"], content=content))

        messages.append(LLMMessage(role="user", content=user_message))

        try:
            # Hardcode budget response to guarantee compliance with business rules
            is_budget_question = bool(re.search(r"\b(budget|cost|price|pricing|how much|amount)\b", user_message, re.IGNORECASE))
            if is_budget_question:
                answer = "Our team will reach out and discuss about the budget with you."
            else:
                llm_response = await self._llm.generate(
                    messages=messages,
                    temperature=0.4,
                )
                answer = llm_response.content.strip()

                logger.debug(
                    "Question answer raw response: %s",
                    answer[:300],
                )

                # Clean up: strip any JSON artifacts the model may produce
                answer = self._clean_answer_response(answer)

            if not answer:
                answer = (
                    "That's a great question! I don't have the specific details "
                    "right now, but our team can definitely help with that."
                )

            # Even in question mode, try regex extraction for personal info
            # so we don't lose data from hybrid messages
            regex_data = self._regex_extract_personal_info(user_message)
            if regex_data:
                logger.info("Regex extracted data from question: %s", list(regex_data.keys()))

            # Append gentle nudge for name/email if still missing
            nudge = self._build_gentle_nudge(session, regex_data)
            if nudge and nudge not in answer:
                answer = f"{answer}\n\n{nudge}"

            return ConversationResult(
                reply=answer,
                extracted_data=regex_data,
            )

        except (ConnectionError, RuntimeError) as e:
            logger.error("Question answering LLM error: %s", e)
            return ConversationResult(
                reply=(
                    "I apologize, but I'm having trouble processing your request "
                    "right now. Could you please try again?"
                ),
                extracted_data={},
            )

    async def _handle_data_message(
        self,
        session: Session,
        user_message: str,
    ) -> ConversationResult:
        """Handle a data-provision message with JSON extraction.

        Standard flow: the LLM extracts structured data from the
        user's message and continues the conversation.

        Args:
            session: The current session.
            user_message: The user's message (not a question).

        Returns:
            ConversationResult with reply and extracted data.
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

        system_prompt = self._build_system_prompt(
            session, rag_context=rag_context
        )

        # Build message list for LLM
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt)
        ]

        # Add conversation history (limited to last 10 messages for small models)
        history = session.get_history_for_llm()
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            content = msg["content"]
            if msg["role"] == "assistant":
                content = self._clean_history_content(content)
            if content:
                messages.append(LLMMessage(role=msg["role"], content=content))

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
                "Conversation Agent raw LLM response: %s",
                raw_content[:300],
            )

            # Parse the structured response
            reply, extracted_data = self._parse_llm_response(raw_content)

            # Enforce budget rule for hybrid messages (e.g. "I am Bob. How much does it cost?")
            is_budget_inquiry = bool(re.search(r"\b(budget|cost|price|pricing|how much|amount)\b", user_message, re.IGNORECASE))
            if is_budget_inquiry and "?" in user_message:
                reply = "Our team will reach out and discuss about the budget with you."

            # Validate extracted data — reject garbage
            validated_data = self._validate_extracted_data(
                extracted_data, user_message
            )

            # Regex fallback: try code-based regex extraction (reliable with small models)
            regex_data = self._regex_extract_personal_info(user_message)
            for field_name, value in regex_data.items():
                if field_name not in validated_data:
                    validated_data[field_name] = value
                    logger.info(
                        "Regex fallback extracted %s: %s", field_name, value
                    )

            # Append gentle nudge for name/email if still missing
            nudge = self._build_gentle_nudge(session, validated_data)
            if nudge and nudge not in reply:
                reply = f"{reply}\n\n{nudge}"

            return ConversationResult(
                reply=reply,
                extracted_data=validated_data,
            )

        except (ConnectionError, RuntimeError) as e:
            logger.error("Conversation Agent LLM error: %s", e)
            return ConversationResult(
                reply=(
                    "I apologize, but I'm having trouble processing your request "
                    "right now. Could you please try again?"
                ),
                extracted_data={},
            )

    # ============================================
    # Intent Detection
    # ============================================

    def _is_question(self, message: str) -> bool:
        """Detect if the user message is a question rather than data provision.

        Uses pattern matching to determine if the user is asking
        something (which should be answered) vs providing personal
        data (which should be extracted).

        Data-provision patterns take priority: if the message contains
        indicators like "my name is" or an email address, it's treated
        as data provision even if the sentence also ends with "?".

        Args:
            message: The user's message text.

        Returns:
            True if the message appears to be a question.
        """
        msg_clean = message.strip().lower()

        # Data provision patterns take priority over question patterns
        for pattern in DATA_PROVISION_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                logger.debug("Data provision pattern matched — treating as data message")
                return False

        # Check against question patterns
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                return True

        return False

    def _is_irrelevant_question(self, message: str) -> bool:
        """Detect if a question is irrelevant to the project consultation.

        Catches off-topic questions (time, weather, math, sports, etc.)
        BEFORE they reach the LLM.

        Args:
            message: The user's message text.

        Returns:
            True if the question is off-topic and should be redirected.
        """
        msg_lower = message.strip().lower()

        # First check: does it contain any relevant keywords?
        for keyword in RELEVANT_KEYWORDS:
            if keyword in msg_lower:
                logger.debug(
                    "Question contains relevant keyword '%s' — treating as relevant",
                    keyword,
                )
                return False

        # Second check: does it match any irrelevant patterns?
        for pattern in IRRELEVANT_PATTERNS:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                logger.debug(
                    "Question matched irrelevant pattern: %s",
                    pattern,
                )
                return True

        return False

    # ============================================
    # System Prompt Builders
    # ============================================

    def _build_question_prompt(
        self, session: Session, rag_context: str = ""
    ) -> str:
        """Build a prompt focused ONLY on answering the user's question directly.

        Args:
            session: The current session state.
            rag_context: Optional RAG context from the knowledge base.

        Returns:
            A focused question-answering system prompt.
        """
        parts: list[str] = [
            "You are a helpful human project consultant for ColorWhistle, a software development company.",
            "",
            "YOUR TASK: Answer the user's question in 2-3 sentences FIRST.",
            "Use the company information below if available.",
            "If you don't have the answer, say: \"I don't have specific details about that right now, but I can have our team follow up with you.\"",
            "",
            "RULES:",
            "- Answer the user's question directly.",
            "- Do NOT ask the user for their name, email, or any missing information. Our automated system will append this.",
            "- Do NOT output JSON. Just write a plain text answer.",
            "- Keep it short: 2-3 sentences maximum.",
            "- Do NOT say you are an AI. Act completely like a human.",
            "- Make sure to use the name 'ColorWhistle' where applicable.",
            "- CRITICAL: If the user asks about budget or pricing, respond EXACTLY with: 'Our team will reach out and discuss about the budget with you.'",
            "- NEVER make up information about ColorWhistle. Only use the company info below.",
            "- ONLY generate the consultant's immediate reply. NEVER simulate, predict, or write out the user's next response. Stop generating immediately after your own reply.",
        ]

        if rag_context:
            parts.append("")
            parts.append("=== COMPANY INFORMATION ===")
            parts.append(rag_context)

        return "\n".join(parts)

    def _build_system_prompt(
        self, session: Session, rag_context: str = ""
    ) -> str:
        """Build a unified system prompt for data-collection messages.

        Combines the persona, collected data context, RAG knowledge
        base context, and data extraction format instructions.

        Args:
            session: The current session state.
            rag_context: Optional RAG context from the knowledge base.

        Returns:
            A complete system prompt string.
        """
        parts: list[str] = [PERSONA_PROMPT]

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

    def _build_gentle_nudge(
        self, session: Session, pending_data: dict
    ) -> str:
        """Build a code-generated gentle nudge for missing name/email.

        This is appended to the LLM's answer via code, NOT
        generated by the LLM, so the nudge is always correct.
        It checks chat history to avoid repeating the same nudge.

        Args:
            session: The current session state.
            pending_data: Data extracted from the current message (not yet in session).

        Returns:
            A gentle data request string, or empty string if nothing needed.
        """
        pi = session.collected_data.personal_info

        has_name = pi.name or pending_data.get("name")
        has_email = pi.email or pending_data.get("email")

        # If both are collected, no nudge needed
        if has_name and has_email:
            return ""

        # Check last bot message to avoid repeating the same nudge
        history = session.get_history_for_llm()
        last_bot_msg = ""
        for msg in reversed(history):
            if msg["role"] == "assistant":
                last_bot_msg = msg["content"].lower()
                break

        if not has_name:
            if "share your name" in last_bot_msg or "may i know your name" in last_bot_msg:
                return ""
            return "Before we wrap up, could you share your name so we can personalize your experience? 😊"
        elif not has_email:
            if "share your email" in last_bot_msg or "email address so" in last_bot_msg:
                return ""
            name_to_use = pi.name or pending_data.get("name", "")
            return f"Thanks, {name_to_use}! Could you also share your email address so our team can follow up with you? 😊"

        return ""

    def _clean_answer_response(self, raw: str) -> str:
        """Clean up an answer-only response from the LLM.

        Small models may still output JSON or markdown even when
        told not to. This strips those artifacts.

        Args:
            raw: The raw LLM response.

        Returns:
            Clean plain-text answer.
        """
        text = raw.strip()

        # Try to extract from JSON if the model still outputs it
        try:
            json_match = re.search(r'"response"\s*:\s*"(.*?)"', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
                # Unescape JSON string
                text = text.replace('\\n', '\n').replace('\\"', '"')
        except Exception:
            pass

        # Remove markdown code blocks
        text = re.sub(r"```(?:json)?.*?```", "", text, flags=re.DOTALL).strip()

        # Remove leftover JSON artifacts
        text = re.sub(r'\{\s*"response".*', "", text, flags=re.DOTALL).strip()
        text = re.sub(r'\{\s*"extracted_data".*', "", text, flags=re.DOTALL).strip()

        # Remove any lines that look like JSON keys
        lines = text.split("\n")
        clean_lines = [
            line for line in lines
            if not re.match(r'^\s*["\{\}]', line.strip())
        ]
        text = "\n".join(clean_lines).strip()

        return text

    # ============================================
    # Response Parsing
    # ============================================

    def _parse_llm_response(
        self, raw_response: str
    ) -> tuple[str, dict]:
        """Parse the LLM's JSON response into components.

        Attempts to extract structured JSON from the response.
        Falls back to using the raw response as plain text if
        JSON parsing fails.

        Args:
            raw_response: The raw text from the LLM.

        Returns:
            A tuple of (reply_text, extracted_data_dict).
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

                if reply:
                    # Clean extracted data — remove empty/null values
                    clean_extracted = {
                        k: v
                        for k, v in extracted.items()
                        if v is not None and str(v).strip()
                    }

                    logger.info(
                        "Parsed conversation response — "
                        "extracted fields: %s",
                        list(clean_extracted.keys()),
                    )

                    return reply, clean_extracted

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

        return clean_response, {}

    def _sanitize_json(self, json_str: str) -> str:
        """Sanitize common JSON issues from small LLM outputs.

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
        self, data: dict, user_message: str
    ) -> dict:
        """Validate extracted data to reject nonsensical extractions.

        Small models often hallucinate data fields. This method
        ensures extracted values actually make sense.

        Args:
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

            # Field-specific validation
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
                msg_digits = re.sub(r"[^\d]", "", user_message)
                if digits not in msg_digits:
                    logger.debug("Rejecting phone: digits not in user message ('%s')", value_str)
                    continue

            elif field_name in ("project_type", "features", "tech_stack", "integrations",
                                "budget", "timeline", "mvp_or_production", "priority_features",
                                "company"):
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
    # Regex-Based Personal Data Extraction
    # ============================================

    def _regex_extract_personal_info(self, message: str) -> dict:
        """Extract personal info (name, email) from user message using regex.

        This is a fallback extraction method that doesn't depend on
        the LLM producing valid JSON. It catches common patterns like
        'my name is X' and email addresses.

        Args:
            message: The user's message text.

        Returns:
            Dictionary with extracted fields, empty if nothing found.
        """
        extracted: dict = {}

        # Extract name patterns
        name_patterns = [
            r"(?:my name is|i am|i'm|call me|name's|this is)\s+([A-Za-z][A-Za-z\s\.\-']{1,40})",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name_candidate = match.group(1).strip()
                # Clean trailing words that aren't part of the name
                name_candidate = re.split(
                    r'\b(?:and|from|at|in|with|here|please|thanks|thank)\b',
                    name_candidate, flags=re.IGNORECASE
                )[0].strip()
                if name_candidate and len(name_candidate) >= 2:
                    extracted["name"] = name_candidate
                    logger.info("Regex extracted name: %s", name_candidate)
                break

        # Extract email
        email_match = re.search(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            message
        )
        if email_match:
            extracted["email"] = email_match.group(0)
            logger.info("Regex extracted email: %s", email_match.group(0))

        return extracted
