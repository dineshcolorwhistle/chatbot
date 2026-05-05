"""
Orchestrator — 🎯 Agent Coordinator & State Manager

The central hub that coordinates all agents and manages the
conversation workflow. This is the ONLY entry point for processing
user messages — routes call the orchestrator, never agents directly.

Responsibilities:
  - Manage session lifecycle (create, load, save)
  - Route messages to the Conversation Agent (single-block approach)
  - Apply extracted data from Conversation Agent to session
  - Track user message count and enforce limits
  - Trigger Summarization Agent and Email Agent at completion
  - Return structured responses to routes

Design:
  - Code-driven, deterministic logic (NOT LLM-powered)
  - Receives all agents and session store via constructor
  - Stateless — all state lives in the Session model
  - Single public method: process_message()
"""

import logging
import asyncio
import re
from datetime import datetime

from models.schemas import (
    ChatResponse,
    ConversationStage,
    Session,
)
from providers.base import LLMProvider
from services.conversation_agent import ConversationAgent
from services.summarization_agent import SummarizationAgent
from services.email_agent import EmailAgent
from services.session_store import BaseSessionStore
from config import app_config

logger = logging.getLogger(__name__)


# ============================================
# Welcome Message
# ============================================

WELCOME_MESSAGE = (
    "👋 Hello! Welcome to ColorWhistle. I'm your project consultant, "
    "and I'm here to help understand your project requirements.\n\n"
    "How can I help you today?"
)

COMPLETED_MESSAGE = (
    "This consultation has been completed. If you'd like to start "
    "a new consultation, please reset your session."
)


class Orchestrator:
    """Coordinates all agents and manages the conversation workflow.

    The orchestrator is the single entry point for processing user
    messages. It routes all messages through the Conversation Agent
    (single-block approach), applies extracted data, tracks message
    limits, and triggers summary/email at completion.

    This is code-driven logic — NOT LLM-powered. The orchestrator
    makes deterministic decisions about workflow control.

    Attributes:
        _conversation_agent: Handles all dialogue.
        _summarization_agent: Generates summaries at completion.
        _email_agent: Composes and sends emails at completion.
        _session_store: Persistence layer for sessions.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        session_store: BaseSessionStore,
        knowledge_base=None,
    ) -> None:
        """Initialize the Orchestrator with all agents.

        Creates agent instances using the shared LLM provider.
        Agents are lightweight and stateless, so creating them
        here is efficient.

        Args:
            llm_provider: Shared LLM provider for all agents.
            session_store: Session persistence backend.
            knowledge_base: Optional KnowledgeBase for RAG context.
        """
        self._conversation_agent = ConversationAgent(llm_provider, knowledge_base=knowledge_base)
        self._summarization_agent = SummarizationAgent(llm_provider)
        self._email_agent = EmailAgent(
            llm_provider, 
            admin_emails=app_config.admin_emails
        )
        self._session_store = session_store
        self._knowledge_base = knowledge_base
        self._bg_tasks = set()

        logger.info("Orchestrator initialized with all agents")

    def _run_background_task(self, coro):
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def process_message(
        self, session_id: str, message: str
    ) -> ChatResponse:
        """Process a user message through the appropriate handler.

        This is the main entry point called by API routes.
        It handles the complete lifecycle:
          1. Load or create session
          2. Route to the correct handler
          3. Apply extracted data
          4. Track message limits
          5. Save session
          6. Return response

        Args:
            session_id: The user's session identifier.
            message: The user's message text.

        Returns:
            ChatResponse with the assistant's reply, current stage,
            and collected data snapshot.
        """
        # 1. Load or create session
        session, is_new = await self._session_store.get_or_create(session_id)

        logger.info(
            "Processing message — session: %s, stage: %s, new: %s",
            session_id,
            session.stage.value,
            is_new,
        )

        # 2. Route based on current stage
        stage = session.stage

        if is_new or stage == ConversationStage.WELCOME:
            return await self._handle_welcome(session, message)

        elif stage == ConversationStage.CONVERSATION:
            return await self._handle_conversation(session, message)

        elif stage == ConversationStage.LIMIT_WARNING:
            return await self._handle_limit_warning(session, message)

        elif stage == ConversationStage.FINAL_INPUT:
            return await self._handle_final_input(session, message)

        elif stage == ConversationStage.COMPLETED:
            return await self._handle_completed(session)

        else:
            logger.error("Unknown stage: %s", stage)
            return ChatResponse(
                reply="Something went wrong. Please try resetting your session.",
                stage=session.stage,
                data_collected=session.collected_data.to_summary_dict(),
            )

    # ============================================
    # Stage Handlers
    # ============================================

    async def _handle_welcome(
        self, session: Session, message: str
    ) -> ChatResponse:
        """Handle the welcome stage.

        For brand-new sessions, sends the welcome message and
        transitions to the CONVERSATION stage. Also processes
        the user's first message through the conversation agent.

        Args:
            session: The current session.
            message: The user's message.

        Returns:
            ChatResponse with welcome or first conversation reply.
        """
        if not session.conversation_history:
            # Brand new session — send welcome and transition
            session.add_message("assistant", WELCOME_MESSAGE)
            session.add_message("user", message)

            # Transition to single conversation stage
            session.stage = ConversationStage.CONVERSATION

            # Process the user's first message through conversation agent
            result = await self._conversation_agent.process_message(
                session, message
            )

            # Apply any extracted data
            self._apply_extracted_data(session, result.extracted_data)

            session.add_message("assistant", result.reply)
            await self._session_store.save(session)

            return ChatResponse(
                reply=result.reply,
                stage=session.stage,
                data_collected=session.collected_data.to_summary_dict(),
            )

        # Already welcomed — transition and process
        session.stage = ConversationStage.CONVERSATION
        return await self._handle_conversation(session, message)

    async def _handle_conversation(
        self, session: Session, message: str
    ) -> ChatResponse:
        """Handle the main conversation flow (single block).

        Routes the message through the Conversation Agent, applies
        extracted data, and checks message limits.

        At the penultimate message (one before max), appends a
        polite one-time request for contact details (name, email,
        company, location) so the user can provide them before
        the session ends.

        Args:
            session: The current session.
            message: The user's message.

        Returns:
            ChatResponse with the agent's reply and updated state.
        """
        # Count user messages BEFORE adding this new one
        user_msg_count = sum(1 for m in session.conversation_history if m.role == "user")
        
        # Add user message to history
        session.add_message("user", message)

        # Process through Conversation Agent
        result = await self._conversation_agent.process_message(
            session, message
        )

        # Apply extracted data to session
        self._apply_extracted_data(session, result.extracted_data)

        # Check if user reached the message limit
        # user_msg_count was count BEFORE this message, so +1 = current total
        current_msg_number = user_msg_count + 1

        if current_msg_number >= app_config.max_user_messages:
            # Max messages reached — transition to limit warning
            # Include the agent's response so the user's message is still answered
            session.stage = ConversationStage.LIMIT_WARNING

            # Retroactive scan: before claiming data is missing, scan ALL
            # user messages in history for names/emails that may have been
            # missed by the LLM extraction in earlier turns.
            self._retroactive_extract_from_history(session)

            pi = session.collected_data.personal_info
            has_name = bool(pi.name)
            has_email = bool(pi.email)

            limit_notice = (
                "You've reached the maximum number of interactions for this session. "
                "Our team will review your request and get back to you shortly.\n\n"
            )

            if not has_name or not has_email:
                missing_parts = []
                if not has_name:
                    missing_parts.append("name")
                if not has_email:
                    missing_parts.append("email")
                missing_str = " and ".join(missing_parts)
                limit_notice += (
                    f"We noticed you haven't shared your {missing_str} yet. "
                    f"Would you like to provide your details in a single message "
                    f"so we can get in touch? (Yes/No)"
                )
            else:
                limit_notice += (
                    f"Would you like to provide any remaining details in a "
                    f"single message before we wrap up? (Yes/No)"
                )

            # Prepend the agent's actual answer so the user's question isn't ignored
            reply = f"{result.reply}\n\n{limit_notice}"

            session.add_message("assistant", reply)
            await self._session_store.save(session)
            return ChatResponse(
                reply=reply,
                stage=session.stage,
                data_collected=session.collected_data.to_summary_dict(),
            )

        # Penultimate message — append contact details request
        reply = result.reply
        if current_msg_number == app_config.max_user_messages - 1:
            contact_request = self._conversation_agent.build_contact_details_request(session)
            if contact_request:
                reply = f"{reply}\n\n{contact_request}"
                logger.info(
                    "Appended contact details request at message %d/%d",
                    current_msg_number,
                    app_config.max_user_messages,
                )

        # Save and return normal conversation reply
        session.add_message("assistant", reply)
        await self._session_store.save(session)

        return ChatResponse(
            reply=reply,
            stage=session.stage,
            data_collected=session.collected_data.to_summary_dict(),
        )

    async def _handle_limit_warning(self, session: Session, message: str) -> ChatResponse:
        """Handle user's response to the limit reached warning.

        Accepts three types of responses:
        1. Explicit "Yes" — transitions to FINAL_INPUT to collect remaining data.
        2. Explicit "No" — completes session immediately.
        3. Direct data provision (e.g. "Dinesh, dinesh@gmail.com") — treated as
           implicit Yes. Extracts the data directly from this message and completes
           the session with a thank-you, skipping the FINAL_INPUT round-trip.
        """
        session.add_message("user", message)

        msg_lower = message.strip().lower()
        is_yes = any(msg_lower.startswith(w) or msg_lower == w for w in ["yes", "y", "sure", "ok", "okay", "yeah", "yep"])
        is_no = any(msg_lower.startswith(w) or msg_lower == w for w in ["no", "n", "nope", "cancel", "nah"])

        # Detect if the user is directly providing contact data instead of Yes/No.
        # An email address in the message is a strong signal.
        has_email_in_msg = bool(re.search(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', message
        ))

        if not is_yes and not is_no and has_email_in_msg:
            # User answered with their contact details directly — treat as implicit Yes + data.
            logger.info(
                "Limit warning: user provided contact data directly (implicit Yes) — "
                "extracting and completing session."
            )
            # Extract name/email from this message via retroactive scan
            self._retroactive_extract_from_history(session)

            session.stage = ConversationStage.COMPLETED
            reply = (
                "Thank you for sharing your details! Our team will review your requirements "
                "and get back to you shortly. 👋"
            )
            session.add_message("assistant", reply)
            await self._session_store.save(session)

            # Trigger summary and email
            self._run_background_task(self._process_early_exit(session))

        elif is_yes and not is_no:
            session.stage = ConversationStage.FINAL_INPUT
            # Retroactive scan before checking what's missing
            self._retroactive_extract_from_history(session)
            pi = session.collected_data.personal_info
            missing = []
            if not pi.name:
                missing.append("name")
            if not pi.email:
                missing.append("email address")
            if missing:
                missing_str = ", ".join(missing)
                reply = f"Please share your {missing_str} along with any final details in a single message."
            else:
                reply = "Please provide your complete requirements in a single message."
            session.add_message("assistant", reply)
            await self._session_store.save(session)

        else:
            # Explicit "No" or unrecognised response — wrap up gracefully.
            session.stage = ConversationStage.COMPLETED
            reply = "Thank you for chatting with us! Our team will follow up with you shortly. 👋"
            session.add_message("assistant", reply)
            await self._session_store.save(session)

            # Auto-trigger early exit process to send summary and email
            self._run_background_task(self._process_early_exit(session))

        return ChatResponse(
            reply=reply,
            stage=session.stage,
            data_collected=session.collected_data.to_summary_dict()
        )

    async def _handle_final_input(self, session: Session, message: str) -> ChatResponse:
        """Handle the single final message from the user."""
        session.add_message("user", message)
        
        # Try to extract data from this final message
        result = await self._conversation_agent.process_message(session, message)
        self._apply_extracted_data(session, result.extracted_data)
        
        session.stage = ConversationStage.COMPLETED
        reply = "Thank you for sharing! Our team will review everything and follow up with you shortly. 👋"
        session.add_message("assistant", reply)
        await self._session_store.save(session)
        
        # Auto-trigger early exit process to send summary and email
        self._run_background_task(self._process_early_exit(session))
        
        return ChatResponse(
            reply=reply,
            stage=session.stage,
            data_collected=session.collected_data.to_summary_dict()
        )

    async def _handle_completed(self, session: Session) -> ChatResponse:
        """Handle messages after the conversation is complete.

        Args:
            session: The current session.

        Returns:
            ChatResponse directing user to reset.
        """
        return ChatResponse(
            reply=COMPLETED_MESSAGE,
            stage=session.stage,
            data_collected=session.collected_data.to_summary_dict(),
        )

    # ============================================
    # Data Management
    # ============================================

    def _apply_extracted_data(
        self, session: Session, extracted_data: dict
    ) -> None:
        """Apply extracted data from the Conversation Agent to the session.

        Maps field names to the appropriate collected data model.
        All fields are extracted in any message — no stage gating.

        Args:
            session: The session to update.
            extracted_data: Dictionary of field-value pairs to apply.
        """
        if not extracted_data:
            return

        data = session.collected_data

        for field, value in extracted_data.items():
            if not value or not str(value).strip():
                continue

            value_str = str(value).strip()

            # Personal info fields
            if field == "name" and not data.personal_info.name:
                data.personal_info.name = value_str
                logger.info("Extracted name: %s", value_str)
            elif field == "email" and not data.personal_info.email:
                data.personal_info.email = value_str
                logger.info("Extracted email: %s", value_str)
            elif field == "phone" and not data.personal_info.phone:
                data.personal_info.phone = value_str
                logger.info("Extracted phone: %s", value_str)
            elif field == "company" and not data.personal_info.company:
                data.personal_info.company = value_str
                logger.info("Extracted company: %s", value_str)
            elif field == "location" and not data.personal_info.location:
                data.personal_info.location = value_str
                logger.info("Extracted location: %s", value_str)

            # Tech discovery fields
            elif field == "project_type" and not data.tech_discovery.project_type:
                data.tech_discovery.project_type = value_str
                logger.info("Extracted project_type: %s", value_str)
            elif field == "tech_stack" and not data.tech_discovery.tech_stack:
                data.tech_discovery.tech_stack = value_str
                logger.info("Extracted tech_stack: %s", value_str)
            elif field == "features" and not data.tech_discovery.features:
                data.tech_discovery.features = value_str
                logger.info("Extracted features: %s", value_str)
            elif field == "integrations" and not data.tech_discovery.integrations:
                data.tech_discovery.integrations = value_str
                logger.info("Extracted integrations: %s", value_str)

            # Scope & pricing fields
            elif field == "budget" and not data.scope_pricing.budget:
                data.scope_pricing.budget = value_str
                logger.info("Extracted budget: %s", value_str)
            elif field == "timeline" and not data.scope_pricing.timeline:
                data.scope_pricing.timeline = value_str
                logger.info("Extracted timeline: %s", value_str)
            elif field == "mvp_or_production" and not data.scope_pricing.mvp_or_production:
                data.scope_pricing.mvp_or_production = value_str
                logger.info("Extracted mvp_or_production: %s", value_str)
            elif field == "priority_features" and not data.scope_pricing.priority_features:
                data.scope_pricing.priority_features = value_str
                logger.info("Extracted priority_features: %s", value_str)

        session.updated_at = datetime.utcnow()

    def _retroactive_extract_from_history(self, session: Session) -> None:
        """Scan ALL user messages in history for names and emails.

        This is a safety net used before displaying messages that claim
        the user hasn't shared certain data. If the LLM extraction
        missed a name or email from a previous turn, this regex-based
        scan catches it.

        Only fills in fields that are still empty — never overwrites
        previously extracted values.

        Args:
            session: The session whose history to scan.
        """
        pi = session.collected_data.personal_info

        # Only scan if we're still missing name or email
        if pi.name and pi.email:
            return

        # Gather all user messages
        user_messages = [
            msg.content for msg in session.conversation_history
            if msg.role == "user"
        ]

        for msg_text in user_messages:
            # Extract name if missing
            if not pi.name:
                name_patterns = [
                    r"(?:my name is|i am|i'm|call me|name's|this is)\s+([A-Za-z][A-Za-z\s\.\-']{1,40})",
                ]
                for pattern in name_patterns:
                    match = re.search(pattern, msg_text, re.IGNORECASE)
                    if match:
                        name_candidate = match.group(1).strip()
                        # Clean trailing words that aren't part of the name
                        name_candidate = re.split(
                            r'\b(?:and|from|at|in|with|here|please|thanks|thank|looking|need|want|interested)\b',
                            name_candidate, flags=re.IGNORECASE
                        )[0].strip()
                        if name_candidate and len(name_candidate) >= 2:
                            pi.name = name_candidate
                            logger.info(
                                "Retroactive scan extracted name: %s",
                                name_candidate,
                            )
                            break

            # Extract email if missing
            if not pi.email:
                email_match = re.search(
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    msg_text,
                )
                if email_match:
                    pi.email = email_match.group(0)
                    logger.info(
                        "Retroactive scan extracted email: %s",
                        pi.email,
                    )

            # Stop scanning if we have everything
            if pi.name and pi.email:
                break

        if pi.name or pi.email:
            session.updated_at = datetime.utcnow()

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID (for API introspection).

        Args:
            session_id: The session identifier.

        Returns:
            The Session object if found, None otherwise.
        """
        return await self._session_store.get(session_id)

    async def trigger_early_exit(self, session_id: str) -> bool:
        """Trigger background early exit processing for a session."""
        session = await self._session_store.get(session_id)
        if not session or session.stage in (ConversationStage.WELCOME, ConversationStage.COMPLETED):
            return False
            
        self._run_background_task(self._process_early_exit(session))
        return True
        
    async def _process_early_exit(self, session: Session) -> None:
        """Background task to generate summary and send emails on early exit."""
        logger.info("Processing early exit for session %s", session.session_id)
        
        if not session.summary and len(session.conversation_history) > 2:
            try:
                session.summary = await self._summarization_agent.generate_summary(session)
            except Exception as e:
                logger.warning("Failed to generate summary during early exit: %s", e)
                
        await self._email_agent.compose_and_send(session, is_early_exit=True)

    async def reset_session(self, session_id: str) -> bool:
        """Reset a session by deleting it.

        Args:
            session_id: The session to reset.

        Returns:
            True if the session was found and deleted.
        """
        # Trigger early exit logic if the session was active and abandoned
        await self.trigger_early_exit(session_id)

        deleted = await self._session_store.delete(session_id)
        if deleted:
            logger.info("Session reset: %s", session_id)
        return deleted
