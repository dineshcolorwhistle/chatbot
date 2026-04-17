"""
Orchestrator — 🎯 Agent Coordinator & State Manager

The central hub that coordinates all agents and manages the
conversation workflow. This is the ONLY entry point for processing
user messages — routes call the orchestrator, never agents directly.

Responsibilities:
  - Manage session lifecycle (create, load, save)
  - Route messages to the appropriate agent based on current stage
  - Apply extracted data from Conversation Agent to session
  - Trigger stage transitions when data collection is complete
  - Invoke Summarization Agent at stage 5
  - Invoke Email Agent at stage 6
  - Return structured responses to routes

Design:
  - Code-driven, deterministic logic (NOT LLM-powered)
  - Receives all agents and session store via constructor
  - Stateless — all state lives in the Session model
  - Single public method: process_message()
"""

import logging
import asyncio
from datetime import datetime

from models.schemas import (
    ChatResponse,
    ConversationStage,
    Session,
    get_next_stage,
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
    "I'll walk you through a quick consultation to learn about your "
    "needs — it'll only take a few minutes. Let's get started!\n\n"
    "Could you tell me your name?"
)

COMPLETED_MESSAGE = (
    "This consultation has been completed. If you'd like to start "
    "a new consultation, please reset your session."
)


class Orchestrator:
    """Coordinates all agents and manages the conversation workflow.

    The orchestrator is the single entry point for processing user
    messages. It determines which agent to invoke based on the current
    conversation stage, applies extracted data, manages stage
    transitions, and returns structured responses.

    This is code-driven logic — NOT LLM-powered. The orchestrator
    makes deterministic decisions about workflow control.

    Attributes:
        _conversation_agent: Handles dialogue in stages 1-4.
        _summarization_agent: Generates summaries at stage 5.
        _email_agent: Composes emails at stage 6.
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

        logger.info("Orchestrator initialized with all agents")

    async def process_message(
        self, session_id: str, message: str
    ) -> ChatResponse:
        """Process a user message through the appropriate agent.

        This is the main entry point called by API routes.
        It handles the complete lifecycle:
          1. Load or create session
          2. Route to the correct agent
          3. Apply extracted data
          4. Handle stage transitions
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

        # 2. Handle new sessions with welcome
        if is_new:
            return await self._handle_welcome(session, message)

        # 3. Route based on current stage
        stage = session.stage

        if stage == ConversationStage.WELCOME:
            return await self._handle_welcome(session, message)

        elif stage in (
            ConversationStage.PERSONAL_INFO,
            ConversationStage.TECH_DISCOVERY,
            ConversationStage.SCOPE_PRICING,
        ):
            return await self._handle_conversation(session, message)

        elif stage == ConversationStage.SUMMARY:
            return await self._handle_summary(session, message)

        elif stage == ConversationStage.EMAIL:
            return await self._handle_email(session, message)

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

        For brand-new sessions, sends the welcome message.
        For returning users who already got a welcome, transitions
        to personal info and processes their message.

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

            # Advance to personal info stage
            session.stage = ConversationStage.PERSONAL_INFO

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

        # Already welcomed — process through conversation
        session.stage = ConversationStage.PERSONAL_INFO
        return await self._handle_conversation(session, message)

    async def _handle_conversation(
        self, session: Session, message: str
    ) -> ChatResponse:
        """Handle conversation stages (personal info, tech discovery, scope).

        Routes the message through the Conversation Agent, applies
        extracted data, and handles stage transitions.

        Args:
            session: The current session.
            message: The user's message.

        Returns:
            ChatResponse with the agent's reply and updated state.
        """
        # Add user message to history
        session.add_message("user", message)

        # Process through Conversation Agent
        result = await self._conversation_agent.process_message(
            session, message
        )

        # Apply extracted data to session
        self._apply_extracted_data(session, result.extracted_data)

        # Check for stage advancement
        if result.should_advance:
            next_stage = get_next_stage(session.stage)
            if next_stage:
                logger.info(
                    "Stage transition: %s → %s (session: %s)",
                    session.stage.value,
                    next_stage.value,
                    session.session_id,
                )
                session.stage = next_stage

                # If we just entered summary stage, generate summary automatically
                if next_stage == ConversationStage.SUMMARY:
                    session.add_message("assistant", result.reply)
                    await self._session_store.save(session)
                    return await self._handle_summary(session, "")

        # Save and return
        session.add_message("assistant", result.reply)
        await self._session_store.save(session)

        return ChatResponse(
            reply=result.reply,
            stage=session.stage,
            data_collected=session.collected_data.to_summary_dict(),
        )

    async def _handle_summary(
        self, session: Session, message: str
    ) -> ChatResponse:
        """Handle the summary stage.

        If summary hasn't been generated yet, generates it.
        If summary exists, the user is confirming or requesting changes.

        Args:
            session: The current session.
            message: The user's message (empty for auto-trigger).

        Returns:
            ChatResponse with the summary or confirmation handling.
        """
        if not session.summary:
            # Generate summary
            logger.info(
                "Generating summary for session %s", session.session_id
            )
            summary = await self._summarization_agent.generate_summary(session)
            session.summary = summary

            reply = (
                "Great! I've gathered all the information I need. "
                "Here's a summary of your project requirements:\n\n"
                f"{summary}\n\n"
                "Does this look correct? If you'd like to make any changes, "
                "just let me know. Otherwise, type **'confirm'** or **'yes'** "
                "to proceed."
            )

            session.add_message("assistant", reply)
            await self._session_store.save(session)

            return ChatResponse(
                reply=reply,
                stage=session.stage,
                data_collected=session.collected_data.to_summary_dict(),
            )

        # Summary already exists — check for confirmation
        if message:
            session.add_message("user", message)

        confirmation_words = {"yes", "confirm", "correct", "looks good", "ok", "okay", "sure", "proceed", "go ahead", "perfect", "that's right", "thats right"}
        msg_lower = message.strip().lower()

        if any(word in msg_lower for word in confirmation_words):
            # User confirmed — advance to email stage
            session.stage = ConversationStage.EMAIL

            logger.info(
                "Summary confirmed — advancing to email stage (session: %s)",
                session.session_id,
            )

            # Auto-trigger email composition
            return await self._handle_email(session, "")

        else:
            # User wants changes — let them know
            reply = (
                "I understand you'd like to make some adjustments. "
                "Could you tell me specifically what needs to be changed? "
                "Once we've made the updates, I'll regenerate the summary."
            )

            # For now, regenerate the summary on next message
            # (In a more complex version, we'd parse what to change)
            session.summary = None
            session.add_message("assistant", reply)
            await self._session_store.save(session)

            return ChatResponse(
                reply=reply,
                stage=session.stage,
                data_collected=session.collected_data.to_summary_dict(),
            )

    async def _handle_email(
        self, session: Session, message: str
    ) -> ChatResponse:
        """Handle the email stage.

        Triggers the Email Agent to compose and mock-send emails,
        then transitions to the completed stage.

        Args:
            session: The current session.
            message: The user's message (empty for auto-trigger).

        Returns:
            ChatResponse with the completion message.
        """
        logger.info(
            "Composing and sending emails for session %s",
            session.session_id,
        )

        email_result = await self._email_agent.compose_and_send(session)

        reply_parts = [email_result.message]

        if email_result.user_email:
            reply_parts.append("📧 A confirmation email has been sent to your address, and our team has been notified about your project requirements.")
        elif email_result.admin_email:
            reply_parts.append("Our team has been notified about your project requirements.")

        reply_parts.append("Thank you for your time! We look forward to working with you. If you have any questions in the meantime, feel free to reach out.\n\nHave a wonderful day! 👋")

        reply = "\n\n".join(reply_parts)

        # Transition to completed
        session.stage = ConversationStage.COMPLETED
        session.add_message("assistant", reply)
        await self._session_store.save(session)

        return ChatResponse(
            reply=reply,
            stage=session.stage,
            data_collected=session.collected_data.to_summary_dict(),
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

        Maps field names to the appropriate collected data model
        based on the current conversation stage.

        Args:
            session: The session to update.
            extracted_data: Dictionary of field-value pairs to apply.
        """
        if not extracted_data:
            return

        stage = session.stage
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
            
        asyncio.create_task(self._process_early_exit(session))
        return True
        
    async def _process_early_exit(self, session: Session) -> None:
        """Background task to analyze intent and send emails on early exit."""
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
