"""
Summarization Agent — 📋 LLM-Powered Summary Generator

Generates structured lead summaries at stage 5 of the conversation.
Takes all collected data and conversation history to produce a clean,
professional summary of the lead's requirements.

Design:
  - Single invocation per session (called once when all data is collected)
  - Receives LLMProvider via constructor
  - Returns a formatted summary string
  - Uses conversation history for context enrichment beyond raw fields
"""

import logging

from models.schemas import Session
from providers.base import LLMProvider, LLMMessage

logger = logging.getLogger(__name__)


# ============================================
# System Prompt
# ============================================

SUMMARIZATION_SYSTEM_PROMPT = """You are a professional project analyst. Your task is to generate a clean, dynamic summary of a potential client's project requirements based on the conversation history.

INSTRUCTIONS:
1. Create a professional, well-organized summary that flows naturally and is directly based on the user's conversation.
2. Only include details that the user explicitly provided or discussed. DO NOT use generic placeholders like "Not specified" or "To be determined". If a detail wasn't discussed, simply omit it.
3. Organize the summary with appropriate descriptive headers (e.g., Client Info, Project Details, Scope, etc.) tailored to what was actually discussed.
4. Highlight key requirements, priorities, and any notable observations or context from the conversation.
5. Improvise the structure so it fits the unique conversation, rather than forcing a rigid template.
6. Keep the language professional, concise, and easy to read.

---
Important: Output ONLY the summary text. Do not include any JSON, markdown code blocks, or extra formatting wrappers.
"""


class SummarizationAgent:
    """LLM-powered agent that generates structured lead summaries.

    Called once per session at stage 5, after all data collection
    stages are complete. Uses both the structured collected data
    and the conversation history for maximum context.

    Attributes:
        _llm: The LLM provider instance used for generation.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the Summarization Agent.

        Args:
            llm_provider: The LLM provider to use for summary generation.
        """
        self._llm = llm_provider

    async def generate_summary(self, session: Session) -> str:
        """Generate a structured lead summary from session data.

        Combines the collected data fields and conversation history
        into a prompt for the LLM, which produces a formatted summary.

        Args:
            session: The complete session with collected data and history.

        Returns:
            A formatted summary string ready for display and email inclusion.

        Raises:
            RuntimeError: If the LLM fails to generate a summary.
        """
        # Build the context for summarization
        context = self._build_summary_context(session)

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=SUMMARIZATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=context),
        ]

        try:
            llm_response = await self._llm.generate(
                messages=messages,
                temperature=0.3,  # Lower temp for more consistent summaries
                max_tokens=1500,
            )

            summary = llm_response.content.strip()

            if not summary:
                logger.error("Summarization Agent returned empty response")
                summary = self._generate_fallback_summary(session)

            logger.info(
                "Summary generated for session %s (%d chars)",
                session.session_id,
                len(summary),
            )

            return summary

        except ConnectionError as e:
            logger.error("Summarization Agent connection error: %s", e)
            return self._generate_fallback_summary(session)

        except RuntimeError as e:
            logger.error("Summarization Agent LLM error: %s", e)
            return self._generate_fallback_summary(session)

    def _build_summary_context(self, session: Session) -> str:
        """Build the context prompt with all collected data and relevant history.

        Args:
            session: The current session.

        Returns:
            A formatted context string for the LLM.
        """
        parts: list[str] = []

        # Add structured collected data
        parts.append("=== COLLECTED DATA ===")

        data = session.collected_data
        summary_dict = data.to_summary_dict()

        if summary_dict:
            for key, value in summary_dict.items():
                parts.append(f"{key}: {value}")
        else:
            parts.append("(No structured data collected)")

        # Add conversation history for context enrichment
        parts.append("\n=== CONVERSATION HISTORY ===")

        history = session.get_history_for_llm()
        if history:
            for msg in history:
                role_label = "Client" if msg["role"] == "user" else "Consultant"
                parts.append(f"{role_label}: {msg['content']}")
        else:
            parts.append("(No conversation history)")

        parts.append(
            "\n=== TASK ===\n"
            "Based on the data above, generate a comprehensive lead summary. "
            "Extract any additional insights from the conversation that aren't "
            "captured in the structured fields."
        )

        return "\n".join(parts)

    def _generate_fallback_summary(self, session: Session) -> str:
        """Generate a basic summary without LLM if the LLM is unavailable.

        Uses the structured data fields directly to build a simple
        but functional summary.

        Args:
            session: The current session.

        Returns:
            A basic formatted summary string.
        """
        logger.warning("Using fallback summary generation (no LLM)")

        data = session.collected_data
        pi = data.personal_info
        td = data.tech_discovery
        sp = data.scope_pricing

        summary_lines = [
            "📋 **Lead Summary**",
            "",
        ]
        
        if pi.name or pi.email:
            summary_lines.append("**Client Information:**")
            if pi.name: summary_lines.append(f"- Name: {pi.name}")
            if pi.email: summary_lines.append(f"- Email: {pi.email}")
            summary_lines.append("")
            
        if td.project_type or td.tech_stack or td.features or td.integrations:
            summary_lines.append("**Project Details:**")
            if td.project_type: summary_lines.append(f"- Type: {td.project_type}")
            if td.tech_stack: summary_lines.append(f"- Technology: {td.tech_stack}")
            if td.features: summary_lines.append(f"- Key Features: {td.features}")
            if td.integrations: summary_lines.append(f"- Integrations: {td.integrations}")
            summary_lines.append("")
            
        if sp.budget or sp.timeline or sp.mvp_or_production or sp.priority_features:
            summary_lines.append("**Scope & Budget:**")
            if sp.budget: summary_lines.append(f"- Budget Range: {sp.budget}")
            if sp.timeline: summary_lines.append(f"- Timeline: {sp.timeline}")
            if sp.mvp_or_production: summary_lines.append(f"- Project Scope: {sp.mvp_or_production}")
            if sp.priority_features: summary_lines.append(f"- Priority Features: {sp.priority_features}")
            summary_lines.append("")

        summary_lines.extend([
            "**Observations:**",
            "- Summary generated using fallback mode (LLM unavailable)",
            "- Please review the conversation history for additional context"
        ])

        return "\n".join(summary_lines)
