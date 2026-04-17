"""
Email Agent — 📧 LLM-Powered Email Composer

Composes professional notification emails at stage 6:
  - Thank-you email for the user/client
  - Lead notification email for the admin/sales team

For MVP: Emails are printed to console and logged to file.
Future: Swap the send method for real SMTP (SendGrid, SES, etc.).

Design:
  - Single invocation per session (called once after summary confirmation)
  - Receives LLMProvider via constructor
  - Uses the generated summary + user data to compose personalized emails
  - Returns EmailResult with both composed emails
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from models.schemas import Session
from providers.base import LLMProvider, LLMMessage

logger = logging.getLogger(__name__)

# Email log directory (relative to backend)
EMAIL_LOG_DIR = Path(__file__).parent.parent / "email_logs"


# ============================================
# Email Result Types
# ============================================

@dataclass
class ComposedEmail:
    """A single composed email.

    Attributes:
        to: Recipient email address.
        subject: Email subject line.
        body: Full email body text.
        email_type: Either "user_thankyou" or "admin_notification".
    """

    to: str
    subject: str
    body: str
    email_type: str


@dataclass
class EmailResult:
    """Result from the Email Agent.

    Attributes:
        user_email: The thank-you email composed for the client.
        admin_email: The notification email composed for the admin.
        success: Whether email composition and mock-sending succeeded.
        message: A human-readable status message.
    """

    user_email: ComposedEmail
    admin_email: ComposedEmail
    success: bool = True
    message: str = "Emails composed and sent successfully"


# ============================================
# System Prompts
# ============================================

USER_EMAIL_SYSTEM_PROMPT = """You are composing a professional thank-you email on behalf of ColorWhistle, a software development company.

INSTRUCTIONS:
1. Write a warm, professional thank-you email to the potential client
2. Acknowledge the information they shared during the consultation
3. Briefly summarize the key project requirements
4. Let them know the team will review their requirements and follow up
5. Keep it concise (under 200 words)
6. Use a professional but warm tone

OUTPUT FORMAT:
Write ONLY the email body text. Do NOT include Subject, To, or From headers.
Do NOT wrap in markdown code blocks.
Start directly with the greeting (e.g., "Dear [Name],")
End with a professional sign-off.
"""

ADMIN_EMAIL_SYSTEM_PROMPT = """You are composing an internal lead notification email for the ColorWhistle sales/development team.

INSTRUCTIONS:
1. Write a clear, structured internal notification about a new lead
2. Include all relevant client and project details
3. Highlight key requirements and any concerns
4. Include the full lead summary
5. Add a recommendation for next steps
6. Use a professional, internal communication tone

OUTPUT FORMAT:
Write ONLY the email body text. Do NOT include Subject, To, or From headers.
Do NOT wrap in markdown code blocks.
Start with a brief header like "New Lead Notification" then list details.
"""

INTENT_SYSTEM_PROMPT = """You are an intent analysis AI for a software development agency.
Analyze the provided conversation history and collected data to determine the user's intent.

Respond with ONLY the word "HIGH" if the user shows an intent to purchase, start a project, or has a high interest in the services. 
Respond with ONLY the word "LOW" if the user is just browsing, asking general questions, spamming, or is not interested.
Do not provide any explanation, just the single word "HIGH" or "LOW".
"""


class EmailAgent:
    """LLM-powered agent that composes and mock-sends notification emails.

    Called once per session at stage 6, after the summary has been
    generated and confirmed. Composes two emails: one for the client
    and one for the admin/sales team.

    For MVP, emails are printed to console and saved to log files.

    Attributes:
        _llm: The LLM provider instance used for email composition.
        _admin_email: The admin email address for notifications.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        admin_emails: list[str] = None,
    ) -> None:
        """Initialize the Email Agent.

        Args:
            llm_provider: The LLM provider to use for email composition.
            admin_emails: List of admin email addresses for lead notifications.
        """
        self._llm = llm_provider
        self._admin_emails = admin_emails or ["admin@colorwhistle.com"]

    async def analyze_intent(self, session: Session) -> bool:
        """Analyze if the session has high interest or purchase intent."""
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in session.conversation_history])
        data_summary = session.collected_data.to_summary_dict()
        
        context = f"=== CONVERSATION HISTORY ===\n{history}\n\n=== COLLECTED DATA ===\n{data_summary}"
        
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=INTENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=context),
        ]
        
        try:
            response = await self._llm.generate(
                messages=messages,
                temperature=0.1,
                max_tokens=10,
            )
            result = response.content.strip().upper()
            return "HIGH" in result
        except Exception as e:
            logger.error("Intent analysis failed: %s", e)
            return True  # Default to true if fails so we don't drop leads

    async def compose_and_send(self, session: Session, is_early_exit: bool = False) -> EmailResult | None:
        """Compose both emails and mock-send them depending on intent and data.

        Uses the LLM to generate personalized email content, then
        logs them to console and file.

        Args:
            session: The complete session with summary and collected data.
            is_early_exit: Whether this is triggered by user exiting early.

        Returns:
            EmailResult with composed emails and status, or None if skipped.
        """
        try:
            # 1. Analyze Intent
            is_high_intent = await self.analyze_intent(session)
            if not is_high_intent:
                logger.info("Session %s determined as low intent. Skipping emails.", session.session_id)
                return EmailResult(
                    user_email=None,  # type: ignore
                    admin_email=None, # type: ignore
                    success=True,
                    message="Session finished cleanly (no emails sent due to low intent)."
                )

            # Compose admin email first (always sent for high intent)
            admin_email = await self._compose_admin_email(session)
            self._mock_send(admin_email)

            # Compose user email ONLY if user provided an email address
            user_email = None
            if session.collected_data.personal_info.email:
                user_email = await self._compose_user_email(session)
                self._mock_send(user_email)
            else:
                logger.info("No user email found for session %s. Skipping user email.", session.session_id)

            # Save to log file
            self._save_to_log(session.session_id, user_email, admin_email)

            logger.info(
                "Emails composed and sent for session %s (Early Exit: %s)",
                session.session_id, is_early_exit
            )

            if is_early_exit:
                message = "Session saved."
            elif user_email:
                message = "Thank you! We've sent a confirmation to your email and notified our team."
            else:
                message = "Thank you! Our team has been notified."

            return EmailResult(
                user_email=user_email,
                admin_email=admin_email,
                success=True,
                message=message,
            )

        except Exception as e:
            logger.error("Email Agent error: %s", e)

            # Generate fallback emails
            admin_email = self._compose_fallback_admin_email(session)
            self._mock_send(admin_email)

            user_email = None
            if session.collected_data.personal_info.email:
                user_email = self._compose_fallback_user_email(session)
                self._mock_send(user_email)

            self._save_to_log(session.session_id, user_email, admin_email)

            return EmailResult(
                user_email=user_email,
                admin_email=admin_email,
                success=True,
                message="Confirmation sent! Our team will be in touch shortly.",
            )

    async def _compose_user_email(self, session: Session) -> ComposedEmail:
        """Compose the thank-you email for the client.

        Args:
            session: The current session.

        Returns:
            ComposedEmail with the user's thank-you email.
        """
        data = session.collected_data
        user_email_address = data.personal_info.email or "client@example.com"
        user_name = data.personal_info.name or "Valued Client"

        # Build context for email composition
        context = self._build_email_context(session, email_type="user")

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=USER_EMAIL_SYSTEM_PROMPT),
            LLMMessage(role="user", content=context),
        ]

        llm_response = await self._llm.generate(
            messages=messages,
            temperature=0.5,
            max_tokens=800,
        )

        body = llm_response.content.strip()

        if not body:
            return self._compose_fallback_user_email(session)

        return ComposedEmail(
            to=user_email_address,
            subject=f"Thank You for Your Inquiry, {user_name}! — ColorWhistle",
            body=body,
            email_type="user_thankyou",
        )

    async def _compose_admin_email(self, session: Session) -> ComposedEmail:
        """Compose the lead notification email for admin.

        Args:
            session: The current session.

        Returns:
            ComposedEmail with the admin notification email.
        """
        context = self._build_email_context(session, email_type="admin")

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=ADMIN_EMAIL_SYSTEM_PROMPT),
            LLMMessage(role="user", content=context),
        ]

        llm_response = await self._llm.generate(
            messages=messages,
            temperature=0.3,
            max_tokens=1200,
        )

        body = llm_response.content.strip()

        if not body:
            return self._compose_fallback_admin_email(session)

        user_name = session.collected_data.personal_info.name or "Unknown"

        return ComposedEmail(
            to=", ".join(self._admin_emails),
            subject=f"🔔 New Lead: {user_name} — Project Inquiry",
            body=body,
            email_type="admin_notification",
        )

    def _build_email_context(
        self, session: Session, email_type: str
    ) -> str:
        """Build context for email composition.

        Args:
            session: The current session.
            email_type: Either "user" or "admin".

        Returns:
            Context string for the LLM.
        """
        data = session.collected_data
        parts: list[str] = []

        parts.append("=== CLIENT INFORMATION ===")
        parts.append(f"Name: {data.personal_info.name or 'Not provided'}")
        parts.append(f"Email: {data.personal_info.email or 'Not provided'}")

        parts.append("\n=== PROJECT DETAILS ===")
        parts.append(
            f"Project Type: {data.tech_discovery.project_type or 'Not specified'}"
        )
        parts.append(
            f"Technology: {data.tech_discovery.tech_stack or 'To be determined'}"
        )
        parts.append(
            f"Features: {data.tech_discovery.features or 'Not specified'}"
        )
        parts.append(
            f"Integrations: {data.tech_discovery.integrations or 'None'}"
        )

        parts.append("\n=== SCOPE & BUDGET ===")
        parts.append(
            f"Budget: {data.scope_pricing.budget or 'Not discussed'}"
        )
        parts.append(
            f"Timeline: {data.scope_pricing.timeline or 'Not discussed'}"
        )
        parts.append(
            f"Scope: {data.scope_pricing.mvp_or_production or 'Not specified'}"
        )

        if session.summary:
            parts.append(f"\n=== GENERATED SUMMARY ===\n{session.summary}")

        if email_type == "user":
            parts.append(
                "\n=== TASK ===\n"
                "Compose a warm thank-you email to this client. "
                "Reference their specific project briefly."
            )
        else:
            parts.append(
                "\n=== TASK ===\n"
                "Compose an internal notification email about this new lead "
                "for the development/sales team. Include all key details."
            )

        return "\n".join(parts)

    def _mock_send(self, email: ComposedEmail) -> None:
        """Mock-send an email by printing to console.

        Args:
            email: The composed email to mock-send.
        """
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"📧 MOCK EMAIL — {email.email_type.upper()}")
        print(separator)
        print(f"To: {email.to}")
        print(f"Subject: {email.subject}")
        print(f"---")
        print(email.body)
        print(f"{separator}\n")

        logger.info(
            "Mock email sent — type: %s, to: %s",
            email.email_type,
            email.to,
        )

    def _save_to_log(
        self,
        session_id: str,
        user_email: ComposedEmail,
        admin_email: ComposedEmail,
    ) -> None:
        """Save composed emails to a log file.

        Creates a timestamped log file in the email_logs directory.

        Args:
            session_id: The session identifier.
            user_email: The client's email.
            admin_email: The admin's email.
        """
        try:
            EMAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"email_{session_id}_{timestamp}.log"
            filepath = EMAIL_LOG_DIR / filename

            content_parts = [
                f"Session ID: {session_id}",
                f"Timestamp: {datetime.utcnow().isoformat()}",
                "",
                "=" * 60,
                "ADMIN NOTIFICATION EMAIL",
                "=" * 60,
                f"To: {admin_email.to}",
                f"Subject: {admin_email.subject}",
                "",
                admin_email.body,
            ]
            
            if user_email:
                content_parts.extend([
                    "",
                    "=" * 60,
                    "USER THANK-YOU EMAIL",
                    "=" * 60,
                    f"To: {user_email.to}",
                    f"Subject: {user_email.subject}",
                    "",
                    user_email.body,
                ])

            filepath.write_text("\n".join(content_parts), encoding="utf-8")

            logger.info("Email log saved: %s", filepath)

        except OSError as e:
            logger.warning("Failed to save email log: %s", e)

    def _compose_fallback_user_email(
        self, session: Session
    ) -> ComposedEmail:
        """Compose a fallback user email without LLM.

        Args:
            session: The current session.

        Returns:
            A basic thank-you email.
        """
        data = session.collected_data
        name = data.personal_info.name or "Valued Client"
        email = data.personal_info.email or "client@example.com"

        body = (
            f"Dear {name},\n\n"
            f"Thank you for taking the time to discuss your project requirements "
            f"with us at ColorWhistle.\n\n"
            f"We have received your project details and our team will review them "
            f"shortly. You can expect to hear back from us within 1-2 business days "
            f"with a preliminary assessment and proposed next steps.\n\n"
            f"If you have any additional information to share or questions in the "
            f"meantime, please don't hesitate to reach out.\n\n"
            f"Best regards,\n"
            f"The ColorWhistle Team"
        )

        return ComposedEmail(
            to=email,
            subject=f"Thank You for Your Inquiry, {name}! — ColorWhistle",
            body=body,
            email_type="user_thankyou",
        )

    def _compose_fallback_admin_email(
        self, session: Session
    ) -> ComposedEmail:
        """Compose a fallback admin email without LLM.

        Args:
            session: The current session.

        Returns:
            A basic admin notification email.
        """
        data = session.collected_data
        summary_dict = data.to_summary_dict()

        details = "\n".join(
            f"- {key}: {value}" for key, value in summary_dict.items()
        )

        body = (
            f"🔔 New Lead Notification\n\n"
            f"A new lead has completed the qualification process.\n\n"
            f"Lead Details:\n{details}\n\n"
            f"Session ID: {session.session_id}\n\n"
            f"Please review and follow up within 24 hours."
        )

        name = data.personal_info.name or "Unknown"

        return ComposedEmail(
            to=", ".join(self._admin_emails),
            subject=f"🔔 New Lead: {name} — Project Inquiry",
            body=body,
            email_type="admin_notification",
        )
