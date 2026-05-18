"""
Daily Summary Scheduler — 📊 Automated Conversation Reports

Generates and emails daily PDF reports of widget conversations,
grouped by namespace. Each namespace gets its own PDF attachment.

Report format per conversation entry:
  - Client Name
  - Client Email
  - Client Company (if available)
  - Conversation transcript
  - Intent (HIGH / LOW — inferred by LLM)

The job runs on a configurable cron schedule (DAILY_SUMMARY_CRON env var)
and can also be triggered manually via the admin API.

Design:
  - Business logic is fully decoupled from the scheduler trigger
  - Uses ReportLab for PDF generation (no system dependencies)
  - Reuses existing EmailAgent.analyze_intent() for intent classification
  - Sends emails via the existing SMTP / console-log pipeline
"""

import logging
import smtplib
import textwrap
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import app_config
from models.schemas import Session

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# ============================================
# PDF Generation
# ============================================

def _build_namespace_pdf(
    namespace: str,
    sessions: list[Session],
    intents: dict[str, str],
    summaries: dict[str, str],
    report_date: str,
) -> bytes:
    """Generate a PDF report for a single namespace.

    Args:
        namespace: The namespace identifier (e.g., 'eduwhistle').
        sessions: List of sessions belonging to this namespace.
        intents: Mapping of session_id -> intent string ('HIGH' / 'LOW').
        summaries: Mapping of session_id -> summary string.
        report_date: Human-readable date string for the report header.

    Returns:
        Raw PDF bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    # Modern styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=24,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=20,
        spaceAfter=12,
        borderPadding=(8, 12, 8, 12),
        backColor=colors.HexColor("#f1f5f9"),
    )
    label_style = ParagraphStyle(
        "FieldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#475569"),
    )
    value_style = ParagraphStyle(
        "FieldValue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#334155"),
        leading=14,
    )
    conversation_style = ParagraphStyle(
        "ConversationText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#1e293b"),
        leading=16,
        borderPadding=(12, 12, 12, 12),
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=1,
    )
    intent_high_style = ParagraphStyle(
        "IntentHigh",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#059669"),
    )
    intent_low_style = ParagraphStyle(
        "IntentLow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#dc2626"),
    )

    elements: list = []

    # Title
    elements.append(Paragraph(
        f"Daily Conversation Summary — {namespace.title()}",
        title_style,
    ))
    elements.append(Paragraph(
        f"Report Date: {report_date} &nbsp;|&nbsp; Total Conversations: {len(sessions)}",
        subtitle_style,
    ))

    # Separator
    separator_data = [["" ]]
    separator_table = Table(separator_data, colWidths=[doc.width])
    separator_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#e2e8f0")),
    ]))
    elements.append(separator_table)
    elements.append(Spacer(1, 12))

    for idx, session in enumerate(sessions, 1):
        pi = session.collected_data.personal_info
        client_name = pi.name or "Not provided"
        client_email = pi.email or "Not provided"
        client_company = pi.company if pi.company else None
        intent = intents.get(session.session_id, "N/A")

        # Section heading
        elements.append(Paragraph(
            f"Conversation #{idx} — {client_name}",
            section_heading,
        ))

        # Build info table
        info_data = [
            [Paragraph("Client Name:", label_style), Paragraph(client_name, value_style)],
            [Paragraph("Client Email:", label_style), Paragraph(client_email, value_style)],
        ]
        if client_company:
            info_data.append([
                Paragraph("Client Company:", label_style),
                Paragraph(client_company, value_style),
            ])

        # Intent row
        intent_style = intent_high_style if intent == "HIGH" else intent_low_style
        info_data.append([
            Paragraph("Intent:", label_style),
            Paragraph(f"● {intent}", intent_style),
        ])

        info_table = Table(info_data, colWidths=[100, doc.width - 120])
        info_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 12))

        # Conversation transcript
        elements.append(Paragraph("Conversation Summary:", label_style))
        elements.append(Spacer(1, 16))
        
        summary_text = summaries.get(session.session_id, "No summary available.")
        safe_summary = (
            summary_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        wrapped_html = safe_summary.replace("\n", "<br/>")
        elements.append(Paragraph(wrapped_html, conversation_style))
        elements.append(Spacer(1, 16))

        # Entry separator
        elements.append(Spacer(1, 6))
        sep_data = [["" ]]
        sep_table = Table(sep_data, colWidths=[doc.width])
        sep_table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#cccccc")),
        ]))
        elements.append(sep_table)
        elements.append(Spacer(1, 8))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ============================================
# Email Delivery
# ============================================

def _send_summary_email(
    namespace: str,
    pdf_bytes: bytes,
    report_date: str,
    session_count: int,
) -> None:
    """Send (or mock-send) the daily summary email with PDF attachment.

    Args:
        namespace: Namespace identifier for the report.
        pdf_bytes: Raw PDF bytes to attach.
        report_date: Human-readable date for the subject line.
        session_count: Number of conversations in the report.
    """
    admin_emails = app_config.admin_emails
    to_address = ", ".join(admin_emails)
    subject = f"📊 Daily Summary — {namespace.title()} — {report_date} ({session_count} conversations)"
    filename = f"daily_summary_{namespace}_{report_date.replace(' ', '_').replace(',', '')}.pdf"

    body = (
        f"Hi,\n\n"
        f"Please find attached the daily conversation summary for namespace "
        f"\"{namespace}\" covering {report_date}.\n\n"
        f"Total widget conversations: {session_count}\n\n"
        f"This is an automated report generated by the chatbot system.\n\n"
        f"Regards,\n"
        f"CW Chatbot System"
    )

    # Always log to console
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"📊 DAILY SUMMARY EMAIL — {namespace.upper()}")
    print(separator)
    print(f"To: {to_address}")
    print(f"Subject: {subject}")
    print(f"Attachment: {filename} ({len(pdf_bytes)} bytes)")
    print(f"---")
    print(body)
    print(f"{separator}\n")

    # Try SMTP delivery
    if not app_config.smtp_host or not app_config.smtp_user:
        logger.info(
            "SMTP not configured. Daily summary email logged to console — namespace: %s",
            namespace,
        )
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = app_config.smtp_from
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach PDF
        pdf_attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        pdf_attachment.add_header(
            "Content-Disposition", "attachment", filename=filename
        )
        msg.attach(pdf_attachment)

        server = smtplib.SMTP(app_config.smtp_host, app_config.smtp_port)
        server.starttls()
        server.login(app_config.smtp_user, app_config.smtp_password)
        server.send_message(msg)
        server.quit()

        logger.info(
            "Daily summary email sent via SMTP — namespace: %s, to: %s",
            namespace,
            to_address,
        )
    except Exception as e:
        logger.error(
            "Failed to send daily summary email — namespace: %s, error: %s",
            namespace,
            e,
        )


# ============================================
# Core Job Logic
# ============================================

async def execute_daily_summary() -> dict:
    """Execute the daily summary job.

    This is the core business function that:
      1. Queries MongoDB for widget sessions from the last 24 hours
      2. Groups sessions by namespace
      3. Analyzes intent for each session via the LLM
      4. Generates a PDF report per namespace
      5. Emails the PDF reports to admin

    Can be called by:
      - APScheduler cron trigger (automatic)
      - Admin API endpoint (manual trigger)

    Returns:
        Summary dict with job execution results.
    """
    from providers.factory import create_llm_provider
    from services.email_agent import EmailAgent
    from services.mongo_store import session_store

    logger.info("=" * 60)
    logger.info("Starting Daily Summary Job")
    logger.info("=" * 60)

    # 1. Calculate time window
    since = datetime.utcnow() - timedelta(hours=24)
    report_date = datetime.utcnow().strftime("%B %d, %Y")

    # 2. Fetch widget sessions
    sessions = await session_store.get_widget_sessions_since(since)

    if not sessions:
        logger.info("No widget sessions found in the last 24 hours. Skipping report.")
        return {
            "status": "skipped",
            "reason": "No widget sessions in the last 24 hours",
            "report_date": report_date,
        }

    # 3. Group sessions by namespace
    namespace_groups: dict[str, list[Session]] = defaultdict(list)
    for session in sessions:
        ns = session.namespace or "unknown"
        namespace_groups[ns].append(session)

    logger.info(
        "Found %d widget sessions across %d namespace(s): %s",
        len(sessions),
        len(namespace_groups),
        list(namespace_groups.keys()),
    )

    # 4. Analyze intent and generate summary for all sessions
    llm_provider = create_llm_provider()
    email_agent = EmailAgent(llm_provider, admin_emails=app_config.admin_emails)

    intents: dict[str, str] = {}
    summaries: dict[str, str] = {}
    for session in sessions:
        try:
            is_high = await email_agent.analyze_intent(session)
            intents[session.session_id] = "HIGH" if is_high else "LOW"
            summary = await email_agent.summarize_conversation(session)
            summaries[session.session_id] = summary
        except Exception as e:
            logger.warning(
                "Intent or summary analysis failed for session %s: %s",
                session.session_id,
                e,
            )
            intents[session.session_id] = "N/A"
            summaries[session.session_id] = "Summary unavailable due to error."

    # 5. Generate PDF and send email for each namespace
    results: dict[str, dict] = {}
    for namespace, ns_sessions in namespace_groups.items():
        try:
            pdf_bytes = _build_namespace_pdf(
                namespace=namespace,
                sessions=ns_sessions,
                intents=intents,
                summaries=summaries,
                report_date=report_date,
            )

            _send_summary_email(
                namespace=namespace,
                pdf_bytes=pdf_bytes,
                report_date=report_date,
                session_count=len(ns_sessions),
            )

            results[namespace] = {
                "status": "sent",
                "session_count": len(ns_sessions),
                "pdf_size_bytes": len(pdf_bytes),
            }
            logger.info(
                "Daily summary for namespace '%s': %d sessions, PDF %d bytes",
                namespace,
                len(ns_sessions),
                len(pdf_bytes),
            )

        except Exception as e:
            logger.error(
                "Failed to generate/send daily summary for namespace '%s': %s",
                namespace,
                e,
                exc_info=True,
            )
            results[namespace] = {
                "status": "failed",
                "error": str(e),
            }

    logger.info("Daily Summary Job completed. Results: %s", results)
    return {
        "status": "completed",
        "report_date": report_date,
        "total_sessions": len(sessions),
        "namespaces": results,
    }
