"""
Pydantic Models & Schemas

Defines all data structures used across the application:
  - API request/response models
  - Internal session and conversation models
  - Conversation stage enum
  - Collected data structure

These models serve as the single source of truth for data shapes
throughout the backend. Agents, routes, and services all reference
these models — no ad-hoc dictionaries.
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# ============================================
# Conversation Stages
# ============================================

class ConversationStage(str, Enum):
    """Defines the sequential stages of the lead qualification flow.

    The orchestrator uses this to determine which agent to invoke
    and when to transition between stages.
    """

    WELCOME = "welcome"
    PERSONAL_INFO = "personal_info"
    TECH_DISCOVERY = "tech_discovery"
    SCOPE_PRICING = "scope_pricing"
    SUMMARY = "summary"
    EMAIL = "email"
    COMPLETED = "completed"


# Stage ordering for sequential transitions
STAGE_ORDER: list[ConversationStage] = [
    ConversationStage.WELCOME,
    ConversationStage.PERSONAL_INFO,
    ConversationStage.TECH_DISCOVERY,
    ConversationStage.SCOPE_PRICING,
    ConversationStage.SUMMARY,
    ConversationStage.EMAIL,
    ConversationStage.COMPLETED,
]


def get_next_stage(current: ConversationStage) -> ConversationStage | None:
    """Get the next stage in the conversation flow.

    Args:
        current: The current conversation stage.

    Returns:
        The next ConversationStage, or None if already at the final stage.
    """
    try:
        current_index = STAGE_ORDER.index(current)
        if current_index < len(STAGE_ORDER) - 1:
            return STAGE_ORDER[current_index + 1]
        return None
    except ValueError:
        return None


# ============================================
# Conversation Message
# ============================================

class ConversationMessage(BaseModel):
    """A single message in the conversation history.

    Attributes:
        role: Who sent the message — "user" or "assistant".
        content: The text content of the message.
        timestamp: When the message was created.
    """

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# Collected Data (Lead Information)
# ============================================

class PersonalInfo(BaseModel):
    """Personal details collected during Stage 2.

    All fields are optional — they get populated one at a time
    as the conversation progresses.
    """

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None

    def is_complete(self) -> bool:
        """Check if all required personal fields are collected.

        Company is optional, so only name, email, and phone are required.
        """
        return all([self.name, self.email, self.phone])

    def get_missing_fields(self) -> list[str]:
        """Return list of required fields not yet collected."""
        missing = []
        if not self.name:
            missing.append("name")
        if not self.email:
            missing.append("email")
        if not self.phone:
            missing.append("phone")
        return missing


class TechDiscovery(BaseModel):
    """Technical requirements collected during Stage 3.

    All fields are optional — populated through conversation.
    """

    project_type: str | None = None
    tech_stack: str | None = None
    features: str | None = None
    integrations: str | None = None

    def is_complete(self) -> bool:
        """Check if all required tech fields are collected.

        At minimum, project_type and features are required.
        """
        return all([self.project_type, self.features])

    def get_missing_fields(self) -> list[str]:
        """Return list of required fields not yet collected."""
        missing = []
        if not self.project_type:
            missing.append("project_type")
        if not self.features:
            missing.append("features")
        return missing


class ScopePricing(BaseModel):
    """Scope and pricing details collected during Stage 4.

    All fields are optional — populated through conversation.
    """

    budget: str | None = None
    timeline: str | None = None
    mvp_or_production: str | None = None
    priority_features: str | None = None

    def is_complete(self) -> bool:
        """Check if all required scope fields are collected.

        At minimum, budget and timeline are required.
        """
        return all([self.budget, self.timeline])

    def get_missing_fields(self) -> list[str]:
        """Return list of required fields not yet collected."""
        missing = []
        if not self.budget:
            missing.append("budget")
        if not self.timeline:
            missing.append("timeline")
        return missing


class CollectedData(BaseModel):
    """All data collected across the entire conversation.

    Aggregates personal info, tech discovery, and scope/pricing
    into a single structure that agents can reference.
    """

    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    tech_discovery: TechDiscovery = Field(default_factory=TechDiscovery)
    scope_pricing: ScopePricing = Field(default_factory=ScopePricing)

    def to_summary_dict(self) -> dict:
        """Convert collected data to a flat dictionary for summary generation.

        Returns:
            A dictionary with all non-None fields for easy rendering.
        """
        result = {}

        # Personal info
        if self.personal_info.name:
            result["Client Name"] = self.personal_info.name
        if self.personal_info.email:
            result["Email"] = self.personal_info.email
        if self.personal_info.phone:
            result["Phone"] = self.personal_info.phone
        if self.personal_info.company:
            result["Company"] = self.personal_info.company

        # Tech discovery
        if self.tech_discovery.project_type:
            result["Project Type"] = self.tech_discovery.project_type
        if self.tech_discovery.tech_stack:
            result["Technology"] = self.tech_discovery.tech_stack
        if self.tech_discovery.features:
            result["Features"] = self.tech_discovery.features
        if self.tech_discovery.integrations:
            result["Integrations"] = self.tech_discovery.integrations

        # Scope & pricing
        if self.scope_pricing.budget:
            result["Budget"] = self.scope_pricing.budget
        if self.scope_pricing.timeline:
            result["Timeline"] = self.scope_pricing.timeline
        if self.scope_pricing.mvp_or_production:
            result["Scope"] = self.scope_pricing.mvp_or_production
        if self.scope_pricing.priority_features:
            result["Priority Features"] = self.scope_pricing.priority_features

        return result


# ============================================
# Session
# ============================================

class Session(BaseModel):
    """Complete session state for a single user conversation.

    This is the core data structure managed by the orchestrator.
    It tracks everything about an ongoing conversation.

    Attributes:
        session_id: Unique identifier for this session.
        stage: Current conversation stage.
        collected_data: All data gathered from the user so far.
        conversation_history: Full message history (user + assistant).
        summary: Generated summary text (populated at Stage 5).
        created_at: When the session was created.
        updated_at: When the session was last updated.
    """

    session_id: str
    stage: ConversationStage = ConversationStage.WELCOME
    collected_data: CollectedData = Field(default_factory=CollectedData)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    summary: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: "user" or "assistant".
            content: The message text.
        """
        self.conversation_history.append(
            ConversationMessage(role=role, content=content)
        )
        self.updated_at = datetime.utcnow()

    def get_history_for_llm(self) -> list[dict[str, str]]:
        """Get conversation history formatted for LLM consumption.

        Returns:
            List of {"role": ..., "content": ...} dicts,
            without timestamps (LLMs don't need them).
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]


# ============================================
# API Request Models
# ============================================

class ChatRequest(BaseModel):
    """Request body for POST /api/chat.

    Attributes:
        session_id: Client-generated session identifier.
        message: The user's message text.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique session identifier",
        examples=["user-001"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message text",
        examples=["Hello, I need help with a project"],
    )


class ResetRequest(BaseModel):
    """Request body for POST /api/reset.

    Attributes:
        session_id: Session to reset.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Session ID to reset",
        examples=["user-001"],
    )


# ============================================
# API Response Models
# ============================================

class ChatResponse(BaseModel):
    """Response body for POST /api/chat.

    Attributes:
        reply: The assistant's response message.
        stage: Current conversation stage after processing.
        data_collected: Snapshot of all collected data so far.
    """

    reply: str
    stage: ConversationStage
    data_collected: dict = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Response body for GET /api/session/{session_id}.

    Returns the complete session state for debugging and inspection.
    """

    session_id: str
    stage: ConversationStage
    collected_data: CollectedData
    conversation_history: list[ConversationMessage]
    summary: str | None
    created_at: datetime
    updated_at: datetime


class ResetResponse(BaseModel):
    """Response body for POST /api/reset."""

    message: str
    session_id: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    llm_provider: dict
