"""
Chat API Routes — Thin HTTP Handlers

Provides REST API endpoints for the chatbot:
  - POST /api/chat — Process a user message
  - GET /api/session/{session_id} — Get session state
  - POST /api/reset — Reset a session

Design:
  - Routes are thin — NO business logic here
  - All processing is delegated to the Orchestrator
  - Uses Pydantic models for request/response validation
  - Proper error handling with meaningful HTTP status codes
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from models.schemas import (
    ChatRequest,
    ChatResponse,
    ResetRequest,
    ResetResponse,
    SessionResponse,
)

logger = logging.getLogger(__name__)

# Create router with /api prefix
router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Process a user message and return the AI response.

    Routes the message to the Orchestrator, which determines
    the appropriate agent and handles all business logic.

    Args:
        request: The FastAPI request (for accessing app state).
        body: Validated ChatRequest with session_id and message.

    Returns:
        ChatResponse with reply, current stage, and collected data.

    Raises:
        HTTPException 500: If message processing fails unexpectedly.
    """
    orchestrator = request.app.state.orchestrator

    try:
        logger.info(
            "POST /api/chat — session: %s, message: %s",
            body.session_id,
            body.message[:50] + "..." if len(body.message) > 50 else body.message,
        )

        response = await orchestrator.process_message(
            session_id=body.session_id,
            message=body.message,
        )

        logger.info(
            "Response — session: %s, stage: %s",
            body.session_id,
            response.stage.value,
        )

        return response

    except Exception as e:
        logger.error(
            "Error processing chat message (session: %s): %s",
            body.session_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message. Please try again.",
        )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(request: Request, session_id: str) -> SessionResponse:
    """Get the full state of a session.

    Returns conversation history, collected data, current stage,
    and summary if available. Useful for debugging and session
    inspection.

    Args:
        request: The FastAPI request.
        session_id: The session to retrieve.

    Returns:
        SessionResponse with complete session state.

    Raises:
        HTTPException 404: If no session exists with this ID.
    """
    orchestrator = request.app.state.orchestrator

    try:
        session = await orchestrator.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        return SessionResponse(
            session_id=session.session_id,
            stage=session.stage,
            collected_data=session.collected_data,
            conversation_history=session.conversation_history,
            summary=session.summary,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving session %s: %s", session_id, e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving the session.",
        )


@router.post("/reset", response_model=ResetResponse)
async def reset_session(
    request: Request, body: ResetRequest
) -> ResetResponse:
    """Reset a session, clearing all conversation state.

    Deletes the session from storage. The next message from this
    session ID will start a fresh conversation.

    Args:
        request: The FastAPI request.
        body: Validated ResetRequest with session_id.

    Returns:
        ResetResponse confirming the reset.
    """
    orchestrator = request.app.state.orchestrator

    try:
        was_deleted = await orchestrator.reset_session(body.session_id)

        if was_deleted:
            message = f"Session '{body.session_id}' has been reset successfully."
        else:
            message = f"No active session found for '{body.session_id}'. A new session will be created on next message."

        logger.info("POST /api/reset — session: %s, deleted: %s", body.session_id, was_deleted)

        return ResetResponse(
            message=message,
            session_id=body.session_id,
        )

    except Exception as e:
        logger.error("Error resetting session %s: %s", body.session_id, e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while resetting the session.",
        )
