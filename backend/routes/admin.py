"""
Knowledge Base Admin Routes — 📚 Document Management API

Provides REST API endpoints for managing the RAG knowledge base:
  - POST /api/admin/ingest — Ingest PDF documents into Pinecone
  - GET /api/admin/kb-stats — Get knowledge base statistics
  - POST /api/admin/kb-clear — Clear all vectors from the index

Design:
  - Admin routes — intended for internal/developer use
  - No business logic — delegates to KnowledgeBase service
  - Proper error handling with meaningful status codes
"""

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router with /api/admin prefix
router = APIRouter(prefix="/api/admin", tags=["Knowledge Base Admin"])


# ============================================
# Response Models
# ============================================

class IngestResponse(BaseModel):
    """Response body for POST /api/admin/ingest."""

    message: str
    stats: dict


class KBStatsResponse(BaseModel):
    """Response body for GET /api/admin/kb-stats."""

    status: str
    stats: dict


class KBClearResponse(BaseModel):
    """Response body for POST /api/admin/kb-clear."""

    message: str
    success: bool


# ============================================
# Endpoints
# ============================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: Request) -> IngestResponse:
    """Ingest all PDF documents from the documents directory into Pinecone.

    Processes each PDF: extracts text → chunks → embeds → upserts to Pinecone.
    This is an idempotent operation — re-running will update existing vectors.

    Returns:
        IngestResponse with processing statistics.

    Raises:
        HTTPException 503: If knowledge base is not initialized.
        HTTPException 500: If ingestion fails.
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized. Check Pinecone configuration.",
        )

    try:
        # Determine documents directory path
        documents_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "documents",
        )

        if not os.path.exists(documents_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Documents directory not found: {documents_dir}",
            )

        logger.info("Starting document ingestion from: %s", documents_dir)
        stats = await knowledge_base.ingest_documents(documents_dir)

        return IngestResponse(
            message=(
                f"Ingestion complete. Processed {stats['files_processed']} files, "
                f"created {stats['total_chunks']} chunks, "
                f"upserted {stats['total_vectors_upserted']} vectors."
            ),
            stats=stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document ingestion failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get("/kb-stats", response_model=KBStatsResponse)
async def get_kb_stats(request: Request) -> KBStatsResponse:
    """Get statistics about the Pinecone knowledge base index.

    Returns vector count, dimension, and namespace information.

    Returns:
        KBStatsResponse with index statistics.

    Raises:
        HTTPException 503: If knowledge base is not initialized.
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized.",
        )

    try:
        stats = await knowledge_base.get_index_stats()
        return KBStatsResponse(status="connected", stats=stats)

    except Exception as e:
        logger.error("Failed to get KB stats: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )


@router.post("/kb-clear", response_model=KBClearResponse)
async def clear_kb(request: Request) -> KBClearResponse:
    """Clear all vectors from the Pinecone knowledge base.

    WARNING: This deletes all embedded documents. You'll need to
    re-run ingestion after clearing.

    Returns:
        KBClearResponse confirming the operation.

    Raises:
        HTTPException 503: If knowledge base is not initialized.
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized.",
        )

    try:
        success = await knowledge_base.clear_index()

        if success:
            return KBClearResponse(
                message="All vectors cleared from the knowledge base.",
                success=True,
            )
        else:
            return KBClearResponse(
                message="Failed to clear knowledge base.",
                success=False,
            )

    except Exception as e:
        logger.error("Failed to clear KB: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear knowledge base: {str(e)}",
        )
