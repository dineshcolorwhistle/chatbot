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
from pydantic import BaseModel, Field

from config import pinecone_config

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


class IngestRequest(BaseModel):
    """Request body for POST /api/admin/ingest."""

    namespace: str | None = Field(
        None,
        description="Pinecone namespace to ingest into. Defaults to config value.",
        examples=["colorwhistle", "client-abc"],
    )
    documents_dir: str | None = Field(
        None,
        description=(
            "Custom path to the documents directory. "
            "If not provided, looks for 'documents/{namespace}/' first, "
            "then falls back to 'documents/'."
        ),
    )


class KBClearRequest(BaseModel):
    """Request body for POST /api/admin/kb-clear."""

    namespace: str | None = Field(
        None,
        description="Pinecone namespace to clear. Defaults to config value.",
        examples=["colorwhistle", "client-abc"],
    )


# ============================================
# Endpoints
# ============================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: Request, body: IngestRequest | None = None) -> IngestResponse:
    """Ingest all PDF documents from a directory into a Pinecone namespace.

    Processes each PDF: extracts text → chunks → embeds → upserts to Pinecone.
    This is an idempotent operation — re-running will update existing vectors.

    Directory resolution order:
      1. Custom path from request body (documents_dir)
      2. Per-namespace directory: documents/{namespace}/
      3. Default directory: documents/

    Args (request body):
        namespace: Target Pinecone namespace (default: config value).
        documents_dir: Custom documents directory path (optional).

    Returns:
        IngestResponse with processing statistics.

    Raises:
        HTTPException 503: If knowledge base is not initialized.
        HTTPException 404: If no documents directory is found.
        HTTPException 500: If ingestion fails.
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized. Check Pinecone configuration.",
        )

    try:
        # Determine target namespace
        target_namespace = body.namespace if body and body.namespace else pinecone_config.namespace

        # Create a scoped KB instance for the target namespace
        from services.knowledge_base import KnowledgeBase
        scoped_kb = KnowledgeBase(namespace=target_namespace)
        scoped_kb._pc = knowledge_base._pc
        scoped_kb._index = knowledge_base._index

        # Resolve documents directory (priority: custom → per-namespace → default)
        backend_root = os.path.dirname(os.path.dirname(__file__))

        if body and body.documents_dir:
            # 1. Custom path from request
            documents_dir = body.documents_dir
        else:
            # 2. Per-namespace directory: documents/{namespace}/
            namespace_dir = os.path.join(backend_root, "documents", target_namespace)
            default_dir = os.path.join(backend_root, "documents")

            if os.path.exists(namespace_dir) and os.listdir(namespace_dir):
                documents_dir = namespace_dir
            else:
                # 3. Default directory: documents/
                documents_dir = default_dir

        if not os.path.exists(documents_dir):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Documents directory not found. Tried:\n"
                    f"  - Per-namespace: documents/{target_namespace}/\n"
                    f"  - Default: documents/\n"
                    f"Create one of these directories and add PDF files."
                ),
            )

        logger.info(
            "Starting document ingestion from: %s into namespace: '%s'",
            documents_dir,
            target_namespace,
        )
        stats = await scoped_kb.ingest_documents(documents_dir)

        return IngestResponse(
            message=(
                f"Ingestion complete into namespace '{target_namespace}'. "
                f"Source: {documents_dir}. "
                f"Processed {stats['files_processed']} files, "
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
async def clear_kb(request: Request, body: KBClearRequest | None = None) -> KBClearResponse:
    """Clear all vectors from a specific namespace in the Pinecone knowledge base.

    WARNING: This deletes all embedded documents in the target namespace.
    You'll need to re-run ingestion after clearing.

    Accepts an optional namespace in the request body. Defaults to the
    config namespace if not specified.

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
        target_namespace = body.namespace if body and body.namespace else None
        success = await knowledge_base.clear_namespace(target_namespace)

        ns_label = target_namespace or pinecone_config.namespace

        if success:
            return KBClearResponse(
                message=f"All vectors cleared from namespace '{ns_label}'.",
                success=True,
            )
        else:
            return KBClearResponse(
                message=f"Failed to clear namespace '{ns_label}'.",
                success=False,
            )

    except Exception as e:
        logger.error("Failed to clear KB: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear knowledge base: {str(e)}",
        )
