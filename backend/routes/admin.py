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
import shutil

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import pinecone_config
from routes.auth import get_current_admin

logger = logging.getLogger(__name__)

# Create router with /api/admin prefix
router = APIRouter(prefix="/api/admin", tags=["Knowledge Base Admin"], dependencies=[Depends(get_current_admin)])


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

class ExtractYoutubeRequest(BaseModel):
    """Request body for POST /api/admin/extract-youtube."""
    
    url: str = Field(
        ...,
        description="YouTube URL to extract transcripts from."
    )


class NamespaceDeleteRequest(BaseModel):
    """Request body for DELETE /api/admin/namespace/{namespace}."""

    delete_local_files: bool = Field(
        True,
        description="Whether to also delete local documents/{namespace}/ folder.",
    )


class CronStatus(BaseModel):
    """Request/Response body for cron status."""

    enabled: bool = Field(
        ...,
        description="Whether the daily summary cron job is enabled."
    )

# ============================================
# Endpoints
# ============================================

@router.post("/extract-youtube")
async def extract_youtube_transcripts(request: ExtractYoutubeRequest, background_tasks: BackgroundTasks):
    """Extract YouTube transcripts and return as a PDF download.
    
    This endpoint does NOT save the PDF to the backend. It generates the PDF
    and returns it as a direct download, then deletes the temporary file.
    """
    from services.youtube_service import YouTubeExtractorService
    
    url = request.url
    
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided.")
        
    video_id = YouTubeExtractorService.extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {url}")
        
    transcript = YouTubeExtractorService.get_transcript(url)
    if not transcript:
        raise HTTPException(status_code=400, detail=f"Failed to get transcript for video ID {video_id} (URL: {url}). Make sure subtitles are enabled.")
        
    try:
        file_path = YouTubeExtractorService.save_as_pdf(transcript, video_id)
        
        # Add background task to delete the temporary file after sending it
        background_tasks.add_task(os.remove, file_path)
        
        return FileResponse(
            path=file_path,
            filename=f"youtube_{video_id}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.error("Failed to save PDF for %s: %s", video_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to save PDF for video {video_id}: {str(e)}")

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


@router.post("/upload")
async def upload_and_ingest(
    request: Request,
    namespace: str = Form(..., description="Pinecone namespace to ingest into."),
    files: list[UploadFile] = File(..., description="PDF files to upload and ingest.")
):
    """Upload PDF files to a namespace and automatically run ingestion."""
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized. Check Pinecone configuration.",
        )

    # Validate files
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for '{file.filename}'. Only PDFs are allowed."
            )

    try:
        # 1. Create the directory for the namespace
        backend_root = os.path.dirname(os.path.dirname(__file__))
        namespace_dir = os.path.join(backend_root, "documents", namespace)
        os.makedirs(namespace_dir, exist_ok=True)

        # 2. Save the uploaded files
        saved_files = []
        for file in files:
            file_path = os.path.join(namespace_dir, file.filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(file.filename)

        logger.info(f"Saved {len(saved_files)} files to {namespace_dir}")

        # 3. Trigger Ingestion
        from services.knowledge_base import KnowledgeBase
        scoped_kb = KnowledgeBase(namespace=namespace)
        scoped_kb._pc = knowledge_base._pc
        scoped_kb._index = knowledge_base._index

        stats = await scoped_kb.ingest_documents(namespace_dir)

        return {
            "message": f"Successfully uploaded {len(saved_files)} files and completed ingestion into namespace '{namespace}'.",
            "saved_files": saved_files,
            "stats": stats
        }

    except Exception as e:
        logger.error("Upload and ingest failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Upload and ingest failed: {str(e)}",
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


# ============================================
# Namespace Management
# ============================================

@router.get("/namespaces")
async def list_namespaces(request: Request):
    """List all Pinecone namespaces with their vector counts.

    Also includes whether a local documents directory exists for each namespace.

    Returns:
        List of namespace objects with name, vector_count, and has_local_docs.
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized.",
        )

    try:
        stats = await knowledge_base.get_index_stats()
        namespaces_raw = stats.get("namespaces", {})

        backend_root = os.path.dirname(os.path.dirname(__file__))
        documents_root = os.path.join(backend_root, "documents")

        namespaces = []
        for ns_name, ns_info in namespaces_raw.items():
            # ns_info is a NamespaceSummary object — extract vector_count
            vector_count = getattr(ns_info, "vector_count", 0)
            local_dir = os.path.join(documents_root, ns_name)
            has_local_docs = os.path.isdir(local_dir)

            namespaces.append({
                "name": ns_name,
                "vector_count": vector_count,
                "has_local_docs": has_local_docs,
            })

        # Sort alphabetically
        namespaces.sort(key=lambda x: x["name"])

        return {
            "namespaces": namespaces,
            "total_vectors": stats.get("total_vectors", 0),
            "index_name": stats.get("index_name", "unknown"),
        }

    except Exception as e:
        logger.error("Failed to list namespaces: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list namespaces: {str(e)}",
        )


@router.delete("/namespace/{namespace}")
async def delete_namespace(
    namespace: str,
    request: Request,
    body: NamespaceDeleteRequest | None = None,
):
    """Delete an entire namespace from Pinecone.

    This removes ALL vectors within the specified namespace.
    Pinecone automatically removes empty namespaces from the index stats.

    Optionally also deletes the local documents/{namespace}/ directory.

    Args:
        namespace: The namespace to delete (path parameter).
        body: Optional request body with delete_local_files flag.

    Returns:
        Confirmation of the deletion.

    Raises:
        HTTPException 503: If knowledge base is not initialized.
        HTTPException 400: If namespace is empty.
        HTTPException 500: If deletion fails.
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)

    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not initialized.",
        )

    namespace = namespace.strip()
    if not namespace:
        raise HTTPException(
            status_code=400,
            detail="Namespace cannot be empty.",
        )

    delete_local = body.delete_local_files if body else True

    try:
        # 1. Clear all vectors from the namespace in Pinecone
        success = await knowledge_base.clear_namespace(namespace)

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete vectors from namespace '{namespace}'.",
            )

        # 2. Optionally delete local documents folder
        local_deleted = False
        backend_root = os.path.dirname(os.path.dirname(__file__))
        namespace_dir = os.path.join(backend_root, "documents", namespace)

        if delete_local and os.path.isdir(namespace_dir):
            shutil.rmtree(namespace_dir)
            local_deleted = True
            logger.info(
                "Deleted local documents directory: %s", namespace_dir
            )

        logger.info(
            "Namespace '%s' fully deleted (vectors: cleared, local_docs: %s)",
            namespace,
            "deleted" if local_deleted else "skipped/not_found",
        )

        return {
            "success": True,
            "namespace": namespace,
            "vectors_cleared": True,
            "local_files_deleted": local_deleted,
            "message": (
                f"Namespace '{namespace}' has been deleted. "
                f"All vectors removed from Pinecone."
                + (f" Local documents folder also removed." if local_deleted else "")
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete namespace '%s': %s", namespace, e, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete namespace '{namespace}': {str(e)}",
        )


# ============================================
# Daily Summary — Manual Trigger
# ============================================

@router.post("/trigger-daily-summary")
async def trigger_daily_summary(background_tasks: BackgroundTasks):
    """Manually trigger the daily conversation summary job.

    Runs the daily summary in a background task so the API
    responds immediately. Generates per-namespace PDF reports
    of widget conversations from the last 24 hours and emails
    them to the configured admin addresses.

    Returns:
        Status message confirming the job was triggered.
    """
    from services.scheduler import execute_daily_summary

    async def _run_and_log():
        try:
            result = await execute_daily_summary()
            logger.info("Manual daily summary completed: %s", result)
        except Exception as e:
            logger.error("Manual daily summary failed: %s", e, exc_info=True)

    background_tasks.add_task(_run_and_log)

    logger.info("POST /api/admin/trigger-daily-summary — Job triggered manually")
    return {
        "status": "triggered",
        "message": "Daily summary job is running in the background. Check logs for results.",
    }


@router.get("/cron-status", response_model=CronStatus)
async def get_cron_status(request: Request) -> CronStatus:
    """Get the current enable/disable status of the daily summary cron job."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if not scheduler:
        return CronStatus(enabled=False)
    
    job = scheduler.get_job("daily_summary")
    if not job:
        return CronStatus(enabled=False)
        
    return CronStatus(enabled=job.next_run_time is not None)


@router.post("/cron-status", response_model=CronStatus)
async def set_cron_status(request: Request, body: CronStatus) -> CronStatus:
    """Enable or disable the daily summary cron job."""
    from dotenv import set_key
    import os

    scheduler = getattr(request.app.state, "scheduler", None)
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized.")
        
    job = scheduler.get_job("daily_summary")
    if not job:
        raise HTTPException(status_code=503, detail="Daily summary job not found.")

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    
    if body.enabled:
        scheduler.resume_job("daily_summary")
        set_key(env_path, "DAILY_SUMMARY_ENABLED", "true")
        logger.info("Daily summary cron job enabled.")
    else:
        scheduler.pause_job("daily_summary")
        set_key(env_path, "DAILY_SUMMARY_ENABLED", "false")
        logger.info("Daily summary cron job disabled.")
        
    return CronStatus(enabled=body.enabled)
