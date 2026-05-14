"""
FastAPI Application Entry Point

Initializes the FastAPI app with:
  - CORS middleware for frontend access
  - Lifespan handler for startup/shutdown events
  - Health check endpoint
  - LLM provider health verification on startup
  - Orchestrator initialization with all agents
  - Chat API routes
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import app_config, pinecone_config, scheduler_config
from providers.factory import create_llm_provider
from services.orchestrator import Orchestrator
from services.mongo_store import session_store
from services.knowledge_base_factory import KnowledgeBaseFactory
from services.scheduler import execute_daily_summary
from routes.chat import router as chat_router
from routes.admin import router as admin_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if app_config.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Startup:
      - Initialize the LLM provider
      - Run health check to verify connectivity
      - Initialize the KnowledgeBase Factory (Pinecone + namespace support)
      - Initialize the Orchestrator with all agents
    Shutdown:
      - Cleanup resources
    """
    logger.info("=" * 60)
    logger.info("Starting AI Agentic Chatbot Backend")
    logger.info("=" * 60)

    # Initialize LLM provider
    llm_provider = create_llm_provider()
    app.state.llm_provider = llm_provider

    # Health check
    is_healthy = await llm_provider.health_check()
    if is_healthy:
        logger.info("LLM provider is healthy and ready")
    else:
        logger.warning(
            "LLM provider health check failed. "
            "The application will start but LLM calls may fail."
        )

    # Initialize Knowledge Base Factory (namespace-scoped RAG pipeline)
    kb_factory = None
    if pinecone_config.api_key:
        try:
            kb_factory = KnowledgeBaseFactory()
            kb_ready = await kb_factory.initialize()
            if kb_ready:
                logger.info(
                    "Knowledge Base Factory is ready (default namespace: '%s')",
                    pinecone_config.namespace,
                )
                # Expose default KB for admin routes and health checks
                app.state.knowledge_base = kb_factory.default
            else:
                logger.warning(
                    "Knowledge Base initialization failed. "
                    "RAG features will be unavailable."
                )
                kb_factory = None
                app.state.knowledge_base = None
        except Exception as e:
            logger.warning("Knowledge Base setup error: %s. RAG features disabled.", e)
            kb_factory = None
            app.state.knowledge_base = None
    else:
        logger.info("No Pinecone API key configured — RAG features disabled")
        app.state.knowledge_base = None

    app.state.kb_factory = kb_factory

    # Initialize Orchestrator (coordinates all agents)
    orchestrator = Orchestrator(
        llm_provider=llm_provider,
        session_store=session_store,
        kb_factory=kb_factory,
    )
    app.state.orchestrator = orchestrator
    logger.info("Orchestrator initialized with all agents")

    # Initialize Daily Summary Scheduler
    scheduler = AsyncIOScheduler()
    try:
        cron_expr = scheduler_config.daily_summary_cron
        parts = cron_expr.strip().split()
        if len(parts) == 5:
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )
            scheduler.add_job(
                execute_daily_summary,
                trigger=trigger,
                id="daily_summary",
                name="Daily Conversation Summary Report",
                replace_existing=True,
            )
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info(
                "Daily summary scheduler started — cron: '%s'",
                cron_expr,
            )
        else:
            logger.warning(
                "Invalid DAILY_SUMMARY_CRON format: '%s'. Expected 5 fields. Scheduler disabled.",
                cron_expr,
            )
            app.state.scheduler = None
    except Exception as e:
        logger.warning("Failed to start daily summary scheduler: %s", e)
        app.state.scheduler = None

    logger.info("Backend is ready — listening on %s:%s", app_config.host, app_config.port)
    logger.info("=" * 60)

    yield  # App is running

    # Shutdown
    logger.info("Shutting down AI Agentic Chatbot Backend")
    if getattr(app.state, 'scheduler', None):
        app.state.scheduler.shutdown(wait=False)
        logger.info("Daily summary scheduler stopped")


# Create FastAPI app
app = FastAPI(
    title="AI Agentic Chatbot API",
    description="LLM-powered multi-agent chatbot for lead qualification",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat_router)
app.include_router(admin_router)


# --- Health & Status Endpoints ---


@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Application health check endpoint."""
    llm_healthy = False
    if app.state.llm_provider:
        llm_healthy = await app.state.llm_provider.health_check()

    # Also check MongoDB health
    db_healthy = False
    try:
        from services.mongo_store import session_store
        # Simple ping to check connection
        await session_store._db.command("ping")
        db_healthy = True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")

    return {
        "status": "healthy" if llm_healthy and db_healthy else "degraded",
        "llm_provider": {
            "healthy": llm_healthy,
            "provider": app.state.llm_provider.__class__.__name__
            if app.state.llm_provider
            else None,
        },
        "database": {
            "healthy": db_healthy,
            "provider": "MongoDB"
        }
    }


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "AI Agentic Chatbot API",
        "version": "0.1.0",
        "docs": "/docs",
    }


# --- Main ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
    )
