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

from config import app_config, pinecone_config
from providers.factory import create_llm_provider
from services.orchestrator import Orchestrator
from services.memory_store import session_store
from services.knowledge_base import KnowledgeBase
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
      - Initialize the Knowledge Base (Pinecone + embeddings)
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

    # Initialize Knowledge Base (RAG pipeline)
    knowledge_base = None
    if pinecone_config.api_key:
        try:
            knowledge_base = KnowledgeBase()
            kb_ready = await knowledge_base.initialize()
            if kb_ready:
                logger.info("Knowledge Base (Pinecone) is ready")
                app.state.knowledge_base = knowledge_base
            else:
                logger.warning(
                    "Knowledge Base initialization failed. "
                    "RAG features will be unavailable."
                )
                knowledge_base = None
        except Exception as e:
            logger.warning("Knowledge Base setup error: %s. RAG features disabled.", e)
            knowledge_base = None
    else:
        logger.info("No Pinecone API key configured — RAG features disabled")

    app.state.knowledge_base = knowledge_base

    # Initialize Orchestrator (coordinates all agents)
    orchestrator = Orchestrator(
        llm_provider=llm_provider,
        session_store=session_store,
        knowledge_base=knowledge_base,
    )
    app.state.orchestrator = orchestrator
    logger.info("Orchestrator initialized with all agents")

    logger.info("Backend is ready — listening on %s:%s", app_config.host, app_config.port)
    logger.info("=" * 60)

    yield  # App is running

    # Shutdown
    logger.info("Shutting down AI Agentic Chatbot Backend")


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


@app.get("/health")
async def health_check():
    """Application health check endpoint."""
    llm_healthy = False
    if app.state.llm_provider:
        llm_healthy = await app.state.llm_provider.health_check()

    return {
        "status": "healthy",
        "llm_provider": {
            "healthy": llm_healthy,
            "provider": app.state.llm_provider.__class__.__name__
            if app.state.llm_provider
            else None,
        },
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
