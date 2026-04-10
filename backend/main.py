"""
FastAPI Application Entry Point

Initializes the FastAPI app with:
  - CORS middleware for frontend access
  - Lifespan handler for startup/shutdown events
  - Health check endpoint
  - LLM provider health verification on startup
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import app_config
from providers.factory import create_llm_provider

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if app_config.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# LLM provider — initialized at startup
llm_provider = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Startup:
      - Initialize the LLM provider
      - Run health check to verify connectivity
    Shutdown:
      - Cleanup resources
    """
    global llm_provider

    logger.info("=" * 60)
    logger.info("Starting AI Agentic Chatbot Backend")
    logger.info("=" * 60)

    # Initialize LLM provider
    llm_provider = create_llm_provider()
    app.state.llm_provider = llm_provider

    # Health check
    is_healthy = await llm_provider.health_check()
    if is_healthy:
        logger.info("✅ LLM provider is healthy and ready")
    else:
        logger.warning(
            "⚠️  LLM provider health check failed. "
            "The application will start but LLM calls may fail."
        )

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
