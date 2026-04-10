# AI Agentic Conversational Chatbot

LLM-powered multi-agent chatbot for business lead qualification and requirement gathering.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript (Vite) |
| Backend | FastAPI (Python) |
| LLM | Ollama (local) / Cloud (OpenAI-compatible) |
| Sessions | In-memory |

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

### Configuration

Copy `.env.example` to `.env` and configure:

```env
LLM_PROVIDER=ollama              # "ollama" or "cloud"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Architecture

Multi-agent system with 4 specialized agents:

- **🎯 Orchestrator** — Coordinates agents, manages state (code-driven)
- **🗣️ Conversation Agent** — Natural dialogue with users (LLM-powered)
- **📋 Summarization Agent** — Generates structured summaries (LLM-powered)
- **📧 Email Agent** — Composes notification emails (LLM-powered)

## Project Documentation

- [PROJECT_SCOPE.md](./PROJECT_SCOPE.md) — Full project specification
- [PROJECT_WALKTHROUGH.md](./PROJECT_WALKTHROUGH.md) — Architecture, phases, progress

## Phase Progress

| Phase | Module | Status |
|-------|--------|--------|
| Phase 1 | LLM Provider Layer | ✅ Complete |
| Phase 2 | Models & Session Store | ✅ Complete |
| Phase 3 | Conversation Agent | ⬜ Not Started |
| Phase 4 | Summarization Agent | ⬜ Not Started |
| Phase 5 | Email Agent | ⬜ Not Started |
| Phase 6 | Orchestrator | ⬜ Not Started |
| Phase 7 | API Routes | ⬜ Not Started |
| Phase 8 | Frontend | ⬜ Not Started |
| Phase 9 | Polish & Docs | ⬜ Not Started |
