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

### Prerequisites

- **Python 3.11+** (backend)
- **Node.js 18+** (frontend)
- **Ollama** running locally with a model pulled (`ollama pull llama3.2`)

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

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

### Configuration

Copy `.env.example` to `.env` and configure:

```env
LLM_PROVIDER=ollama              # "ollama" or "cloud"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

CLOUD_API_KEY=your-key           # Only needed for cloud provider
CLOUD_BASE_URL=https://api.openai.com/v1
CLOUD_MODEL=gpt-4o-mini
```

## Architecture

Multi-agent system with 4 specialized agents:

- **🎯 Orchestrator** — Coordinates agents, manages state & stage transitions (code-driven)
- **🗣️ Conversation Agent** — Natural dialogue with users across stages 1-4 (LLM-powered)
- **📋 Summarization Agent** — Generates structured lead summaries at stage 5 (LLM-powered)
- **📧 Email Agent** — Composes notification emails at stage 6 (LLM-powered)

### Conversation Flow

```
Welcome → Personal Info → Tech Discovery → Scope & Pricing → Summary → Email → Complete
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a message and get AI response |
| GET | `/api/session/{id}` | Get full session state |
| POST | `/api/reset` | Reset a session |
| GET | `/health` | Health check with LLM provider status |

## Folder Structure

```
chatbot/
├── backend/
│   ├── main.py                        # FastAPI entry point, CORS, lifespan
│   ├── config.py                      # Settings from .env
│   ├── routes/
│   │   └── chat.py                    # Thin API handlers → orchestrator
│   ├── services/
│   │   ├── orchestrator.py            # 🎯 Agent coordinator & state manager
│   │   ├── conversation_agent.py      # 🗣️ LLM-powered dialogue agent
│   │   ├── summarization_agent.py     # 📋 LLM-powered summary generator
│   │   ├── email_agent.py            # 📧 LLM-powered email composer
│   │   ├── session_store.py          # Abstract session interface
│   │   └── memory_store.py           # In-memory session implementation
│   ├── models/
│   │   └── schemas.py                # Pydantic request/response models
│   └── providers/
│       ├── base.py                   # Abstract LLM provider interface
│       ├── ollama_provider.py        # Ollama local implementation
│       ├── cloud_provider.py         # Cloud (OpenAI-compatible) provider
│       └── factory.py               # Provider factory
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx        # Chat UI component
│   │   │   └── ChatWindow.css        # Chat styling
│   │   ├── pages/
│   │   │   └── ChatPage.tsx          # Main chat page
│   │   ├── api.ts                    # API client
│   │   ├── App.tsx                   # Root component
│   │   ├── main.tsx                  # Entry point
│   │   └── index.css                 # Global design system
│   ├── index.html
│   └── package.json
│
├── PROJECT_SCOPE.md
├── PROJECT_WALKTHROUGH.md
└── README.md
```

## Project Documentation

- [PROJECT_SCOPE.md](./PROJECT_SCOPE.md) — Full project specification
- [PROJECT_WALKTHROUGH.md](./PROJECT_WALKTHROUGH.md) — Architecture, phases, progress

## Phase Progress

| Phase | Module | Status |
|-------|--------|--------|
| Phase 1 | LLM Provider Layer | ✅ Complete |
| Phase 2 | Models & Session Store | ✅ Complete |
| Phase 3 | Conversation Agent | ✅ Complete |
| Phase 4 | Summarization Agent | ✅ Complete |
| Phase 5 | Email Agent | ✅ Complete |
| Phase 6 | Orchestrator | ✅ Complete |
| Phase 7 | API Routes | ✅ Complete |
| Phase 8 | Frontend | ✅ Complete |
| Phase 9 | Polish & Docs | ✅ Complete |

## Running Tests

```bash
cd backend
.\venv\Scripts\activate
python test_agents.py     # Agent tests (Phases 3-5)
python test_phase2.py     # Schema & session tests (Phase 2)
```

## License

Private — ColorWhistle
