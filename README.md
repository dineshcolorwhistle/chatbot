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

### Conversation Stages

```
Welcome → Personal Info → Tech Discovery → Scope & Pricing → Summary → Email → Complete
```

### Application Process Flow

The complete request lifecycle when a **user sends a message**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER SENDS A MESSAGE                               │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. FRONTEND (React + TypeScript)                                           │
│     ChatWindow.tsx → api.ts → POST /api/chat                               │
│     • User types message in chat UI                                         │
│     • api.ts sends { session_id, message } to backend                       │
│     • Shows loading state while waiting for response                        │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  2. API ROUTE (routes/chat.py)                                              │
│     POST /api/chat — Thin HTTP handler                                      │
│     • Validates request via Pydantic (ChatRequest)                          │
│     • Delegates to orchestrator.process_message(session_id, message)        │
│     • Returns ChatResponse { reply, stage, data_collected }                 │
│     ⚠️  NO business logic — routes are thin wrappers                       │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  3. ORCHESTRATOR (services/orchestrator.py) — 🎯 Central Hub               │
│     Code-driven, deterministic logic (NOT LLM-powered)                      │
│                                                                              │
│     Step 3a: Load or create session from MemoryStore                        │
│              └── If new session → send welcome message                      │
│                                                                              │
│     Step 3b: Route based on current stage:                                  │
│              ┌─────────────────────────────────────────────────────┐        │
│              │ WELCOME          → _handle_welcome()               │        │
│              │ PERSONAL_INFO    → _handle_conversation() ──┐      │        │
│              │ TECH_DISCOVERY   → _handle_conversation() ──┤      │        │
│              │ SCOPE_PRICING    → _handle_conversation() ──┘      │        │
│              │ SUMMARY          → _handle_summary()               │        │
│              │ EMAIL            → _handle_email()                 │        │
│              │ COMPLETED        → return static message           │        │
│              └─────────────────────────────────────────────────────┘        │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
          ┌────────────┼────────────────────────┐
          │            │                        │
          ▼            ▼                        ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────────────────────────────┐
│ Stages 1-4  │ │  Stage 5    │ │  Stage 6                                 │
│ Conversation│ │  Summary    │ │  Email                                   │
│ Agent  🗣️   │ │  Agent  📋  │ │  Agent  📧                               │
└──────┬──────┘ └──────┬──────┘ └──────────────────┬───────────────────────┘
       │               │                           │
       ▼               │                           │
┌──────────────────────────────────────┐           │
│  4. CONVERSATION AGENT (Stages 1-4) │           │
│     🗣️ LLM-Powered Dialogue         │           │
│                                      │           │
│     Step 4a: Intent Detection        │           │
│     • Is user asking a QUESTION?     │           │
│       (pattern matching on ?, who,   │           │
│        what, how, etc.)              │           │
│     • Or providing DATA?             │           │
│              │                       │           │
│              ▼                       │           │
│     Step 4b: RAG Context Retrieval   │           │
│     • Query KnowledgeBase (Pinecone) │           │
│     • Embed user message via Ollama  │           │
│     • Semantic search for relevant   │           │
│       company knowledge chunks       │           │
│     • Format results for LLM prompt  │           │
│              │                       │           │
│              ▼                       │           │
│     Step 4c: Build System Prompt     │           │
│     • Persona prompt (ColorWhistle   │           │
│       consultant persona)            │           │
│     • Stage-specific instructions    │           │
│       (what data to collect)         │           │
│     • RAG context (if available)     │           │
│     • Already-collected data context │           │
│       (to avoid re-asking)           │           │
│     • JSON extraction instructions   │           │
│              │                       │           │
│              ▼                       │           │
│     Step 4d: LLM Generation         │           │
│     • Send system + history +        │           │
│       user message to Ollama/Cloud   │           │
│     • Temperature: 0.4               │           │
│     • Expect structured JSON:        │           │
│       { response, extracted_data,    │           │
│         stage_complete }             │           │
│              │                       │           │
│              ▼                       │           │
│     Step 4e: Parse LLM Response      │           │
│     • Try JSON from code blocks      │           │
│     • Try JSON with "response" key   │           │
│     • Fallback: raw text response    │           │
│              │                       │           │
│              ▼                       │           │
│     Step 4f: Data Validation         │           │
│     • Reject garbage values          │           │
│     • Validate format (email regex,  │           │
│       phone digit count, name alpha) │           │
│     • Cross-reference with user msg  │           │
│       (anti-hallucination check)     │           │
│     • If question → force-clear      │           │
│       extracted data                 │           │
│              │                       │           │
│              ▼                       │           │
│     Return ConversationResult:       │           │
│     { reply, extracted_data,         │           │
│       should_advance }              │           │
└──────────────┬───────────────────────┘           │
               │                                   │
               ▼                                   │
┌──────────────────────────────────────┐           │
│  5. BACK IN ORCHESTRATOR             │           │
│     Post-Processing                  │           │
│                                      │           │
│     Step 5a: Apply Extracted Data    │           │
│     • Map fields to session model:   │           │
│       - personal_info: name, email  │           │
│       - tech_discovery: project_type,│           │
│         tech_stack, features,        │           │
│         integrations                 │           │
│       - scope_pricing: budget,       │           │
│         timeline, mvp_or_production, │           │
│         priority_features            │           │
│              │                       │           │
│              ▼                       │           │
│     Step 5b: Stage Transition Check  │           │
│     • If should_advance = true:      │           │
│       advance to next stage          │           │
│     • Auto-triggers:                 │           │
│       - SCOPE → SUMMARY:            │           │
│         auto-invoke Summary Agent    │ ◄─────────┤
│       - SUMMARY confirmed:          │           │
│         auto-invoke Email Agent      │ ◄─────────┘
│              │                       │
│              ▼                       │
│     Step 5c: Save Session            │
│     • Persist to MemoryStore         │
│     • Update timestamps              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  6. SUMMARY AGENT (Stage 5) — 📋 Auto-triggered                            │
│     • Builds context from collected data + conversation history             │
│     • LLM generates structured lead summary (temp: 0.3)                    │
│     • Presents summary to user for confirmation                             │
│     • User confirms → advance to Email stage                                │
│     • User requests changes → regenerate summary                            │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  7. EMAIL AGENT (Stage 6) — 📧 Auto-triggered                              │
│     • Compose 2 emails via LLM:                                             │
│       - Thank-you email → client                                            │
│       - Lead notification → admin team                                      │
│     • Mock-send (print to console + save to email_logs/)                    │
│     • Transition to COMPLETED stage                                         │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  8. RESPONSE RETURNED TO FRONTEND                                           │
│     ChatResponse { reply, stage, data_collected }                           │
│     • Frontend updates chat messages                                        │
│     • Stage indicator updates in header                                     │
│     • Data snapshot available for debugging                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Key Decision Points

| Decision Point | Logic | Type |
|----------------|-------|------|
| Stage routing | Based on `session.stage` enum | Code-driven |
| Question vs Data detection | Regex pattern matching on user message | Code-driven |
| RAG context retrieval | Embed question → Pinecone similarity search | AI-powered |
| Conversation response | LLM generates reply + extracts data as JSON | AI-powered |
| Data validation | Regex + cross-reference with user message | Code-driven |
| Stage advancement | Check if required fields are collected | Code-driven |
| Summary generation | LLM produces structured lead summary | AI-powered |
| Email composition | LLM composes personalized emails | AI-powered |

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
