# AI Agentic Conversational Chatbot

LLM-powered multi-agent chatbot for business lead qualification and requirement gathering.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript (Vite) |
| Backend | FastAPI (Python) |
| LLM | Ollama (local) / Cloud (OpenAI-compatible) |
| RAG | Pinecone + Ollama Embeddings |
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

### Widget Build

Build the embeddable widget for integration into external websites:

```bash
cd frontend
npm run build:widget
# Output: dist-widget/widget.js (single self-contained file)
```

Embed on any website with a single `<script>` tag:

```html
<script
  src="https://your-cdn.com/widget.js"
  data-api-url="https://api.colorwhistle.com"
  data-company-name="ColorWhistle"
  data-logo-url="https://colorwhistle.com/logo.svg"
  data-position="bottom-right"
></script>
```

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `data-api-url` | ✅ | `http://localhost:8000` | Backend API URL |
| `data-company-name` | ❌ | `ColorWhistle` | Company name in header |
| `data-logo-url` | ❌ | — | Logo image URL |
| `data-position` | ❌ | `bottom-right` | `bottom-right` or `bottom-left` |
| `data-greeting` | ❌ | — | Custom greeting message |

Test locally: open http://localhost:5173/test-widget.html

### Configuration

Copy `.env.example` to `.env` and configure:

```env
# LLM Provider
LLM_PROVIDER=ollama              # "ollama" or "cloud"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

CLOUD_API_KEY=your-key           # Only needed for cloud provider
CLOUD_BASE_URL=https://api.openai.com/v1
CLOUD_MODEL=gpt-4o-mini

# Conversation Limits
MAX_USER_MESSAGES=5              # Maximum interactions allowed per session

# Email (SMTP)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your-email@example.com
SMTP_PASSWORD=your-password
SMTP_FROM=noreply@colorwhistle.com
ADMIN_EMAILS=admin@colorwhistle.com

# RAG (Pinecone)
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=colorwhistle-kb
EMBEDDING_MODEL=nomic-embed-text
```

## Architecture

### Overview

Multi-agent system with 4 specialized agents operating in a **single-section conversation flow** — no rigid stage-based routing:

- **🎯 Orchestrator** — Coordinates agents, manages state & message limits (code-driven)
- **🗣️ Conversation Agent** — Natural dialogue with users in a single unified block (LLM-powered)
- **📋 Summarization Agent** — Generates dynamic lead summaries at completion (LLM-powered)
- **📧 Email Agent** — Composes & sends notification emails at completion (LLM-powered)

### Conversation Lifecycle

```
Welcome → Conversation (single block) → Limit Warning → Final Input → Completed
                                                                       ↓
                                                            Summary + Email (background)
```

| Stage | Description |
|-------|-------------|
| `welcome` | First-time greeting, transitions immediately to conversation |
| `conversation` | **Single unified block** — the agent naturally answers questions, extracts data (name, email, project details), and gently nudges for contact info. No sub-stages. |
| `limit_warning` | Triggered when `MAX_USER_MESSAGES` is reached. Asks if user wants to provide final details (Yes/No). |
| `final_input` | User provides remaining info in one message. Data is extracted, then session completes. |
| `completed` | Session closed. Summary generated & emails dispatched in the background. |

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
│              └── If new session → send welcome + process first message      │
│                                                                              │
│     Step 3b: Route based on current stage:                                  │
│              ┌─────────────────────────────────────────────────────┐        │
│              │ WELCOME        → _handle_welcome()                 │        │
│              │ CONVERSATION   → _handle_conversation()            │        │
│              │ LIMIT_WARNING  → _handle_limit_warning()           │        │
│              │ FINAL_INPUT    → _handle_final_input()             │        │
│              │ COMPLETED      → return static message             │        │
│              └─────────────────────────────────────────────────────┘        │
│                                                                              │
│     Step 3c: Track user message count against MAX_USER_MESSAGES             │
│              └── If limit reached → transition to LIMIT_WARNING             │
│                                                                              │
│     Step 3d: On completion → background task:                               │
│              └── Summarization Agent → Email Agent (async)                  │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
          ┌────────────┼─────────────────────────┐
          │            │                         │
          ▼            ▼                         ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────────────────┐
│ Conversation │ │  Completion  │ │  Completion                              │
│  Stage 🗣️   │ │  Summary 📋  │ │  Email  📧                               │
│   (active)   │ │ (background) │ │ (background)                             │
└──────┬───────┘ └──────┬───────┘ └──────────────────┬───────────────────────┘
       │                │                            │
       ▼                │                            │
┌──────────────────────────────────────┐             │
│  4. CONVERSATION AGENT               │             │
│     🗣️ LLM-Powered Dialogue          │             │
│     (Single unified block — no       │             │
│      sub-stages or rigid routing)    │             │
│                                      │             │
│     Step 4a: Intent Detection        │             │
│     • Is user asking a QUESTION?     │             │
│       → Answer-first flow            │             │
│     • Is it IRRELEVANT? (weather,    │             │
│       sports, math, etc.)            │             │
│       → Polite redirect              │             │
│     • Is it DATA provision?          │             │
│       → JSON extraction flow         │             │
│              │                       │             │
│              ▼                       │             │
│     Step 4b: RAG Context Retrieval   │             │
│     • Query KnowledgeBase (Pinecone) │             │
│     • Embed user message via Ollama  │             │
│     • Semantic search for relevant   │             │
│       company knowledge chunks       │             │
│              │                       │             │
│              ▼                       │             │
│     Step 4c: Build System Prompt     │             │
│     • Persona prompt (ColorWhistle   │             │
│       consultant persona)            │             │
│     • RAG context (if available)     │             │
│     • Already-collected data context │             │
│       (to avoid re-asking)           │             │
│     • JSON extraction instructions   │             │
│              │                       │             │
│              ▼                       │             │
│     Step 4d: LLM Generation         │             │
│     • Send system + history +        │             │
│       user message to Ollama/Cloud   │             │
│     • Temperature: 0.4               │             │
│     • Expect structured JSON:        │             │
│       { response, extracted_data }   │             │
│              │                       │             │
│              ▼                       │             │
│     Step 4e: Response Processing     │             │
│     • Parse JSON from code blocks    │             │
│     • Validate extracted data        │             │
│     • Regex fallback for name/email  │             │
│     • Append gentle nudge for        │             │
│       missing name/email (code-      │             │
│       generated, not LLM-generated)  │             │
│              │                       │             │
│              ▼                       │             │
│     Return ConversationResult:       │             │
│     { reply, extracted_data }        │             │
└──────────────┬───────────────────────┘             │
               │                                     │
               ▼                                     │
┌──────────────────────────────────────┐             │
│  5. BACK IN ORCHESTRATOR             │             │
│     Post-Processing                  │             │
│                                      │             │
│     Step 5a: Apply Extracted Data    │             │
│     • Map fields to session model:   │             │
│       - personal_info: name, email,  │             │
│         phone, company               │             │
│       - tech_discovery: project_type,│             │
│         tech_stack, features,        │             │
│         integrations                 │             │
│       - scope_pricing: budget,       │             │
│         timeline, mvp_or_production, │             │
│         priority_features            │             │
│              │                       │             │
│              ▼                       │             │
│     Step 5b: Message Limit Check     │             │
│     • If user_messages >= MAX:       │             │
│       transition to LIMIT_WARNING    │             │
│     • Otherwise: continue in         │             │
│       CONVERSATION stage             │             │
│              │                       │             │
│              ▼                       │             │
│     Step 5c: Save Session            │             │
│     • Persist to MemoryStore         │             │
│     • Update timestamps              │             │
└──────────────┬───────────────────────┘             │
               │                                     │
               ▼                                     │
┌──────────────────────────────────────────────────────────────────────────────┐
│  6. COMPLETION (Background Tasks)                                            │
│     Triggered when session reaches COMPLETED stage:                          │
│     • User declines final input (No at limit warning)                        │
│     • User submits final input                                               │
│     • User resets/exits session while active                                 │
│                                                                              │
│     📋 Summarization Agent                                                   │
│     • Builds context from collected data + conversation history              │
│     • LLM generates dynamic, conversation-based summary (temp: 0.3)         │
│     • Only includes data actually discussed (no placeholders)                │
│                                                                              │
│     📧 Email Agent                                                           │
│     • Requires user email — skips all emails if no email provided            │
│     • Composes 2 emails via LLM:                                             │
│       - Thank-you email → client                                             │
│       - Lead notification → admin team                                       │
│     • Sends via SMTP (or mock-sends to console if not configured)            │
│     • Saves to email_logs/ directory                                         │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  7. RESPONSE RETURNED TO FRONTEND                                           │
│     ChatResponse { reply, stage, data_collected }                           │
│     • Frontend updates chat messages                                        │
│     • Stage indicator updates in header                                     │
│     • Data snapshot available for debugging                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| **Single-section conversation** | No sub-stage routing (personal_info → tech → scope) | The LLM naturally handles mixed conversations; rigid stages felt robotic |
| **Message limit enforcement** | `MAX_USER_MESSAGES` config, code-driven | Prevents infinite sessions; gives user a graceful exit path |
| **Data extraction** | LLM JSON extraction + regex fallback | Small models sometimes fail JSON; regex catches name/email reliably |
| **Gentle nudges** | Code-appended (not LLM-generated) | Prevents the LLM from repeating nudges or generating awkward asks |
| **Budget handling** | Hard-coded redirect to human team | Business rule: never suggest pricing via chatbot |
| **Irrelevant question filter** | Regex patterns before LLM call | Saves LLM calls for off-topic questions (weather, sports, etc.) |
| **Summary/email dispatch** | Background async tasks | User gets immediate response; emails fire asynchronously |
| **Email gate** | Requires user email address | No email = no notifications dispatched (prevents empty leads) |

#### Key Decision Points

| Decision Point | Logic | Type |
|----------------|-------|------|
| Stage routing | Based on `session.stage` enum | Code-driven |
| Question vs Data detection | Regex pattern matching on user message | Code-driven |
| Irrelevant question filtering | Regex patterns + relevant keyword whitelist | Code-driven |
| RAG context retrieval | Embed question → Pinecone similarity search | AI-powered |
| Conversation response | LLM generates reply + extracts data as JSON | AI-powered |
| Data validation | Regex + cross-reference with user message | Code-driven |
| Message limit enforcement | Count user messages against MAX_USER_MESSAGES | Code-driven |
| Summary generation | LLM produces dynamic, conversation-based summary | AI-powered |
| Email composition | LLM composes personalized emails | AI-powered |
| Email dispatch gate | Check if user provided an email address | Code-driven |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a message and get AI response |
| GET | `/api/session/{id}` | Get full session state |
| POST | `/api/reset` | Reset a session |
| POST | `/api/exit` | Trigger early exit (summary + email in background) |
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
│   │   ├── conversation_agent.py      # 🗣️ LLM-powered dialogue (single block)
│   │   ├── summarization_agent.py     # 📋 LLM-powered summary generator
│   │   ├── email_agent.py            # 📧 LLM-powered email composer + SMTP
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
│   │   │   ├── ChatWindow.tsx        # Chat UI component (page + widget mode)
│   │   │   └── ChatWindow.css        # Chat styling
│   │   ├── pages/
│   │   │   └── ChatPage.tsx          # Main chat page (standalone)
│   │   ├── widget/                   # 🔌 Embeddable widget module
│   │   │   ├── WidgetLauncher.tsx    # FAB + expandable panel
│   │   │   ├── WidgetHeader.tsx      # Compact widget header
│   │   │   ├── widget-entry.tsx      # IIFE entry point
│   │   │   └── widget.css           # Widget-scoped styles
│   │   ├── api.ts                    # API client (configurable URL)
│   │   ├── App.tsx                   # Root component
│   │   ├── main.tsx                  # Entry point
│   │   └── index.css                 # Global design system
│   ├── vite.widget.config.ts         # Widget build config (IIFE + CSS inject)
│   ├── test-widget.html              # Widget integration test page
│   ├── dist-widget/                  # Built widget output
│   │   └── widget.js                 # Self-contained embeddable bundle
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
| Phase 10 | Embeddable Widget | ✅ Complete |

## Running Tests

```bash
cd backend
.\venv\Scripts\activate
python test_agents.py     # Agent tests (Phases 3-5)
python test_phase2.py     # Schema & session tests (Phase 2)
```

## License

Private — ColorWhistle
