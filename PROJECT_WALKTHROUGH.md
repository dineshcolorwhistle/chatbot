# Project Walkthrough — AI Agentic Conversational Chatbot

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Multi-Agent System](#multi-agent-system)
- [Conversation Flow](#conversation-flow)
- [Folder Structure](#folder-structure)
- [Request Lifecycle](#request-lifecycle)
- [API Contracts](#api-contracts)
- [LLM Provider Design](#llm-provider-design)
- [Implementation Phases](#implementation-phases)
- [Phase Progress Tracker](#phase-progress-tracker)
- [Development Rules](#development-rules)
- [Future Migration Paths](#future-migration-paths)

---

## Overview

A local MVP **AI agentic conversational chatbot** for **business lead qualification and requirement gathering**.

The system uses **multiple specialized AI agents**, each handling a distinct responsibility, orchestrated through a central coordinator. All conversational intelligence is powered by LLMs (Ollama local / cloud models).

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript (Vite) |
| Backend | FastAPI (Python) |
| LLM Provider | Ollama (local), Cloud models (OpenAI-compatible) |
| Session Storage | In-memory Python dictionary |
| Email | Mock email service (console output / log file) |
| Environment | Localhost development |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + TS)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  ChatPage.tsx │  │ChatWindow.tsx│  │      api.ts        │    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘    │
└─────────┼─────────────────┼───────────────────┼────────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                   routes/chat.py                         │     │
│  │              (Thin handlers, no logic)                    │     │
│  └──────────────────────┬──────────────────────────────────┘     │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────────┐     │
│  │              🎯 Orchestrator Agent                       │     │
│  │         (Code-driven, deterministic)                     │     │
│  │    Manages state, stages, agent coordination             │     │
│  └────┬─────────────────┬─────────────────┬────────────────┘     │
│       │                 │                 │                       │
│  ┌────▼─────┐     ┌────▼─────┐     ┌────▼─────┐                │
│  │🗣️ Convo  │     │📋 Summary│     │📧 Email  │                │
│  │  Agent   │     │  Agent   │     │  Agent   │                │
│  └────┬─────┘     └────┬─────┘     └────┬─────┘                │
│       │                 │                 │                       │
│  ┌────▼─────────────────▼─────────────────▼────────────────┐     │
│  │                LLM Provider Layer                        │     │
│  │     ┌──────────────┐    ┌──────────────────┐            │     │
│  │     │ Ollama Local │    │ Cloud (OpenAI    │            │     │
│  │     │ (localhost)  │    │  Compatible API) │            │     │
│  │     └──────────────┘    └──────────────────┘            │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │              In-Memory Session Store                      │     │
│  └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent System

### Agent 1 — 🎯 Orchestrator (Code-Driven)
- **Role:** Coordinate all agents and manage workflow
- **LLM:** ❌ Not LLM-powered (deterministic logic)
- **Responsibilities:**
  - Manage conversation state and stage transitions
  - Route messages to the Conversation Agent
  - Detect when data collection is complete
  - Trigger Summarization Agent at the right time
  - Trigger Email Agent after summary confirmation
  - Handle session lifecycle (create, track, reset)

### Agent 2 — 🗣️ Conversation Agent (LLM-Powered)
- **Role:** Lead natural dialogue with users
- **LLM:** ✅ Every user message
- **Triggered:** Stages 1–4
- **Responsibilities:**
  - Guide users through lead qualification flow
  - Ask intelligent, context-aware follow-up questions
  - Validate responses naturally (email, phone formats)
  - Handle off-topic responses and redirect gracefully
  - Extract structured data from free-form conversation

### Agent 3 — 📋 Summarization Agent (LLM-Powered)
- **Role:** Generate structured lead summaries
- **LLM:** ✅ Single invocation per session
- **Triggered:** Stage 5
- **Responsibilities:**
  - Take conversation history and extracted data
  - Generate clean, structured lead summary
  - Highlight key requirements, priorities, concerns

### Agent 4 — 📧 Email Agent (LLM-Powered)
- **Role:** Compose notification emails
- **LLM:** ✅ Single invocation per session
- **Triggered:** Stage 6
- **Responsibilities:**
  - Compose professional thank-you email for user
  - Compose detailed lead notification for admin
  - For MVP: print to console / log file

---

## Conversation Flow

```
[Start] → Stage 1: Welcome
              ↓
         Stage 2: Personal Details (name, email, phone, company)
              ↓
         Stage 3: Technical Discovery (project type, tech, features)
              ↓
         Stage 4: Scope & Pricing (budget, timeline, MVP vs prod)
              ↓
         Stage 5: Summary Generation (📋 Summarization Agent)
              ↓
         Stage 6: Email Notification (📧 Email Agent)
              ↓
          [End]
```

**Stages 1–4:** Conversation Agent handles all dialogue  
**Stage 5:** Summarization Agent generates summary  
**Stage 6:** Email Agent composes & sends mock emails  

---

## Folder Structure

```
chatbot/
├── backend/
│   ├── main.py                        # FastAPI entry point, CORS, lifespan
│   ├── config.py                      # Settings loaded from .env
│   ├── .env                           # Environment variables (API keys, config)
│   ├── .env.example                   # Example env file (committed to git)
│   ├── requirements.txt               # Python dependencies
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py                    # Thin API handlers → orchestrator
│   ├── services/
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # 🎯 Agent coordinator & state manager
│   │   ├── conversation_agent.py      # 🗣️ LLM-powered dialogue agent
│   │   ├── summarization_agent.py     # 📋 LLM-powered summary generator
│   │   └── email_agent.py            # 📧 LLM-powered email composer
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic request/response models
│   └── providers/
│       ├── __init__.py
│       ├── base.py                   # Abstract LLM provider interface
│       └── ollama_provider.py        # Ollama local implementation
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatWindow.tsx
│   │   ├── pages/
│   │   │   └── ChatPage.tsx
│   │   ├── api.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html
│   └── package.json
│
├── PROJECT_SCOPE.md
├── PROJECT_WALKTHROUGH.md
└── README.md
```

---

## Request Lifecycle

```
User types message
       ↓
Frontend sends POST /api/chat { session_id, message }
       ↓
Routes (chat.py) → delegates to Orchestrator
       ↓
Orchestrator loads/creates session
       ↓
Orchestrator checks current stage:
  ├── Stage 1-4 → Conversation Agent → LLM → response + extracted data
  ├── Stage 5   → Summarization Agent → LLM → structured summary
  └── Stage 6   → Email Agent → LLM → composed emails → mock send
       ↓
Orchestrator saves updated session
       ↓
Response returned to frontend
       ↓
Frontend displays bot message
```

---

## API Contracts

### POST /api/chat
```json
// Request
{
  "session_id": "user-001",
  "message": "Hello"
}

// Response
{
  "reply": "Welcome! I'm here to help understand your project needs...",
  "stage": "personal_info",
  "data_collected": {}
}
```

### GET /api/session/{session_id}
Returns full session data (conversation history, current stage, collected fields).

### POST /api/reset
Resets session state for a given session ID.

---

## LLM Provider Design

### Provider Interface
```python
class LLMProvider(ABC):
    async def generate(self, system_prompt: str, messages: list, temperature: float = 0.7) -> str:
        """Generate a response from the LLM."""
        ...
```

### Supported Providers

| Provider | Use Case | API Key Required | Endpoint |
|----------|----------|-----------------|----------|
| **Ollama Local** | Free, private, local dev | ❌ No | `http://localhost:11434` |
| **Cloud (OpenAI-compatible)** | Production-quality, any cloud LLM | ✅ Yes | Configurable |

### Configuration (.env)
```env
LLM_PROVIDER=ollama              # ollama or cloud
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

CLOUD_API_KEY=your-api-key-here
CLOUD_BASE_URL=https://api.openai.com/v1
CLOUD_MODEL=gpt-4o-mini
```

---

## Implementation Phases

### Phase 1 — LLM Provider Layer ⬅️ START HERE
- `providers/base.py` — Abstract LLM interface
- `providers/ollama_provider.py` — Ollama local implementation
- `config.py` — Settings from .env
- `.env` / `.env.example` — Environment configuration
- `requirements.txt` — Python dependencies

### Phase 2 — Models & Session Store
- `models/schemas.py` — Pydantic request/response models
- Session management in orchestrator

### Phase 3 — Conversation Agent
- `services/conversation_agent.py`
- System prompts for each stage
- Data extraction from conversation

### Phase 4 — Summarization Agent
- `services/summarization_agent.py`
- Summary generation from collected data

### Phase 5 — Email Agent
- `services/email_agent.py`
- Email composition + mock sending

### Phase 6 — Orchestrator
- `services/orchestrator.py`
- Agent coordination, stage transitions

### Phase 7 — API Routes & Backend Integration
- `routes/chat.py`, `main.py`
- FastAPI endpoints, CORS, error handling

### Phase 8 — Frontend
- React + TypeScript chat UI
- API client, components, pages

### Phase 9 — Polish & Documentation
- Styling, UX polish, error handling
- README, API docs, testing

---

## Phase Progress Tracker

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

---

## Development Rules

1. ✅ Use LLM-powered agents for all conversational intelligence
2. ✅ Keep agents modular and single-responsibility
3. ✅ Use abstract provider interface for LLM swappability
4. ✅ No business logic in routes — delegate to orchestrator/services
5. ✅ Update README and walkthrough after every module
6. ✅ Code must be clean, modular, and scalable
7. ❌ Do NOT use production DB — in-memory sessions only
8. ❌ Do NOT use Docker — local dev only
9. ❌ Do NOT create widget version yet

---

## Future Migration Paths

| Future Feature | Design Decision Now |
|---------------|-------------------|
| Redis sessions | Abstract session interface |
| PostgreSQL | Pydantic models map to ORM easily |
| New LLM providers | Abstract provider interface |
| Widget embedding | Chat component is self-contained |
| WordPress integration | REST API already compatible |
| CRM integration | Add webhook/adapter in services |
| Real email (SendGrid/SES) | Swap email agent's send method |
| Multi-language | Prompt-based, easy to add |
| Additional agents | Orchestrator pattern supports adding agents |
