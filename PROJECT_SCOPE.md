# Project Scope — AI Agentic Conversational Chatbot

## Overview

Build a local MVP **AI agentic conversational chatbot** application for **business lead qualification and requirement gathering**.

The system uses **multiple specialized AI agents**, each handling a distinct responsibility, orchestrated through a central coordinator.

**Development Phase:** Local development only (initial phase).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript (Vite) |
| Backend | FastAPI (Python) |
| LLM Provider | OpenAI API (default), swappable to Ollama / Qwen / Gemini |
| Session Storage | In-memory Python dictionary |
| Email | Mock email service (console output / log file) |
| IDE | Antigravity IDE |
| Environment | Localhost development |

---

## Project Goal

Create an AI conversational chatbot web application powered by **LLM-based agents** that interact with users in a structured conversation flow.

The chatbot collects:

### 1. Personal Information
- Full Name
- Email Address
- Contact Number
- Company Name (optional)

### 2. Project Requirement Details
- Project Type
- Technology Preference
- Required Features / Modules
- Scope Details
- Timeline
- Budget

### 3. Final Summary
- AI-generated structured summary of the entire conversation

### 4. Email Notification
- AI-composed notification emails sent to user and admin
- For local MVP: print email content to backend logs / console

---

## Multi-Agent Architecture

The system uses **4 specialized AI agents**, each with a single responsibility:

### Agent 1 — Conversation Agent
**Role:** Lead the conversation with the user  
**Responsibilities:**
- Guide the user through the lead qualification flow
- Ask intelligent, context-aware follow-up questions
- Validate user responses naturally (e.g., email format, phone format)
- Handle off-topic responses gracefully and redirect
- Maintain conversational tone while staying on track
- Transition between conversation stages naturally

**LLM Usage:** Every user message is processed through the LLM with a system prompt defining the agent's personality, current stage, and collected data.

### Agent 2 — Summarization Agent
**Role:** Generate structured summaries  
**Responsibilities:**
- Take the full conversation history and extracted data
- Generate a clean, structured lead summary
- Format the summary for display and email inclusion
- Highlight key requirements, priorities, and concerns

**LLM Usage:** Invoked once after all data is collected. Receives conversation history + extracted fields as context.

### Agent 3 — Email Agent
**Role:** Compose and send notification emails  
**Responsibilities:**
- Compose a professional thank-you email for the user
- Compose a detailed lead notification email for the admin
- Include the structured summary in both emails
- For MVP: print formatted email content to console / log file

**LLM Usage:** Invoked after summarization. Uses the summary + user data to compose personalized emails.

### Agent 4 — Orchestrator Agent
**Role:** Coordinate all agents and manage workflow  
**Responsibilities:**
- Manage conversation state and stage transitions
- Route incoming messages to the Conversation Agent
- Detect when data collection is complete
- Trigger Summarization Agent at the right time
- Trigger Email Agent after summary is confirmed
- Handle session lifecycle (create, track, reset)

**Implementation:** Primarily code-driven logic (not LLM) that coordinates the LLM-powered agents. This keeps the orchestration deterministic and predictable.

---

## Conversation Flow

### Stage 1 — Welcome
- Display welcome message
- Brief introduction of the chatbot's purpose
- Begin collecting personal information

### Stage 2 — Personal Details
- Collect: Name, Email, Phone, Company Name
- Ask one question at a time
- Validate responses naturally through conversation
- Store each response in session state

### Stage 3 — Technical Discovery
- Collect: Project type, tech stack, features, integrations
- Allow free-form discussion
- Ask clarifying questions when responses are vague
- Examples: AI chatbot, MERN app, FastAPI backend, WordPress integration, admin dashboard, payment integration

### Stage 4 — Scope & Pricing
- Collect: Budget, timeline, MVP vs production, priority features
- Handle sensitive topics (budget) diplomatically

### Stage 5 — Summary Generation
- Trigger Summarization Agent
- Display structured summary to user
- Ask for confirmation or corrections

### Stage 6 — Email Notification
- Trigger Email Agent
- Compose and send (mock) emails
- Display confirmation to user
- End conversation gracefully

---

## Backend Requirements

### API Endpoints

#### POST /api/chat
Process a user message and return the AI response.

**Request:**
```json
{
  "session_id": "user-001",
  "message": "Hello"
}
```

**Response:**
```json
{
  "reply": "Welcome! I'm here to help understand your project needs. May I know your full name?",
  "stage": "personal_info",
  "data_collected": {}
}
```

#### GET /api/session/{session_id}
Return full session data including conversation history, current stage, and collected fields.

#### POST /api/reset
Reset session state for a given session ID.

### Session Management
Use in-memory session storage.

```python
sessions = {
    "user-001": {
        "stage": "personal_info",
        "collected_data": {
            "name": "DK",
            "email": "dk@email.com"
        },
        "conversation_history": [...],
        "summary": None
    }
}
```

Session must persist while application is running.

---

## Frontend Requirements

Build a clean, modern React chat interface.

### Required UI Elements
- Chat message list (bot + user bubbles)
- User input field with send button
- Typing indicator for bot responses
- Auto-scroll to latest message
- Summary display card
- Session ID support
- Responsive design

The UI should behave like a professional chat application.

---

## Folder Structure

```
chatbot/
├── backend/
│   ├── main.py                       # FastAPI entry point, CORS, lifespan
│   ├── config.py                     # LLM provider config, API keys, settings
│   ├── routes/
│   │   └── chat.py                   # API route handlers (thin, no business logic)
│   ├── services/
│   │   ├── orchestrator.py           # Orchestrator — coordinates agents & state
│   │   ├── conversation_agent.py     # Conversation Agent — handles user dialogue
│   │   ├── summarization_agent.py    # Summarization Agent — generates summaries
│   │   └── email_agent.py           # Email Agent — composes & sends emails
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   └── providers/
│       ├── base.py                  # Abstract LLM provider interface
│       └── openai_provider.py       # OpenAI implementation (default)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatWindow.tsx       # Chat UI component
│   │   ├── pages/
│   │   │   └── ChatPage.tsx         # Main chat page
│   │   ├── api.ts                   # API client
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   └── tsconfig.json
│
├── PROJECT_SCOPE.md
└── README.md
```

---

## LLM Provider Design

### Provider Interface
All LLM interactions go through an abstract provider interface:

```python
class LLMProvider:
    async def generate(self, system_prompt: str, messages: list, temperature: float) -> str:
        raise NotImplementedError
```

### Default Provider: OpenAI
- Uses `openai` Python package
- Model: `gpt-4o-mini` (cost-effective for MVP)
- API key loaded from environment variable

### Swappable Providers
The provider interface allows easy switching to:
- **Ollama** — local models (Llama, Mistral)
- **Qwen** — Alibaba Cloud
- **Gemini** — Google AI
- **Any OpenAI-compatible API**

---

## Agent Prompt Design

Each agent receives a carefully crafted system prompt:

### Conversation Agent Prompt
- Persona: Professional, friendly project consultant
- Awareness of current stage and collected data
- Instructions to ask one question at a time
- Validation rules for email, phone, etc.
- Redirection instructions for off-topic messages

### Summarization Agent Prompt
- Instructions to produce structured output
- Required format specification
- Fields to extract and organize

### Email Agent Prompt
- Tone: Professional, warm
- Email structure template
- Instructions for user email vs admin email differences

---

## Important Development Rules

1. ✅ Use LLM-powered agents for all conversational intelligence
2. ✅ Keep agents modular and single-responsibility
3. ✅ Use abstract provider interface for LLM swappability
4. ✅ No business logic in routes — delegate to orchestrator/services
5. ✅ Code must be clean, modular, and scalable
6. ❌ Do NOT use production DB — in-memory sessions only
7. ❌ Do NOT use Docker — local dev only
8. ❌ Do NOT create widget version yet

---

## Future-Ready Design

Keep code structured for future upgrades:

| Future Feature | Design Decision Now |
|---------------|-------------------|
| Redis sessions | Abstract session interface |
| PostgreSQL | Add ORM layer later |
| Ollama / Qwen / Gemini | Provider interface ready |
| Widget embedding | Chat component is self-contained |
| WordPress integration | REST API already compatible |
| CRM integration | Add webhook/adapter in services |
| Real email (SendGrid/SES) | Swap email agent's send method |
| Multi-language support | Prompt-based, easy to add |
