/**
 * API Client — Backend Communication Layer
 *
 * Provides typed functions to interact with the FastAPI backend.
 * All API calls go through this module — components never make
 * raw fetch calls directly.
 */

const API_BASE_URL = "http://localhost:8000/api";

// ============================================
// Types
// ============================================

export interface ChatResponse {
  reply: string;
  stage: string;
  data_collected: Record<string, string>;
}

export interface SessionResponse {
  session_id: string;
  stage: string;
  collected_data: {
    personal_info: {
      name: string | null;
      email: string | null;
      phone: string | null;
      company: string | null;
    };
    tech_discovery: {
      project_type: string | null;
      tech_stack: string | null;
      features: string | null;
      integrations: string | null;
    };
    scope_pricing: {
      budget: string | null;
      timeline: string | null;
      mvp_or_production: string | null;
      priority_features: string | null;
    };
  };
  conversation_history: Array<{
    role: string;
    content: string;
    timestamp: string;
  }>;
  summary: string | null;
  created_at: string;
  updated_at: string;
}

export interface ResetResponse {
  message: string;
  session_id: string;
}

export interface HealthResponse {
  status: string;
  llm_provider: {
    healthy: boolean;
    provider: string | null;
  };
}

// ============================================
// API Functions
// ============================================

/**
 * Send a chat message and get the AI response.
 */
export async function sendMessage(
  sessionId: string,
  message: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get the full state of a session.
 */
export async function getSession(
  sessionId: string
): Promise<SessionResponse> {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error("Session not found");
    }
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Reset a session, clearing all conversation state.
 */
export async function resetSession(
  sessionId: string
): Promise<ResetResponse> {
  const response = await fetch(`${API_BASE_URL}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Handle early session exit (e.g. user leaves page) without deleting history.
 */
export async function exitSession(
  sessionId: string
): Promise<{ status: string; triggered: boolean }> {
  // Using keepalive allows the request to complete even if the page closes
  const response = await fetch(`${API_BASE_URL}/exit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    keepalive: true,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Check backend health status.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch("http://localhost:8000/health");

  if (!response.ok) {
    throw new Error("Health check failed");
  }

  return response.json();
}

/**
 * Generate a unique session ID.
 */
export function generateSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `user-${timestamp}-${random}`;
}
