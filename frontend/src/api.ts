/**
 * API Client — Backend Communication Layer
 *
 * Provides typed functions to interact with the FastAPI backend.
 * All API calls go through this module — components never make
 * raw fetch calls directly.
 *
 * Features:
 *   - Typed request/response interfaces
 *   - Network error handling with user-friendly messages
 *   - Automatic retry for transient failures
 *   - Timeout protection for slow LLM responses
 */

/**
 * API base URL — defaults to localhost for standalone mode.
 * In widget mode, this is overridden via setApiBaseUrl()
 * which reads from the <script data-api-url="..."> attribute.
 */
let API_BASE_URL = "http://localhost:8000/api";

/**
 * Override the API base URL at runtime.
 * Called by the widget entry point during initialization.
 *
 * @param baseUrl The backend base URL (e.g. "https://api.colorwhistle.com")
 *                — the "/api" suffix is appended automatically if not present.
 */
export function setApiBaseUrl(baseUrl: string): void {
  const cleaned = baseUrl.replace(/\/+$/, ""); // strip trailing slashes
  API_BASE_URL = cleaned.endsWith("/api") ? cleaned : `${cleaned}/api`;
}

/** Get the current API base URL (useful for debugging). */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

/** Default timeout for API calls (ms) — generous for LLM processing */
const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes

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
// Internal Helpers
// ============================================

/**
 * Wraps a fetch call with timeout, network error handling, and retry logic.
 *
 * Converts cryptic browser errors like "TypeError: Failed to fetch" into
 * clear, actionable messages the user can understand.
 */
async function safeFetch(
  url: string,
  options: RequestInit = {},
  retries = 1,
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));

      // Abort errors mean timeout — don't retry
      if (lastError.name === "AbortError") {
        throw new Error(
          "The request timed out. The server may be processing a complex request — please try again."
        );
      }

      // For network errors, retry once (server might be starting up)
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 1000)); // 1s backoff
        continue;
      }
    }
  }

  // All retries exhausted — provide a clear message
  if (
    lastError &&
    (lastError.message.includes("Failed to fetch") ||
      lastError.message.includes("NetworkError") ||
      lastError.message.includes("ERR_CONNECTION_REFUSED") ||
      lastError.message.includes("Load failed"))
  ) {
    throw new Error(
      "Cannot connect to the server. Please make sure the backend is running on http://localhost:8000 and try again."
    );
  }

  throw lastError || new Error("An unexpected network error occurred.");
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
  const response = await safeFetch(`${API_BASE_URL}/chat`, {
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
  const response = await safeFetch(`${API_BASE_URL}/session/${sessionId}`);

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
  const response = await safeFetch(`${API_BASE_URL}/reset`, {
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
  const response = await safeFetch(
    `${API_BASE_URL}/exit`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
      keepalive: true,
    },
    0, // No retries for exit — fire-and-forget
    10_000 // Short timeout
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Check backend health status.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await safeFetch(
    "http://localhost:8000/health",
    {},
    0, // No retries for health checks
    5_000 // 5 second timeout
  );

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
