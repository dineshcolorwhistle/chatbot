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
let API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

/**
 * Pinecone namespace for tenant-scoped KB queries.
 * Set via VITE_NAMESPACE env var (standalone) or setNamespace() (widget mode).
 */
let NAMESPACE: string | undefined = import.meta.env.VITE_NAMESPACE || undefined;

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

/**
 * Override the namespace at runtime.
 * Called by the widget entry point during initialization.
 *
 * @param ns The Pinecone namespace for this widget deployment.
 */
export function setNamespace(ns: string): void {
  NAMESPACE = ns;
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
  namespace: string | null;
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

export function getAuthToken(): string | null {
  return localStorage.getItem("admin_token");
}

export function setAuthToken(token: string) {
  localStorage.setItem("admin_token", token);
  localStorage.setItem("admin_last_activity", Date.now().toString());
}

export function removeAuthToken() {
  localStorage.removeItem("admin_token");
  localStorage.removeItem("admin_last_activity");
}

async function adminFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401) {
    removeAuthToken();
    window.location.href = "/admin/login";
  }
  return response;
}

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
  
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      
      if (response.status === 401 && url.includes("/admin")) {
        removeAuthToken();
        window.location.href = "/admin/login";
      }
      
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
  message: string,
  namespace?: string
): Promise<ChatResponse> {
  const response = await safeFetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
      namespace: namespace || NAMESPACE || undefined,
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
    `${API_BASE_URL}/health`,
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

/**
 * Upload PDF documents for a specific namespace.
 */
export async function uploadDocuments(
  namespace: string,
  files: FileList | File[]
): Promise<{ message: string; saved_files: string[]; stats: any }> {
  const formData = new FormData();
  formData.append("namespace", namespace);
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  const token = getAuthToken();
  const headers = new Headers();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  const response = await fetch(`${API_BASE_URL}/admin/upload`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Extract YouTube transcript and download as a PDF directly.
 */
export async function extractYoutube(
  url: string
): Promise<void> {
  const response = await safeFetch(`${API_BASE_URL}/admin/extract-youtube`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  }, 0);

  if (!response.ok) {
    const errorText = await response.text();
    let errorDetail = `Extraction failed: ${response.status}`;
    try {
      const errorJson = JSON.parse(errorText);
      if (errorJson.detail) errorDetail = errorJson.detail;
    } catch(e) {}
    throw new Error(errorDetail);
  }

  // Get filename from Content-Disposition header if possible, else default
  let filename = "youtube_transcript.pdf";
  const contentDisposition = response.headers.get("content-disposition");
  if (contentDisposition && contentDisposition.includes("filename=")) {
    const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
    if (filenameMatch && filenameMatch.length >= 2) {
      filename = filenameMatch[1];
    }
  }

  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = downloadUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(downloadUrl);
  document.body.removeChild(a);
}

// ============================================
// Namespace Management
// ============================================

export interface NamespaceInfo {
  name: string;
  vector_count: number;
  has_local_docs: boolean;
}

export interface NamespacesResponse {
  namespaces: NamespaceInfo[];
  total_vectors: number;
  index_name: string;
}

export interface NamespaceDeleteResponse {
  success: boolean;
  namespace: string;
  vectors_cleared: boolean;
  local_files_deleted: boolean;
  message: string;
}

/**
 * List all Pinecone namespaces with their vector counts.
 */
export async function listNamespaces(): Promise<NamespacesResponse> {
  const response = await safeFetch(`${API_BASE_URL}/admin/namespaces`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to list namespaces: ${response.status}`);
  }
  return response.json();
}

/**
 * Delete an entire namespace from Pinecone (and optionally local files).
 */
export async function deleteNamespace(
  namespace: string,
  deleteLocalFiles: boolean = true
): Promise<NamespaceDeleteResponse> {
  const response = await safeFetch(`${API_BASE_URL}/admin/namespace/${encodeURIComponent(namespace)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delete_local_files: deleteLocalFiles }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to delete namespace: ${response.status}`);
  }
  return response.json();
}

/**
 * Get the current status of the daily summary cron job.
 */
export async function getCronStatus(): Promise<{ enabled: boolean }> {
  const response = await safeFetch(`${API_BASE_URL}/admin/cron-status`);
  if (!response.ok) {
    throw new Error(`Failed to get cron status: ${response.status}`);
  }
  return response.json();
}

/**
 * Enable or disable the daily summary cron job.
 */
export async function setCronStatus(enabled: boolean): Promise<{ enabled: boolean }> {
  const response = await safeFetch(`${API_BASE_URL}/admin/cron-status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  }, 0);
  if (!response.ok) {
    const errorText = await response.text();
    let errorDetail = `Failed to set cron status: ${response.status}`;
    try {
      const errorJson = JSON.parse(errorText);
      if (errorJson.detail) errorDetail = errorJson.detail;
    } catch(e) {}
    throw new Error(errorDetail);
  }
  return response.json();
}

// ============================================
// Auth Management
// ============================================

export async function login(email: string, password: string): Promise<{ access_token: string }> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Login failed");
  }
  return response.json();
}

export async function createAdmin(name: string, email: string): Promise<any> {
  const response = await adminFetch(`${API_BASE_URL}/auth/create-admin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to create admin");
  }
  return response.json();
}

export async function setPassword(token: string, password: string): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/auth/set-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, password }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to set password");
  }
  return response.json();
}

export async function listAdmins(): Promise<any[]> {
  const response = await adminFetch(`${API_BASE_URL}/auth/list`);
  if (!response.ok) {
    throw new Error("Failed to list admins");
  }
  return response.json();
}
