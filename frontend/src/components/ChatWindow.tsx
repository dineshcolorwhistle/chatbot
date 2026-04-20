/**
 * ChatWindow Component
 *
 * The main chat interface component that handles:
 *   - Message display (user + assistant bubbles)
 *   - User input with send button
 *   - Typing indicator during AI processing
 *   - Auto-scroll to latest message
 *   - Session management (start, reset)
 */

import { useState, useRef, useEffect, useCallback } from "react";
import {
  sendMessage,
  resetSession,
  exitSession,
  generateSessionId,
  type ChatResponse,
} from "../api";
import "./ChatWindow.css";

// ============================================
// Types
// ============================================

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}



// ============================================
// Component
// ============================================

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [currentStage, setCurrentStage] = useState("welcome");
  const [collectedData, setCollectedData] = useState<Record<string, string>>(
    {}
  );
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Initialize session
  useEffect(() => {
    const stored = sessionStorage.getItem("chatbot_session_id");
    if (stored) {
      setSessionId(stored);
    } else {
      const newId = generateSessionId();
      setSessionId(newId);
      sessionStorage.setItem("chatbot_session_id", newId);
    }
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle page exit
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (sessionId && currentStage !== "completed" && currentStage !== "welcome") {
        exitSession(sessionId).catch(() => {});
      }
    };
    
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [sessionId, currentStage]);

  // Focus input after loading
  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  };

  // Send message
  const handleSend = useCallback(async (messageOverride?: string) => {
    const textToSend = messageOverride !== undefined ? messageOverride : input;
    const trimmed = textToSend.trim();
    if (!trimmed || isLoading || !sessionId) return;

    setError(null);

    // Add user message
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    
    if (messageOverride === undefined) {
      setInput("");
      // Reset textarea height
      if (inputRef.current) {
        inputRef.current.style.height = "auto";
      }
    }

    setIsLoading(true);

    try {
      const response: ChatResponse = await sendMessage(sessionId, trimmed);

      // Add assistant message
      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: response.reply,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Update stage and collected data
      setCurrentStage(response.stage);
      setCollectedData(response.data_collected || {});
    } catch (err) {
      const errorMsg =
        err instanceof Error ? err.message : "An unexpected error occurred";
      setError(errorMsg);

      // Add error as system message with the actual error detail
      const isConnectionError = errorMsg.toLowerCase().includes("connect") ||
        errorMsg.toLowerCase().includes("server");
      const errMsg: Message = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: isConnectionError
          ? `⚠️ ${errorMsg}`
          : "I'm sorry, I encountered an error. Please try again or reset the session.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, sessionId]);

  // Handle Enter key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Reset session
  const handleReset = async () => {
    if (!sessionId) return;

    try {
      await resetSession(sessionId);
    } catch {
      // Session might not exist on backend, that's OK
    }

    // Generate new session
    const newId = generateSessionId();
    setSessionId(newId);
    sessionStorage.setItem("chatbot_session_id", newId);

    // Clear state
    setMessages([]);
    setCurrentStage("welcome");
    setCollectedData({});
    setError(null);
    setInput("");
  };



  return (
    <div className="chat-container">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-left">
          <div className="chat-logo">
            <img src="/logo.svg" alt="ColorWhistle Logo" className="logo-icon" />
            <div className="logo-text">
              <h1>ColorWhistle</h1>
              <span className="subtitle">AI Project Consultant</span>
            </div>
          </div>
        </div>
        <div className="chat-header-right">
          <button
            className="reset-btn"
            onClick={handleReset}
            title="Start a new conversation"
            id="reset-session-btn"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M1 4v6h6" />
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
            </svg>
            New Chat
          </button>
        </div>
      </header>

      {/* Stage Progress element removed from UI */}

      {/* Messages Area */}
      <div className="messages-area" id="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h2>Welcome to ColorWhistle</h2>
            <p>
              Tell us about your project and we'll help you scope it out.
              <br />
              Type a message below to get started!
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            {msg.role === "assistant" && (
              <div className="avatar bot-avatar">
                <img src="/logo.svg" alt="Bot" style={{ width: '20px', height: '20px', objectFit: 'contain' }} />
              </div>
            )}
            <div className="message-bubble">
              <div className="message-content">
                {msg.content.split("\n").map((line, i) => (
                  <span key={i}>
                    {line}
                    {i < msg.content.split("\n").length - 1 && <br />}
                  </span>
                ))}
              </div>
              <div className="message-time">
                {msg.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>
            </div>
            {msg.role === "user" && (
              <div className="avatar user-avatar">
                <span>👤</span>
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {isLoading && (
          <div className="message assistant">
            <div className="avatar bot-avatar">
              <img src="/logo.svg" alt="Bot" style={{ width: '20px', height: '20px', objectFit: 'contain' }} />
            </div>
            <div className="message-bubble typing-indicator">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Collected Data Sidebar (mini) */}
      {Object.keys(collectedData).length > 0 && (
        <div className="data-panel">
          <details>
            <summary>
              📊 Collected Data ({Object.keys(collectedData).length} fields)
            </summary>
            <div className="data-items">
              {Object.entries(collectedData).map(([key, value]) => (
                <div key={key} className="data-item">
                  <span className="data-key">{key}:</span>
                  <span className="data-value">{value}</span>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}

      {/* Input Area */}
      <div className="input-area">
        {currentStage === "limit_warning" ? (
          <div className="limit-warning-options">
            <button
              className="option-btn"
              onClick={() => handleSend("Yes")}
              disabled={isLoading}
            >
              Yes
            </button>
            <button
              className="option-btn"
              onClick={() => handleSend("No")}
              disabled={isLoading}
            >
              No
            </button>
          </div>
        ) : (
          <>
            <div className="input-wrapper">
              <textarea
                ref={inputRef}
                id="chat-input"
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  currentStage === "completed"
                    ? "Session completed. Click 'New Chat' to start over."
                    : "Type your message..."
                }
                disabled={isLoading || currentStage === "completed"}
                rows={1}
              />
              <button
                className="send-btn"
                onClick={() => handleSend()}
                disabled={!input.trim() || isLoading || currentStage === "completed"}
                id="send-message-btn"
                title="Send message"
              >
                {isLoading ? (
                  <div className="send-spinner"></div>
                ) : (
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                  </svg>
                )}
              </button>
            </div>
            <div className="input-hint">
              Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
            </div>
          </>
        )}
      </div>
    </div>
  );
}
