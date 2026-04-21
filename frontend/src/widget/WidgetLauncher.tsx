/**
 * WidgetLauncher — Embeddable Chat Widget Root
 *
 * Renders:
 *   1. A floating action button (FAB) in the bottom corner
 *   2. An expandable chat panel when the FAB is clicked
 *
 * This is the top-level component for the widget build.
 * It wraps the existing ChatWindow with widget-specific
 * chrome (header, close button, FAB).
 */

import { useState, useCallback, useRef } from "react";
import ChatWindow from "../components/ChatWindow";
import WidgetHeader from "./WidgetHeader";

export interface WidgetConfig {
  /** Backend API base URL. */
  apiBaseUrl: string;
  /** Company name shown in the widget header. */
  companyName: string;
  /** URL to the company logo (absolute or relative). */
  logoUrl?: string;
  /** Widget position on screen. */
  position?: "bottom-right" | "bottom-left";
  /** Initial greeting message override. */
  greeting?: string;
}

interface WidgetLauncherProps {
  config: WidgetConfig;
}

export default function WidgetLauncher({ config }: WidgetLauncherProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const chatResetRef = useRef<(() => void) | null>(null);

  const handleOpen = useCallback(() => {
    setIsOpen(true);
    setIsClosing(false);
  }, []);

  const handleClose = useCallback(() => {
    setIsClosing(true);
    // Wait for exit animation to complete
    setTimeout(() => {
      setIsOpen(false);
      setIsClosing(false);
    }, 250);
  }, []);

  const handleToggle = useCallback(() => {
    if (isOpen) {
      handleClose();
    } else {
      handleOpen();
    }
  }, [isOpen, handleClose, handleOpen]);

  const handleReset = useCallback(() => {
    if (chatResetRef.current) {
      chatResetRef.current();
    }
  }, []);

  return (
    <>
      {/* Chat Panel */}
      {isOpen && (
        <div className={`cw-panel ${isClosing ? "cw-panel--closing" : ""}`}>
          {/* Widget Header */}
          <WidgetHeader
            companyName={config.companyName}
            logoUrl={config.logoUrl}
            onClose={handleClose}
            onReset={handleReset}
          />

          {/* Chat Body */}
          <div className="cw-widget-body">
            <ChatWindow
              mode="widget"
              apiBaseUrl={config.apiBaseUrl}
              onResetRef={chatResetRef}
            />
          </div>

          {/* Powered By Footer */}
          <div className="cw-powered-by">
            <a
              href="https://colorwhistle.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              ⚡ Powered by {config.companyName}
            </a>
          </div>
        </div>
      )}

      {/* Floating Action Button */}
      <button
        className={`cw-fab ${isOpen ? "cw-fab--open" : ""}`}
        onClick={handleToggle}
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {/* Chat icon */}
        <svg
          className="cw-fab-icon"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z" />
          <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z" />
        </svg>

        {/* Close icon */}
        <svg
          className="cw-fab-close-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </>
  );
}
