/**
 * Widget Entry Point — Standalone Bootstrap
 *
 * This is the entry point for the embeddable widget build (IIFE).
 * When loaded via <script> tag on a host website, it:
 *
 *   1. Reads configuration from data-* attributes on the <script> tag
 *   2. Creates an isolated container div
 *   3. Renders the WidgetLauncher (FAB + chat panel) into it
 *
 * Usage on host website:
 *   <script
 *     src="https://chat.colorwhistle.com/widget.js"
 *     data-api-url="https://api.colorwhistle.com"
 *     data-company-name="ColorWhistle"
 *     data-logo-url="https://colorwhistle.com/logo.svg"
 *     data-position="bottom-right"
 *   ></script>
 */

import { createRoot } from "react-dom/client";
import { StrictMode } from "react";
import WidgetLauncher, { type WidgetConfig } from "./WidgetLauncher";

// Import widget styles (will be inlined into the IIFE bundle)
import "./widget.css";

// Import the ChatWindow styles too (needed inside the widget)
import "../components/ChatWindow.css";

/**
 * Find our own script tag and read data-* attributes for configuration.
 */
function getWidgetConfig(): WidgetConfig {
  // Find the script tag that loaded us
  const scripts = document.querySelectorAll("script[data-api-url]");
  const scriptTag = scripts[scripts.length - 1]; // last matching script

  const apiBaseUrl =
    scriptTag?.getAttribute("data-api-url") || "http://localhost:8000";
  const companyName =
    scriptTag?.getAttribute("data-company-name") || "ColorWhistle";
  const logoUrl =
    scriptTag?.getAttribute("data-logo-url") || undefined;
  const position =
    (scriptTag?.getAttribute("data-position") as "bottom-right" | "bottom-left") ||
    "bottom-right";
  const greeting =
    scriptTag?.getAttribute("data-greeting") || undefined;

  return {
    apiBaseUrl,
    companyName,
    logoUrl,
    position,
    greeting,
  };
}

/**
 * Initialize the widget — create container, parse config, render.
 */
function initWidget(): void {
  // Prevent double-initialization
  if (document.getElementById("cw-chat-widget-root")) {
    console.warn("[ColorWhistle Chat] Widget already initialized.");
    return;
  }

  const config = getWidgetConfig();

  // Create the widget container
  const container = document.createElement("div");
  container.id = "cw-chat-widget-root";
  container.className = `cw-chat-widget ${
    config.position === "bottom-left" ? "cw-position-left" : ""
  }`;

  // Ensure container doesn't interfere with host page layout
  container.style.cssText =
    "position:fixed;top:0;left:0;width:0;height:0;overflow:visible;z-index:2147483647;pointer-events:none;";

  document.body.appendChild(container);

  // All interactive children need pointer-events restored
  const style = document.createElement("style");
  style.textContent = `
    #cw-chat-widget-root .cw-fab,
    #cw-chat-widget-root .cw-panel {
      pointer-events: auto;
    }
  `;
  document.head.appendChild(style);

  // Render the widget
  const root = createRoot(container);
  root.render(
    <StrictMode>
      <WidgetLauncher config={config} />
    </StrictMode>
  );

  console.info(
    `[ColorWhistle Chat] Widget initialized — API: ${config.apiBaseUrl}`
  );
}

// Auto-initialize when the DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initWidget);
} else {
  initWidget();
}
