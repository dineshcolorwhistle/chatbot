/**
 * WidgetHeader — Compact Chat Widget Header
 *
 * Renders a slim header for the widget panel with:
 *   - Brand logo + name + online status
 *   - New Chat (reset) button
 *   - Close button
 *
 * This replaces the full-page chat-header when running
 * in widget mode.
 */

interface WidgetHeaderProps {
  /** Company name to display. */
  companyName: string;
  /** URL to the company logo image. */
  logoUrl?: string;
  /** Called when the close button is clicked. */
  onClose: () => void;
  /** Called when the new-chat/reset button is clicked. */
  onReset: () => void;
}

export default function WidgetHeader({
  companyName,
  logoUrl,
  onClose,
  onReset,
}: WidgetHeaderProps) {
  return (
    <div className="cw-widget-header">
      <div className="cw-widget-header-left">
        {logoUrl && (
          <div className="cw-widget-header-logo">
            <img src={logoUrl} alt={`${companyName} logo`} />
          </div>
        )}
        <div className="cw-widget-header-info">
          <h2>{companyName}</h2>
          <div className="cw-widget-header-status">
            <span className="cw-status-dot" />
            <span style={{ fontSize: '11px', color: 'var(--cw-text-tertiary)' }}>
              Online
            </span>
          </div>
        </div>
      </div>

      <div className="cw-widget-header-actions">
        {/* New Chat / Reset */}
        <button
          className="cw-widget-header-btn"
          onClick={onReset}
          title="Start a new conversation"
          aria-label="Start a new conversation"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M1 4v6h6" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
        </button>

        {/* Close / Minimize */}
        <button
          className="cw-widget-header-btn"
          onClick={onClose}
          title="Close chat"
          aria-label="Close chat"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  );
}
