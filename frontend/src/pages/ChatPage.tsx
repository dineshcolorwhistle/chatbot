/**
 * ChatPage — Main Chat Page
 *
 * Wraps the ChatWindow component with page-level layout.
 * This is the entry point rendered in App.tsx.
 */

import ChatWindow from "../components/ChatWindow";

export default function ChatPage() {
  return (
    <main className="chat-page">
      <ChatWindow />
    </main>
  );
}
