/**
 * App Root Component
 *
 * Entry point that renders the ChatPage.
 * Designed for easy future expansion (routing, providers, etc.).
 */

import { useState, useEffect } from "react";
import ChatPage from "./pages/ChatPage";
import AdminPage from "./pages/AdminPage";
import LoginPage from "./pages/LoginPage";
import SetPasswordPage from "./pages/SetPasswordPage";

function App() {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  if (currentPath === "/admin") {
    return <AdminPage />;
  }
  
  if (currentPath === "/admin/login") {
    return <LoginPage />;
  }
  
  if (currentPath.startsWith("/set-password")) {
    return <SetPasswordPage />;
  }

  return <ChatPage />;
}

export default App;
