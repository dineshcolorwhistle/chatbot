import { useState, type FormEvent, useEffect } from "react";
import { setPassword } from "../api";
import "./LoginPage.css";

export default function SetPasswordPage() {
  const [password, setPasswordValue] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const t = params.get("token");
    if (t) setToken(t);
  }, []);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!token) {
      setError("No token found in URL");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters long");
      return;
    }
    
    setError("");
    setLoading(true);
    
    try {
      await setPassword(token, password);
      setSuccess(true);
      setTimeout(() => {
        window.location.href = "/admin/login";
      }, 2000);
    } catch (err: any) {
      setError(err.message || "Failed to set password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-wrapper">
      <div className="login-card">
        <div className="login-card-header">
          <h1>Setup Security</h1>
          <p>Create a secure password for your admin account</p>
        </div>

        {success ? (
          <div className="login-success">
            Password set successfully! Redirecting to login...
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="login-form">
            {error && <div className="login-error">{error}</div>}
            
            <div className="login-input-group">
              <label htmlFor="new-password">New Password</label>
              <input 
                id="new-password"
                type="password" 
                value={password} 
                onChange={e => setPasswordValue(e.target.value)} 
                required 
                placeholder="••••••••"
              />
            </div>
            
            <div className="login-input-group">
              <label htmlFor="confirm-password">Confirm Password</label>
              <input 
                id="confirm-password"
                type="password" 
                value={confirmPassword} 
                onChange={e => setConfirmPassword(e.target.value)} 
                required 
                placeholder="••••••••"
              />
            </div>
            
            <button 
              type="submit" 
              className="login-btn"
              disabled={loading}
            >
              {loading ? "Saving..." : "Set Password"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
