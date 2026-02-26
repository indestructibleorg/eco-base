import { useState } from "react";
import { useAuth } from "../hooks/useAuth";

export default function Settings() {
  const { user } = useAuth();
  const [apiKeyVisible, setApiKeyVisible] = useState(false);
  const [copied, setCopied] = useState(false);

  const token = localStorage.getItem("eco-auth")
    ? JSON.parse(localStorage.getItem("eco-auth") || "{}").state?.token || ""
    : "";

  function copyToken() {
    if (token) {
      navigator.clipboard.writeText(token).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  }

  return (
    <div>
      <h1 className="page-title">Settings</h1>

      {/* Profile */}
      <div className="card mb-4">
        <div className="text-xs text-muted mb-3">Profile</div>
        <div className="grid grid-2 gap-4">
          <div>
            <label className="text-xs text-muted" style={{ display: "block", marginBottom: 4 }}>Email</label>
            <div className="input" style={{ background: "var(--color-surface-2)", cursor: "default" }}>
              {user?.email || "—"}
            </div>
          </div>
          <div>
            <label className="text-xs text-muted" style={{ display: "block", marginBottom: 4 }}>Role</label>
            <div className="input" style={{ background: "var(--color-surface-2)", cursor: "default" }}>
              {user?.role || "—"}
            </div>
          </div>
          <div>
            <label className="text-xs text-muted" style={{ display: "block", marginBottom: 4 }}>User ID</label>
            <div className="input font-mono text-xs" style={{ background: "var(--color-surface-2)", cursor: "default" }}>
              {user?.id || "—"}
            </div>
          </div>
          <div>
            <label className="text-xs text-muted" style={{ display: "block", marginBottom: 4 }}>Name</label>
            <div className="input" style={{ background: "var(--color-surface-2)", cursor: "default" }}>
              {user?.name || "—"}
            </div>
          </div>
        </div>
      </div>

      {/* API Token */}
      <div className="card mb-4">
        <div className="text-xs text-muted mb-3">API Access Token</div>
        <div className="flex gap-3 items-center">
          <div
            className="input font-mono text-xs flex-1"
            style={{
              background: "var(--color-surface-2)",
              cursor: "default",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {apiKeyVisible ? (token || "No token") : "••••••••••••••••••••••••••••••••"}
          </div>
          <button
            className="btn btn-ghost text-xs"
            onClick={() => setApiKeyVisible(!apiKeyVisible)}
          >
            {apiKeyVisible ? "Hide" : "Show"}
          </button>
          <button
            className="btn btn-ghost text-xs"
            onClick={copyToken}
            disabled={!token}
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <div className="text-xs text-muted mt-4">
          Use this token in the Authorization header: <code>Bearer {"<token>"}</code>
        </div>
      </div>

      {/* Environment */}
      <div className="card mb-4">
        <div className="text-xs text-muted mb-3">Environment</div>
        <div className="flex flex-col gap-2">
          <div className="table-row">
            <span className="text-sm">API URL</span>
            <span className="font-mono text-xs text-muted">
              {import.meta.env.VITE_API_URL || "http://localhost:3000"}
            </span>
          </div>
          <div className="table-row">
            <span className="text-sm">WebSocket URL</span>
            <span className="font-mono text-xs text-muted">
              {import.meta.env.VITE_WS_URL || "ws://localhost:3000/ws"}
            </span>
          </div>
          <div className="table-row">
            <span className="text-sm">Platform</span>
            <span className="font-mono text-xs text-muted">indestructibleeco v1.0</span>
          </div>
          <div className="table-row">
            <span className="text-sm">Frontend</span>
            <span className="font-mono text-xs text-muted">React + Vite + Zustand</span>
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="card" style={{ borderColor: "rgba(247,37,133,0.3)" }}>
        <div className="text-xs text-danger mb-3">Danger Zone</div>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-bold">Clear Local Data</div>
            <div className="text-xs text-muted">
              Remove all cached data, tokens, and preferences from this browser.
            </div>
          </div>
          <button
            className="btn btn-danger text-xs"
            onClick={() => {
              localStorage.clear();
              sessionStorage.clear();
              window.location.href = "/login";
            }}
          >
            Clear All Data
          </button>
        </div>
      </div>
    </div>
  );
}