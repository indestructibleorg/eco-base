import { Link, useLocation, Outlet } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { useWebSocket } from "../hooks/useWebSocket";

const NAV_ITEMS = [
  { path: "/", label: "Dashboard", icon: "📊" },
  { path: "/ai", label: "AI Playground", icon: "🤖" },
  { path: "/models", label: "Models", icon: "🧠" },
  { path: "/platforms", label: "Platforms", icon: "🔌" },
  { path: "/settings", label: "Settings", icon: "⚙️" },
];

export function Layout() {
  const { user, logout } = useAuth();
  const location = useLocation();
  useWebSocket();

  return (
    <div className="flex">
      <aside className="sidebar">
        <div className="sidebar-brand">indestructibleeco</div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`sidebar-link${location.pathname === item.path ? " active" : ""}`}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
        <div style={{ borderTop: "1px solid var(--color-border)", paddingTop: 16 }}>
          <div className="text-sm text-muted">{user?.email}</div>
          <div className="text-xs text-muted" style={{ marginBottom: 8 }}>{user?.role}</div>
          <button onClick={logout} className="btn btn-ghost" style={{ width: "100%", fontSize: 12 }}>
            Logout
          </button>
        </div>
      </aside>
      <div className="page-content">
        <Outlet />
      </div>
    </div>
  );
}