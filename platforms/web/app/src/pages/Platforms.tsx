import { useState, useEffect } from "react";
import { api } from "../lib/api";
import { usePlatformStore } from "../store/platform";

export default function Platforms() {
  const { platforms, loading, setPlatforms, setLoading, addPlatform, removePlatform } = usePlatformStore();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newSlug, setNewSlug] = useState("");
  const [newType, setNewType] = useState("custom");
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    api.platforms()
      .then((data: any) => {
        setPlatforms(data?.platforms || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [setPlatforms, setLoading]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim() || !newSlug.trim()) return;
    setError("");

    try {
      const resp = await api.post<{ platform: any }>("/api/v1/platforms", {
        name: newName,
        slug: newSlug,
        type: newType,
      });
      addPlatform(resp.platform);
      setShowCreate(false);
      setNewName("");
      setNewSlug("");
    } catch (err: any) {
      setError(err.message || "Failed to create platform");
    }
  }

  async function handleDelete(id: string) {
    try {
      await api.delete(`/api/v1/platforms/${id}`);
      removePlatform(id);
    } catch {
      // Ignore
    }
  }

  const statusBadge = (s: string) =>
    s === "active" ? "badge-success"
    : s === "deploying" ? "badge-warning"
    : s === "inactive" ? "badge-danger"
    : "badge-info";

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="page-title" style={{ marginBottom: 0 }}>
          Platforms ({platforms.length})
        </h1>
        <button
          className="btn btn-primary"
          onClick={() => setShowCreate(!showCreate)}
        >
          {showCreate ? "Cancel" : "+ New Platform"}
        </button>
      </div>

      {/* Create Form */}
      {showCreate && (
        <div className="card mb-4">
          <form onSubmit={handleCreate} className="flex flex-col gap-3">
            <div className="grid grid-3 gap-3">
              <div>
                <label className="text-xs text-muted mb-2" style={{ display: "block" }}>Name</label>
                <input
                  className="input"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="My Platform"
                  required
                />
              </div>
              <div>
                <label className="text-xs text-muted mb-2" style={{ display: "block" }}>Slug</label>
                <input
                  className="input"
                  value={newSlug}
                  onChange={(e) => setNewSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-"))}
                  placeholder="my-platform"
                  required
                />
              </div>
              <div>
                <label className="text-xs text-muted mb-2" style={{ display: "block" }}>Type</label>
                <select
                  className="select w-full"
                  value={newType}
                  onChange={(e) => setNewType(e.target.value)}
                >
                  <option value="custom">Custom</option>
                  <option value="web">Web</option>
                  <option value="api">API</option>
                  <option value="bot">Bot</option>
                  <option value="extension">Extension</option>
                </select>
              </div>
            </div>
            {error && <div className="text-sm text-danger">{error}</div>}
            <div>
              <button type="submit" className="btn btn-primary">Create Platform</button>
            </div>
          </form>
        </div>
      )}

      {/* Platform List */}
      {loading ? (
        <div className="card">
          <div className="text-sm text-muted animate-pulse">Loading platforms...</div>
        </div>
      ) : platforms.length === 0 ? (
        <div className="card">
          <div style={{ textAlign: "center", padding: "40px 20px" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🔌</div>
            <div className="text-md font-bold mb-2">No platforms registered</div>
            <div className="text-sm text-muted">Create your first platform to get started.</div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {platforms.map((p) => (
            <div key={p.id} className="card">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-md font-bold">{p.name}</span>
                  <span className="font-mono text-xs text-muted">{p.slug}</span>
                  <span className="badge badge-info">{p.type}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`badge ${statusBadge(p.status)}`}>{p.status}</span>
                  <button
                    className="btn btn-ghost text-xs"
                    onClick={() => handleDelete(p.id)}
                    style={{ padding: "4px 10px" }}
                  >
                    Delete
                  </button>
                </div>
              </div>
              {p.capabilities && p.capabilities.length > 0 && (
                <div className="flex gap-2 mt-4" style={{ flexWrap: "wrap" }}>
                  {p.capabilities.map((c) => (
                    <span key={c} className="badge badge-success">{c}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}