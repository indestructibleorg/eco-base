import { useState, useEffect } from "react";
import { api } from "../lib/api";

interface ModelInfo {
  id: string;
  name?: string;
  provider?: string;
  status: string;
  capabilities?: string[];
  compatible_engines?: string[];
  context_length?: number;
  uri?: string;
  urn?: string;
}

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api.models()
      .then((data: any) => {
        if (Array.isArray(data)) setModels(data);
        else if (data?.models) setModels(data.models);
        else if (data?.data) setModels(data.data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const filtered = models.filter(
    (m) =>
      m.id.toLowerCase().includes(filter.toLowerCase()) ||
      (m.provider || "").toLowerCase().includes(filter.toLowerCase())
  );

  const badgeClass = (s: string) =>
    s === "available" || s === "ready" ? "badge-success"
    : s === "loading" ? "badge-warning"
    : "badge-danger";

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="page-title" style={{ marginBottom: 0 }}>
          Models ({models.length})
        </h1>
        <input
          className="input"
          style={{ width: 280 }}
          placeholder="Filter models..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      {loading ? (
        <div className="card">
          <div className="text-sm text-muted animate-pulse">Loading models...</div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="card">
          <div className="text-sm text-muted">
            {filter ? "No models match your filter." : "No models registered."}
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {filtered.map((m) => (
            <div
              key={m.id}
              className="card"
              style={{ cursor: "pointer" }}
              onClick={() => setExpandedId(expandedId === m.id ? null : m.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-md font-bold">{m.id}</span>
                  {m.provider && (
                    <span className="badge badge-info">{m.provider}</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {m.context_length && (
                    <span className="text-xs text-muted">
                      {(m.context_length / 1024).toFixed(0)}K ctx
                    </span>
                  )}
                  <span className={`badge ${badgeClass(m.status)}`}>{m.status}</span>
                </div>
              </div>

              {expandedId === m.id && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--color-border)" }}>
                  <div className="grid grid-2 gap-4">
                    {m.capabilities && m.capabilities.length > 0 && (
                      <div>
                        <div className="text-xs text-muted mb-2">Capabilities</div>
                        <div className="flex gap-2" style={{ flexWrap: "wrap" }}>
                          {m.capabilities.map((c) => (
                            <span key={c} className="badge badge-info">{c}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {m.compatible_engines && m.compatible_engines.length > 0 && (
                      <div>
                        <div className="text-xs text-muted mb-2">Compatible Engines</div>
                        <div className="flex gap-2" style={{ flexWrap: "wrap" }}>
                          {m.compatible_engines.map((e) => (
                            <span key={e} className="badge badge-success">{e}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {m.uri && (
                      <div>
                        <div className="text-xs text-muted mb-2">URI</div>
                        <code className="text-xs font-mono">{m.uri}</code>
                      </div>
                    )}
                    {m.urn && (
                      <div>
                        <div className="text-xs text-muted mb-2">URN</div>
                        <code className="text-xs font-mono">{m.urn}</code>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}