import { useState, useRef, useEffect } from "react";
import { useAI } from "../hooks/useAI";
import { ModelSelector } from "../components/ModelSelector";

export default function AIPlayground() {
  const { messages, isGenerating, generate, clearMessages, selectedModel } = useAI();
  const [input, setInput] = useState("");
  const [maxTokens, setMaxTokens] = useState(1024);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;
    generate(input.trim());
    setInput("");
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="page-title" style={{ marginBottom: 0 }}>AI Playground</h1>
        <div className="flex items-center gap-3">
          <ModelSelector />
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted">Max tokens:</label>
            <select
              className="select"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            >
              <option value={256}>256</option>
              <option value={512}>512</option>
              <option value={1024}>1024</option>
              <option value={2048}>2048</option>
              <option value={4096}>4096</option>
            </select>
          </div>
          <button onClick={clearMessages} className="btn btn-ghost text-xs">
            Clear
          </button>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div className="chat-container">
          {/* Messages */}
          <div className="chat-messages">
            {messages.length === 0 && (
              <div style={{ textAlign: "center", padding: "60px 20px" }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>🤖</div>
                <div className="text-lg font-bold mb-2">Start a conversation</div>
                <div className="text-sm text-muted">
                  Send a message to generate AI responses using {selectedModel || "default"} model.
                </div>
              </div>
            )}
            {messages.map((msg) => (
              <div key={msg.id}>
                {msg.role === "user" ? (
                  <div className="chat-bubble chat-bubble-user">
                    {msg.content}
                  </div>
                ) : (
                  <div>
                    <div className="chat-meta">
                      AI
                      {msg.engine && ` · ${msg.engine}`}
                      {msg.model && ` · ${msg.model}`}
                      {msg.latency_ms != null && ` · ${msg.latency_ms}ms`}
                    </div>
                    <div className="chat-bubble chat-bubble-assistant">
                      <pre style={{
                        margin: 0,
                        padding: 0,
                        background: "transparent",
                        color: "inherit",
                        fontSize: "inherit",
                        maxHeight: "none",
                      }}>
                        {msg.content}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isGenerating && (
              <div className="chat-bubble chat-bubble-assistant animate-pulse">
                Generating...
              </div>
            )}
            <div ref={endRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="chat-input-bar">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Send a message..."
              disabled={isGenerating}
              className="input flex-1"
              style={{ background: "var(--color-input-bg)" }}
            />
            <button
              type="submit"
              disabled={isGenerating || !input.trim()}
              className="btn btn-primary"
            >
              {isGenerating ? "..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}