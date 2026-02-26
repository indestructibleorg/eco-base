import { useState, useRef, useEffect } from "react";
import { useAI } from "../hooks/useAI";

export function AIChat() {
  const { messages, isGenerating, generate } = useAI();
  const [input, setInput] = useState("");
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
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id}>
            {msg.role === "user" ? (
              <div className="chat-bubble chat-bubble-user">{msg.content}</div>
            ) : (
              <div>
                <div className="chat-meta">
                  AI
                  {msg.engine && ` · ${msg.engine}`}
                  {msg.latency_ms != null && ` · ${msg.latency_ms}ms`}
                </div>
                <div className="chat-bubble chat-bubble-assistant">
                  <pre style={{
                    margin: 0, padding: 0, background: "transparent",
                    color: "inherit", fontSize: "inherit", maxHeight: "none",
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
      <form onSubmit={handleSubmit} className="chat-input-bar">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Send a message..."
          disabled={isGenerating}
          className="input flex-1"
        />
        <button
          type="submit"
          disabled={isGenerating || !input.trim()}
          className="btn btn-primary"
        >
          Send
        </button>
      </form>
    </div>
  );
}