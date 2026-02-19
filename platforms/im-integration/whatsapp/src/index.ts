/**
 * IndestructibleEco WhatsApp Cloud API Adapter
 *
 * Receives webhooks from Meta Cloud API, normalizes messages,
 * routes to backend AI service, and sends replies.
 *
 * URI: indestructibleeco://platforms/im-integration/whatsapp
 */

import express from "express";

const PORT = parseInt(process.env.ECO_WHATSAPP_PORT || "3002", 10);
const VERIFY_TOKEN = process.env.ECO_WHATSAPP_VERIFY_TOKEN || "";
const ACCESS_TOKEN = process.env.ECO_WHATSAPP_ACCESS_TOKEN || "";
const PHONE_NUMBER_ID = process.env.ECO_WHATSAPP_PHONE_NUMBER_ID || "";
const API_URL = process.env.ECO_API_URL || "http://localhost:3000";
const GRAPH_API = "https://graph.facebook.com/v18.0";

const app = express();
app.use(express.json());

interface NormalizedMessage {
  channel: "whatsapp";
  sender_id: string;
  chat_id: string;
  text: string;
  timestamp: string;
  message_id: string;
}

function extractMessages(body: any): NormalizedMessage[] {
  const messages: NormalizedMessage[] = [];
  const entries = body?.entry || [];
  for (const entry of entries) {
    const changes = entry?.changes || [];
    for (const change of changes) {
      const value = change?.value;
      if (!value?.messages) continue;
      for (const msg of value.messages) {
        if (msg.type !== "text") continue;
        messages.push({
          channel: "whatsapp",
          sender_id: msg.from,
          chat_id: msg.from,
          text: msg.text?.body || "",
          timestamp: new Date(parseInt(msg.timestamp, 10) * 1000).toISOString(),
          message_id: msg.id,
        });
      }
    }
  }
  return messages;
}

async function sendReply(to: string, text: string): Promise<void> {
  await fetch(`${GRAPH_API}/${PHONE_NUMBER_ID}/messages`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      to,
      type: "text",
      text: { body: text },
    }),
  });
}

async function routeToBackend(text: string): Promise<string> {
  try {
    const resp = await fetch(`${API_URL}/api/v1/ai/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, model_id: "default", max_tokens: 1024 }),
    });
    if (!resp.ok) return `Error: ${resp.status}`;
    const data = (await resp.json()) as { content?: string };
    return data.content || "No response generated.";
  } catch {
    return "Service temporarily unavailable.";
  }
}

// Webhook verification
app.get("/webhook/whatsapp", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];
  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    res.status(200).send(challenge);
  } else {
    res.sendStatus(403);
  }
});

// Webhook handler
app.post("/webhook/whatsapp", async (req, res) => {
  const messages = extractMessages(req.body);
  res.sendStatus(200); // Acknowledge immediately

  for (const msg of messages) {
    const reply = await routeToBackend(msg.text);
    await sendReply(msg.sender_id, reply);
  }
});

app.get("/health", (_req, res) =>
  res.json({ status: "healthy", channel: "whatsapp", api: API_URL })
);

app.listen(PORT, () => console.log(`WhatsApp adapter listening on :${PORT}`));
