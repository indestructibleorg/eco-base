/**
 * IndestructibleEco Facebook Messenger Adapter
 *
 * Receives webhooks from Facebook Graph API, normalizes messages,
 * routes to backend AI service, and sends replies.
 *
 * URI: indestructibleeco://platforms/im-integration/messenger
 */

import express from "express";
import crypto from "crypto";

const PORT = parseInt(process.env.ECO_MESSENGER_PORT || "3004", 10);
const VERIFY_TOKEN = process.env.ECO_MESSENGER_VERIFY_TOKEN || "";
const PAGE_ACCESS_TOKEN = process.env.ECO_MESSENGER_PAGE_ACCESS_TOKEN || "";
const APP_SECRET = process.env.ECO_MESSENGER_APP_SECRET || "";
const API_URL = process.env.ECO_API_URL || "http://localhost:3000";
const GRAPH_API = "https://graph.facebook.com/v18.0/me/messages";

const app = express();
app.use(express.json());

function verifySignature(body: string, signature: string): boolean {
  const expected = crypto
    .createHmac("sha256", APP_SECRET)
    .update(body)
    .digest("hex");
  return signature === `sha256=${expected}`;
}

interface NormalizedMessage {
  channel: "messenger";
  sender_id: string;
  text: string;
  timestamp: string;
  message_id: string;
}

function extractMessages(body: any): NormalizedMessage[] {
  const messages: NormalizedMessage[] = [];
  for (const entry of body?.entry || []) {
    for (const event of entry?.messaging || []) {
      if (!event.message?.text) continue;
      messages.push({
        channel: "messenger",
        sender_id: event.sender?.id || "unknown",
        text: event.message.text,
        timestamp: new Date(event.timestamp).toISOString(),
        message_id: event.message.mid,
      });
    }
  }
  return messages;
}

async function sendReply(recipientId: string, text: string): Promise<void> {
  await fetch(`${GRAPH_API}?access_token=${PAGE_ACCESS_TOKEN}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      recipient: { id: recipientId },
      message: { text },
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
app.get("/webhook/messenger", (req, res) => {
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
app.post("/webhook/messenger", async (req, res) => {
  const signature = req.headers["x-hub-signature-256"] as string;
  const rawBody = JSON.stringify(req.body);

  if (APP_SECRET && !verifySignature(rawBody, signature || "")) {
    res.sendStatus(401);
    return;
  }

  res.sendStatus(200);

  const messages = extractMessages(req.body);
  for (const msg of messages) {
    const reply = await routeToBackend(msg.text);
    await sendReply(msg.sender_id, reply);
  }
});

app.get("/health", (_req, res) =>
  res.json({ status: "healthy", channel: "messenger", api: API_URL })
);

app.listen(PORT, () => console.log(`Messenger adapter listening on :${PORT}`));
