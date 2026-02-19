/**
 * IndestructibleEco LINE Messaging API Adapter
 *
 * Receives webhooks from LINE Platform, normalizes messages,
 * routes to backend AI service, and sends replies.
 *
 * URI: indestructibleeco://platforms/im-integration/line
 */

import express from "express";
import crypto from "crypto";

const PORT = parseInt(process.env.ECO_LINE_PORT || "3003", 10);
const CHANNEL_SECRET = process.env.ECO_LINE_CHANNEL_SECRET || "";
const CHANNEL_ACCESS_TOKEN = process.env.ECO_LINE_CHANNEL_ACCESS_TOKEN || "";
const API_URL = process.env.ECO_API_URL || "http://localhost:3000";
const LINE_API = "https://api.line.me/v2/bot/message/reply";

const app = express();
app.use(express.json());

function verifySignature(body: string, signature: string): boolean {
  const hash = crypto
    .createHmac("SHA256", CHANNEL_SECRET)
    .update(body)
    .digest("base64");
  return hash === signature;
}

interface NormalizedMessage {
  channel: "line";
  sender_id: string;
  reply_token: string;
  text: string;
  timestamp: string;
}

function extractMessages(body: any): NormalizedMessage[] {
  const messages: NormalizedMessage[] = [];
  for (const event of body?.events || []) {
    if (event.type !== "message" || event.message?.type !== "text") continue;
    messages.push({
      channel: "line",
      sender_id: event.source?.userId || "unknown",
      reply_token: event.replyToken,
      text: event.message.text,
      timestamp: new Date(event.timestamp).toISOString(),
    });
  }
  return messages;
}

async function sendReply(replyToken: string, text: string): Promise<void> {
  await fetch(LINE_API, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${CHANNEL_ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      replyToken,
      messages: [{ type: "text", text }],
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

app.post("/webhook/line", async (req, res) => {
  const signature = req.headers["x-line-signature"] as string;
  const rawBody = JSON.stringify(req.body);

  if (!verifySignature(rawBody, signature || "")) {
    res.sendStatus(401);
    return;
  }

  res.sendStatus(200);

  const messages = extractMessages(req.body);
  for (const msg of messages) {
    const reply = await routeToBackend(msg.text);
    await sendReply(msg.reply_token, reply);
  }
});

app.get("/health", (_req, res) =>
  res.json({ status: "healthy", channel: "line", api: API_URL })
);

app.listen(PORT, () => console.log(`LINE adapter listening on :${PORT}`));
