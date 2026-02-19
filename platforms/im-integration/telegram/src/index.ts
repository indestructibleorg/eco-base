/**
 * IndestructibleEco Telegram Bot Adapter
 *
 * Supports both polling and webhook modes.
 * Normalizes messages via shared normalizer, routes via shared router.
 *
 * URI: indestructibleeco://platforms/im-integration/telegram
 */

import { Telegraf, Context } from "telegraf";
import type { Update, Message } from "telegraf/types";

const BOT_TOKEN = process.env.ECO_TELEGRAM_BOT_TOKEN || "";
const WEBHOOK_URL = process.env.ECO_TELEGRAM_WEBHOOK_URL || "";
const API_URL = process.env.ECO_API_URL || "http://localhost:3000";
const MODE = process.env.ECO_TELEGRAM_MODE || "polling"; // "polling" | "webhook"

if (!BOT_TOKEN) {
  console.error("ECO_TELEGRAM_BOT_TOKEN is required");
  process.exit(1);
}

const bot = new Telegraf(BOT_TOKEN);

interface NormalizedMessage {
  channel: "telegram";
  sender_id: string;
  chat_id: string;
  text: string;
  timestamp: string;
  raw_type: string;
}

function normalize(ctx: Context): NormalizedMessage | null {
  const msg = ctx.message;
  if (!msg || !("text" in msg)) return null;
  return {
    channel: "telegram",
    sender_id: String(msg.from?.id || "unknown"),
    chat_id: String(msg.chat.id),
    text: msg.text,
    timestamp: new Date(msg.date * 1000).toISOString(),
    raw_type: msg.chat.type,
  };
}

async function routeToBackend(msg: NormalizedMessage): Promise<string> {
  try {
    const resp = await fetch(`${API_URL}/api/v1/ai/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: msg.text,
        model_id: "default",
        max_tokens: 1024,
      }),
    });
    if (!resp.ok) return `Error: ${resp.status} ${resp.statusText}`;
    const data = (await resp.json()) as { content?: string };
    return data.content || "No response generated.";
  } catch (err) {
    console.error("Backend error:", err);
    return "Service temporarily unavailable.";
  }
}

bot.on("message", async (ctx) => {
  const msg = normalize(ctx);
  if (!msg) return;

  const reply = await routeToBackend(msg);
  await ctx.reply(reply);
});

bot.command("start", (ctx) =>
  ctx.reply("IndestructibleEco AI Bot ready. Send any message to begin.")
);

bot.command("health", (ctx) =>
  ctx.reply(
    JSON.stringify({
      status: "healthy",
      channel: "telegram",
      mode: MODE,
      api: API_URL,
    })
  )
);

if (MODE === "webhook" && WEBHOOK_URL) {
  bot.launch({ webhook: { domain: WEBHOOK_URL, port: 3001 } });
  console.log(`Telegram bot started in webhook mode: ${WEBHOOK_URL}`);
} else {
  bot.launch();
  console.log("Telegram bot started in polling mode");
}

process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
