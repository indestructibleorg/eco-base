/**
 * IndestructibleEco Cloudflare Worker — Webhook Router
 *
 * Routes incoming webhooks from IM platforms (WhatsApp, Telegram, LINE, Messenger)
 * to the API service. Validates signatures, normalizes payloads, and forwards.
 *
 * URI: indestructibleeco://backend/cloudflare/webhook-router
 */

interface Env {
  ECO_API_URL: string;
  ECO_WEBHOOK_SECRET: string;
  WHATSAPP_VERIFY_TOKEN: string;
  TELEGRAM_SECRET_TOKEN: string;
  LINE_CHANNEL_SECRET: string;
  MESSENGER_VERIFY_TOKEN: string;
  MESSENGER_APP_SECRET: string;
}

interface WebhookPayload {
  channel: string;
  raw: unknown;
  timestamp: string;
  signature_valid: boolean;
}

async function hmacSha256(secret: string, data: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(data));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function verifyTelegram(
  request: Request,
  env: Env,
  body: string
): Promise<boolean> {
  const token = request.headers.get("X-Telegram-Bot-Api-Secret-Token");
  return token === env.TELEGRAM_SECRET_TOKEN;
}

async function verifyWhatsApp(
  request: Request,
  env: Env,
  body: string
): Promise<boolean> {
  const signature = request.headers.get("X-Hub-Signature-256");
  if (!signature) return false;
  const expected = await hmacSha256(env.ECO_WEBHOOK_SECRET, body);
  return signature === `sha256=${expected}`;
}

async function verifyLine(
  request: Request,
  env: Env,
  body: string
): Promise<boolean> {
  const signature = request.headers.get("X-Line-Signature");
  if (!signature) return false;
  const expected = await hmacSha256(env.LINE_CHANNEL_SECRET, body);
  return signature === expected;
}

async function verifyMessenger(
  request: Request,
  env: Env,
  body: string
): Promise<boolean> {
  const signature = request.headers.get("X-Hub-Signature-256");
  if (!signature) return false;
  const expected = await hmacSha256(env.MESSENGER_APP_SECRET, body);
  return signature === `sha256=${expected}`;
}

function resolveChannel(pathname: string): string | null {
  if (pathname.startsWith("/webhook/whatsapp")) return "whatsapp";
  if (pathname.startsWith("/webhook/telegram")) return "telegram";
  if (pathname.startsWith("/webhook/line")) return "line";
  if (pathname.startsWith("/webhook/messenger")) return "messenger";
  return null;
}

type VerifyFn = (req: Request, env: Env, body: string) => Promise<boolean>;

const VERIFIERS: Record<string, VerifyFn> = {
  whatsapp: verifyWhatsApp,
  telegram: verifyTelegram,
  line: verifyLine,
  messenger: verifyMessenger,
};

async function handleVerificationChallenge(
  request: Request,
  channel: string,
  env: Env
): Promise<Response | null> {
  if (request.method !== "GET") return null;

  const url = new URL(request.url);

  if (channel === "whatsapp" || channel === "messenger") {
    const mode = url.searchParams.get("hub.mode");
    const token = url.searchParams.get("hub.verify_token");
    const challenge = url.searchParams.get("hub.challenge");
    const expected =
      channel === "whatsapp"
        ? env.WHATSAPP_VERIFY_TOKEN
        : env.MESSENGER_VERIFY_TOKEN;
    if (mode === "subscribe" && token === expected && challenge) {
      return new Response(challenge, { status: 200 });
    }
    return new Response("Forbidden", { status: 403 });
  }

  return null;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === "/health") {
      return new Response(
        JSON.stringify({
          status: "healthy",
          service: "webhook-router",
          version: "1.0.0",
          uri: "indestructibleeco://cloudflare/webhook-router/health",
          timestamp: new Date().toISOString(),
        }),
        { headers: { "Content-Type": "application/json" } }
      );
    }

    const channel = resolveChannel(url.pathname);
    if (!channel) {
      return new Response(
        JSON.stringify({ error: "not found", path: url.pathname }),
        { status: 404, headers: { "Content-Type": "application/json" } }
      );
    }

    // Handle GET verification challenges (WhatsApp, Messenger)
    const challengeResp = await handleVerificationChallenge(
      request,
      channel,
      env
    );
    if (challengeResp) return challengeResp;

    // POST: validate signature and forward
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const body = await request.text();
    const verifier = VERIFIERS[channel];
    const signatureValid = verifier ? await verifier(request, env, body) : false;

    if (!signatureValid) {
      return new Response(
        JSON.stringify({ error: "invalid signature", channel }),
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }

    // Forward to API service
    const payload: WebhookPayload = {
      channel,
      raw: JSON.parse(body),
      timestamp: new Date().toISOString(),
      signature_valid: true,
    };

    const apiUrl = env.ECO_API_URL || "https://api.indestructibleeco.com";
    try {
      const resp = await fetch(`${apiUrl}/api/v1/im/webhook`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Webhook-Channel": channel,
          "X-Webhook-Signature-Valid": "true",
        },
        body: JSON.stringify(payload),
      });

      const respBody = await resp.text();
      return new Response(respBody, {
        status: resp.status,
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      return new Response(
        JSON.stringify({
          error: "upstream_error",
          message: err instanceof Error ? err.message : "unknown",
          channel,
        }),
        { status: 502, headers: { "Content-Type": "application/json" } }
      );
    }
  },
};
