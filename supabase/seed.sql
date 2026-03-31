-- eco-base v1.0 — Seed Data
-- URI: eco-base://supabase/seed
-- Provides initial data for local development and testing

-- ─── Service Registry ──────────────────────────────────────────────────
INSERT INTO public.service_registry (service_name, service_endpoint, discovery_protocol, health_check_path, health_status, registry_ttl, k8s_namespace, uri, urn)
VALUES
  ('api-gateway',  'http://localhost:3000', 'consul', '/health', 'healthy', 30, 'eco-base',
   'eco-base://service/api-gateway', 'urn:eco-base:service:api-gateway'),
  ('ai-service',   'http://localhost:8001', 'consul', '/health', 'healthy', 30, 'eco-base',
   'eco-base://service/ai-service',  'urn:eco-base:service:ai-service'),
  ('root-gateway', 'http://localhost:8000', 'consul', '/health', 'healthy', 30, 'eco-base',
   'eco-base://service/root-gateway','urn:eco-base:service:root-gateway')
ON CONFLICT (service_name) DO NOTHING;

-- ─── Default Platforms ──────────────────────────────────────────────────
INSERT INTO public.platforms (name, slug, type, status, config, capabilities, k8s_namespace, deploy_target, urn)
VALUES
  ('Web Frontend',      'web',            'web',     'active', '{"port": 5173}',  ARRAY['ui','dashboard'],     'eco-base', 'vercel',
   'urn:eco-base:platform:web'),
  ('Desktop App',       'desktop',        'desktop', 'active', '{"port": 5174}',  ARRAY['electron','offline'], 'eco-base', 'electron-builder',
   'urn:eco-base:platform:desktop'),
  ('API Gateway',       'api',            'api',     'active', '{"port": 3000}',  ARRAY['rest','websocket'],   'eco-base', 'gke',
   'urn:eco-base:platform:api'),
  ('AI Service',        'ai',             'worker',  'active', '{"port": 8001}',  ARRAY['inference','embedding','governance'], 'eco-base', 'gke',
   'urn:eco-base:platform:ai'),
  ('IM Integration',    'im-integration', 'im',      'active', '{}',              ARRAY['whatsapp','telegram','line','messenger'], 'eco-base', 'cloudflare',
   'urn:eco-base:platform:im-integration'),
  ('Webhook Router',    'webhook-router', 'worker',  'active', '{}',              ARRAY['edge','rate-limiting','dedup'], 'eco-base', 'cloudflare',
   'urn:eco-base:platform:webhook-router')
ON CONFLICT (slug) DO NOTHING;

-- ─── Governance Records ──────────────────────────────────────────────────
INSERT INTO public.governance_records (action, resource_type, resource_id, details, compliance_tags, uri, urn)
VALUES
  ('seed',  'database', 'eco-base',  '{"description": "Initial seed data loaded"}',
   ARRAY['soc2','gitops'],
   'eco-base://governance/seed/initial',
   'urn:eco-base:governance:seed:initial')
ON CONFLICT DO NOTHING;
