# IndestructibleEco - GKE Deployment & Security Remediation

## Current Status
- **Repository**: indestructibleorg/indestructibleeco
- **Latest Commit**: 12fc544 (Supabase config and edge function)
- **Staging Cluster**: eco-staging (6/6 pods Running ✅)
- **Production Cluster**: Deleted (needs recreation after SSD quota increase)

---

## Phase 1: GCP Infrastructure (BLOCKED - Needs Browser)

### SSD Quota Increase
- [ ] Navigate to GCP Console IAM & Admin → Quotas
- [ ] Increase SSD_TOTAL_GB in asia-east1 from 250 to 500GB
- [ ] Verify quota increase applied
- **Blocker**: Org policy caps consumer override at 250, requires browser-based quota request

### OAuth Consent Screen & Credentials
- [ ] Configure OAuth consent screen (External user type)
- [ ] Create OAuth 2.0 Client ID (Web application)
- [ ] Add redirect URIs: `https://staging.autoecoops.io/auth/callback`, `https://production.autoecoops.io/auth/callback`
- [ ] Share Client ID and Client Secret
- **Blocker**: IAP brands API requires org membership, needs GCP Console UI

---

## Phase 2: GKE Cluster Operations

### Staging Cluster
- [x] Verify eco-staging cluster health and pod status
- [x] Fix web-frontend CrashLoopBackOff (nginx upstream + securityContext)
- [x] All 6 pods Running (gateway, ai, api, web, postgres, redis)
- [x] Endpoints live: https://staging.autoecoops.io/ and https://api-staging.autoecoops.io/

### Production Cluster
- [ ] Recreate eco-production cluster per docs/gke-operations.md
- [ ] Deploy production workloads (9 manifests)
- [ ] Verify production endpoints

---

## Phase 3: Supabase Configuration

- [x] Update K8s eco-secrets with real Supabase URL
- [x] Update K8s eco-secrets with real Supabase anon key
- [x] Update GitHub secrets with new Supabase values
- [x] Update ConfigMaps (staging + production)
- [ ] Deploy Supabase Edge Function (hello-world) - needs personal access token
- [ ] Verify Supabase integration end-to-end

---

## Phase 4: Security Remediation

### Completed Fixes
- [x] Fix python-multipart vulnerability (>=0.0.18)
- [x] Add USER to backend/ai/Dockerfile (uid 1001)
- [x] Add USER to backend/api/Dockerfile (uid 1001)
- [x] Add security comment to platforms/web/Dockerfile (nginx requires root)

### Pending Fixes
- [ ] Fix Docker socket exposure (2 Critical findings)
- [ ] Fix privileged service (2 Critical findings)
- [ ] Fix path traversal in JavaScript (21 Critical/High findings)
- [ ] Fix Django URL host injection (1 Critical finding)

---

## Phase 5: ConfigMap Parity

### Keys Missing in Production (23 keys)
- [x] Add AI/engine config keys to production configmap
- [x] Add service URL keys to production configmap
- [x] Add runtime config keys to production configmap

### Keys Missing in Staging (13 keys)
- [x] Add security keys to staging configmap
- [x] Add observability keys to staging configmap
- [x] Add auth keys to staging configmap

### Additional Fixes
- [x] Create production Secret manifest (was missing)
- [x] Add governance block to production configmap
- [x] Fix test assertion for production configmap
- [x] All 53 tests passing

---

## Phase 6: Repo Updates & Integration

- [ ] Commit and push all pending changes
- [ ] Verify CI passes (623 tests)
- [ ] Verify deploy workflow succeeds
- [ ] Update documentation

---

## Test Status
- **Unit tests**: 623 passing ✅
- **K8s tests**: 53 passing ✅
- **QYAML files**: 31 parse correctly ✅
- **Workflow files**: 14 parse correctly ✅