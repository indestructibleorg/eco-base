#!/usr/bin/env bash
# drill-flagger-rollback.sh
# Weekly drill: Flagger canary rollback verification via failure injection
# Artifacts: tests/reports/drill-flagger-rollback-<timestamp>.json
set -euo pipefail

NAMESPACE="platform-01"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_FILE="tests/reports/drill-flagger-rollback-${TIMESTAMP}.json"
mkdir -p tests/reports

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log "=== Flagger Rollback Drill ==="
log "Namespace: $NAMESPACE"
log "Start: $START_TIME"

# ── Cleanup function ──────────────────────────────────────────────────────
cleanup() {
  log "[Cleanup] Removing test resources..."
  kubectl delete canary podinfo-drill -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  kubectl delete deployment podinfo-drill -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  kubectl delete service podinfo-drill -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  kubectl delete deployment flagger-loadtester-drill -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  kubectl delete service flagger-loadtester-drill -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  log "Cleanup complete."
}
trap cleanup EXIT

# ── Step 1: Deploy test workload ──────────────────────────────────────────
log "[Step 1] Deploying test workload..."
kubectl apply -n "$NAMESPACE" -f - <<'EOF'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: podinfo-drill
  namespace: platform-01
  labels:
    app: podinfo-drill
spec:
  replicas: 2
  selector:
    matchLabels:
      app: podinfo-drill
  template:
    metadata:
      labels:
        app: podinfo-drill
    spec:
      containers:
        - name: podinfo
          image: stefanprodan/podinfo:6.7.0
          ports:
            - containerPort: 9898
          resources:
            requests:
              cpu: 10m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 128Mi
---
apiVersion: v1
kind: Service
metadata:
  name: podinfo-drill
  namespace: platform-01
spec:
  selector:
    app: podinfo-drill
  ports:
    - port: 9898
      targetPort: 9898
EOF

kubectl rollout status deployment/podinfo-drill -n "$NAMESPACE" --timeout=120s

# ── Step 2: Configure Flagger Canary ─────────────────────────────────────
log "[Step 2] Configuring Flagger Canary..."
kubectl apply -n "$NAMESPACE" -f - <<'EOF'
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: podinfo-drill
  namespace: platform-01
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: podinfo-drill
  service:
    port: 9898
  analysis:
    interval: 10s
    threshold: 3
    maxWeight: 50
    stepWeight: 10
    metrics:
      - name: request-success-rate
        thresholdRange:
          min: 99
        interval: 30s
      - name: request-duration
        thresholdRange:
          max: 200
        interval: 30s
EOF

sleep 15
INIT_PHASE=$(kubectl get canary podinfo-drill -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
log "Canary initialized: phase=$INIT_PHASE"

# ── Step 3: Deploy load tester ────────────────────────────────────────────
log "[Step 3] Deploying load tester..."
kubectl apply -n "$NAMESPACE" -f - <<'EOF'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flagger-loadtester-drill
  namespace: platform-01
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flagger-loadtester-drill
  template:
    metadata:
      labels:
        app: flagger-loadtester-drill
    spec:
      containers:
        - name: loadtester
          image: ghcr.io/fluxcd/flagger-loadtester:0.34.0
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 10m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 128Mi
---
apiVersion: v1
kind: Service
metadata:
  name: flagger-loadtester-drill
  namespace: platform-01
spec:
  selector:
    app: flagger-loadtester-drill
  ports:
    - port: 80
      targetPort: 8080
EOF

kubectl rollout status deployment/flagger-loadtester-drill -n "$NAMESPACE" --timeout=120s

# ── Step 4: Inject failure ────────────────────────────────────────────────
log "[Step 4] Injecting failure (high-error-rate image)..."
kubectl set image deployment/podinfo-drill podinfo=stefanprodan/podinfo:fake -n "$NAMESPACE" 2>/dev/null || \
  kubectl patch deployment podinfo-drill -n "$NAMESPACE" \
    -p '{"spec":{"template":{"spec":{"containers":[{"name":"podinfo","image":"stefanprodan/podinfo:fake"}]}}}}'

sleep 20

# ── Step 5: Monitor rollback ──────────────────────────────────────────────
log "[Step 5] Monitoring canary rollback (max 3 minutes)..."
ROLLBACK_DETECTED=false
FINAL_PHASE="Unknown"
FINAL_WEIGHT="0"
FAILURE_REASON=""
MONITOR_START=$(date +%s)

for i in $(seq 1 18); do
  sleep 10
  PHASE=$(kubectl get canary podinfo-drill -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
  WEIGHT=$(kubectl get canary podinfo-drill -n "$NAMESPACE" -o jsonpath='{.status.canaryWeight}' 2>/dev/null || echo "0")
  ITERATIONS=$(kubectl get canary podinfo-drill -n "$NAMESPACE" -o jsonpath='{.status.iterations}' 2>/dev/null || echo "0")
  log "  [$(date -u +%H:%M:%S)] phase=$PHASE weight=$WEIGHT iterations=$ITERATIONS"

  if [[ "$PHASE" == "Failed" ]]; then
    ROLLBACK_DETECTED=true
    FINAL_PHASE="$PHASE"
    FINAL_WEIGHT="$WEIGHT"
    FAILURE_REASON=$(kubectl get canary podinfo-drill -n "$NAMESPACE" \
      -o jsonpath='{.status.conditions[?(@.type=="Promoted")].message}' 2>/dev/null || echo "Canary analysis failed")
    break
  fi
  if [[ "$PHASE" == "Succeeded" ]]; then
    FINAL_PHASE="$PHASE"
    break
  fi
done

MONITOR_END=$(date +%s)
ROLLBACK_DURATION=$((MONITOR_END - MONITOR_START))

# ── Step 6: Determine gate result ─────────────────────────────────────────
log "[Step 6] Verifying rollback state..."
log "  Final phase: $FINAL_PHASE"
log "  Final canary weight: $FINAL_WEIGHT"
log "  Rollback duration: ${ROLLBACK_DURATION}s"

if [[ "$ROLLBACK_DETECTED" == "true" ]] && [[ "$FINAL_WEIGHT" == "0" ]]; then
  GATE_RESULT="PASS"
  GATE_MESSAGE="Flagger detected failures and rolled back canary in ${ROLLBACK_DURATION}s"
else
  GATE_RESULT="FAIL"
  GATE_MESSAGE="Flagger did not roll back within 3 minutes (phase=$FINAL_PHASE, weight=$FINAL_WEIGHT)"
fi

log ""
log "=== Drill Result: $GATE_RESULT ==="
log "Report: $REPORT_FILE"

END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ── Write JSON report ─────────────────────────────────────────────────────
python3 - <<PYEOF
import json

report = {
    "drill": "Flagger Rollback",
    "timestamp": "$END_TIME",
    "namespace": "$NAMESPACE",
    "gate_result": "$GATE_RESULT",
    "gate_message": "$GATE_MESSAGE",
    "canary": {
        "final_phase": "$FINAL_PHASE",
        "final_weight": "$FINAL_WEIGHT",
        "failure_reason": "$FAILURE_REASON",
        "rollback_detected": $( [[ "$ROLLBACK_DETECTED" == "true" ]] && echo "true" || echo "false" ),
        "rollback_duration_s": $ROLLBACK_DURATION
    },
    "sli_thresholds": {
        "error_rate_min_pct": 99,
        "p99_latency_max_ms": 200,
        "analysis_interval_s": 10,
        "failure_threshold": 3
    },
    "timing": {
        "start": "$START_TIME",
        "end": "$END_TIME",
        "duration_s": $ROLLBACK_DURATION
    }
}

with open("$REPORT_FILE", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
PYEOF

if [[ "$GATE_RESULT" != "PASS" ]]; then
  exit 1
fi
