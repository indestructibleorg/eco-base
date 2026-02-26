#!/usr/bin/env bash
# drill-eventbus-zone.sh
# Weekly drill: EventBus JetStream cross-zone distribution verification
# Artifacts: tests/reports/drill-eventbus-zone-<timestamp>.json
set -euo pipefail

NAMESPACE="argo-events"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_FILE="tests/reports/drill-eventbus-zone-${TIMESTAMP}.json"
mkdir -p tests/reports

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "=== EventBus Cross-Zone Distribution Drill ==="
log "Namespace: $NAMESPACE"

# ── Check 1: All 3 JetStream pods Running ────────────────────────────────
log "[Check 1] JetStream pod status..."
TOTAL=$(kubectl get pods -n "$NAMESPACE" -l eventbus-name=default --no-headers 2>/dev/null | wc -l)
RUNNING=$(kubectl get pods -n "$NAMESPACE" -l eventbus-name=default --no-headers 2>/dev/null | grep -c "Running" || true)

if [[ "$RUNNING" -eq 3 ]] && [[ "$TOTAL" -eq 3 ]]; then
  CHECK1="PASS"
  CHECK1_DETAIL="3/3 JetStream pods Running"
  log "PASS: $CHECK1_DETAIL"
else
  CHECK1="FAIL"
  CHECK1_DETAIL="${RUNNING}/${TOTAL} JetStream pods Running (expected 3/3)"
  log "FAIL: $CHECK1_DETAIL"
fi

# ── Check 2: Pods span ≥2 zones ──────────────────────────────────────────
log "[Check 2] Zone distribution..."
ZONES=()
for pod in $(kubectl get pods -n "$NAMESPACE" -l eventbus-name=default --no-headers -o custom-columns="NAME:.metadata.name" 2>/dev/null); do
  NODE=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "")
  if [[ -n "$NODE" ]]; then
    ZONE=$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null || echo "unknown")
    ZONES+=("$ZONE")
    log "  $pod → $NODE → $ZONE"
  fi
done

UNIQUE_ZONES=$(printf '%s\n' "${ZONES[@]}" | sort -u | wc -l)
if [[ "$UNIQUE_ZONES" -ge 2 ]]; then
  CHECK2="PASS"
  CHECK2_DETAIL="Pods span ${UNIQUE_ZONES} zones: $(printf '%s\n' "${ZONES[@]}" | sort -u | tr '\n' ' ')"
  log "PASS: $CHECK2_DETAIL"
else
  CHECK2="FAIL"
  CHECK2_DETAIL="Pods span only ${UNIQUE_ZONES} zone(s) — cross-zone distribution failed"
  log "FAIL: $CHECK2_DETAIL"
fi

# ── Check 3: PDB minAvailable=2 enforced ─────────────────────────────────
log "[Check 3] PDB enforcement..."
PDB_MIN=$(kubectl get pdb -n "$NAMESPACE" -o jsonpath='{.items[0].spec.minAvailable}' 2>/dev/null || echo "0")
PDB_DISRUPTIONS=$(kubectl get pdb -n "$NAMESPACE" -o jsonpath='{.items[0].status.disruptionsAllowed}' 2>/dev/null || echo "0")

if [[ "$PDB_MIN" -ge 2 ]]; then
  CHECK3="PASS"
  CHECK3_DETAIL="PDB minAvailable=${PDB_MIN}, disruptionsAllowed=${PDB_DISRUPTIONS}"
  log "PASS: $CHECK3_DETAIL"
else
  CHECK3="FAIL"
  CHECK3_DETAIL="PDB minAvailable=${PDB_MIN} (expected ≥2)"
  log "FAIL: $CHECK3_DETAIL"
fi

# ── Check 4: Zone anti-affinity or topologySpreadConstraints present ────────
log "[Check 4] Zone distribution enforcement (affinity/TSC)..."
STS_SPEC=$(kubectl get statefulset eventbus-default-js -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec}' 2>/dev/null || echo "")

if echo "$STS_SPEC" | grep -q 'topology.kubernetes.io/zone'; then
  CHECK4="PASS"
  CHECK4_DETAIL="Zone topology key present in StatefulSet spec (affinity or topologySpreadConstraints)"
  log "PASS: $CHECK4_DETAIL"
else
  CHECK4="FAIL"
  CHECK4_DETAIL="No zone topology key found in StatefulSet spec"
  log "FAIL: $CHECK4_DETAIL"
fi

# ── Determine gate result ─────────────────────────────────────────────────
if [[ "$CHECK1" == "PASS" ]] && [[ "$CHECK2" == "PASS" ]] && \
   [[ "$CHECK3" == "PASS" ]] && [[ "$CHECK4" == "PASS" ]]; then
  GATE_RESULT="PASS"
  GATE_MESSAGE="EventBus 3/3 pods Running, ≥2 zones, PDB enforced, topologySpreadConstraints active"
else
  GATE_RESULT="FAIL"
  GATE_MESSAGE="One or more EventBus zone distribution checks failed"
fi

log ""
log "=== Drill Result: $GATE_RESULT ==="
log "Report: $REPORT_FILE"

# ── Write JSON report ─────────────────────────────────────────────────────
python3 - <<PYEOF
import json
from datetime import datetime, timezone

report = {
    "drill": "EventBus Cross-Zone Distribution",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "namespace": "$NAMESPACE",
    "gate_result": "$GATE_RESULT",
    "gate_message": "$GATE_MESSAGE",
    "checks": {
        "pod_status": {"result": "$CHECK1", "detail": "$CHECK1_DETAIL"},
        "zone_distribution": {"result": "$CHECK2", "detail": "$CHECK2_DETAIL"},
        "pdb_enforcement": {"result": "$CHECK3", "detail": "$CHECK3_DETAIL"},
        "topology_spread": {"result": "$CHECK4", "detail": "$CHECK4_DETAIL"}
    },
    "zones_observed": $(printf '%s\n' "${ZONES[@]}" | sort -u | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))"),
    "unique_zone_count": $UNIQUE_ZONES
}

with open("$REPORT_FILE", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
PYEOF

if [[ "$GATE_RESULT" != "PASS" ]]; then
  exit 1
fi
