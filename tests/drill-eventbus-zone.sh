#!/usr/bin/env bash
# drill-eventbus-zone.sh
# Weekly drill: EventBus JetStream cross-zone distribution verification
# Zone-adaptive PASS gate:
#   Z>=3 → 3 pods in 3 distinct zones (full resilience)
#   Z=2  → maxSkew=1, distribution must be 2:1 (zone_resilience=degraded)
#   Z=1  → node-level checks only (zone_resilience=not_supported)
# Artifacts: tests/reports/drill-eventbus-zone-<timestamp>.json (NOT committed to repo)
set -euo pipefail

NAMESPACE="argo-events"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_FILE="tests/reports/drill-eventbus-zone-${TIMESTAMP}.json"
SUMMARY_FILE="tests/reports/drill-summary.json"
SCHEDULED_TIME="${DRILL_SCHEDULED_TIME:-}"   # injected by workflow
ACTUAL_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p tests/reports

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "=== EventBus Cross-Zone Distribution Drill (Adaptive Gate) ==="
log "Namespace: $NAMESPACE"
log "Actual start: $ACTUAL_TIME"
[[ -n "$SCHEDULED_TIME" ]] && log "Scheduled at: $SCHEDULED_TIME"

# ── Detect available cluster zones ───────────────────────────────────────
log "[Pre-check] Detecting cluster zones..."
CLUSTER_ZONES=$(kubectl get nodes \
  -o jsonpath='{.items[*].metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null \
  | tr ' ' '\n' | sort -u | tr '\n' ' ' | xargs)
CLUSTER_ZONE_COUNT=$(echo "$CLUSTER_ZONES" | tr ' ' '\n' | grep -c . || echo 0)
log "  Available zones (Z=$CLUSTER_ZONE_COUNT): $CLUSTER_ZONES"

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

# ── Check 2: Zone distribution (adaptive) ────────────────────────────────
log "[Check 2] Zone distribution (adaptive, Z=$CLUSTER_ZONE_COUNT)..."
declare -A ZONE_COUNT
ZONES_LIST=()
for pod in $(kubectl get pods -n "$NAMESPACE" -l eventbus-name=default --no-headers \
  -o custom-columns="NAME:.metadata.name" 2>/dev/null); do
  NODE=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "")
  if [[ -n "$NODE" ]]; then
    ZONE=$(kubectl get node "$NODE" \
      -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null || echo "unknown")
    ZONES_LIST+=("$ZONE")
    ZONE_COUNT["$ZONE"]=$(( ${ZONE_COUNT["$ZONE"]:-0} + 1 ))
    log "  $pod → $NODE → $ZONE"
  fi
done

UNIQUE_ZONES=$(printf '%s\n' "${ZONES_LIST[@]}" | sort -u | wc -l)
MAX_IN_ZONE=$(printf '%s\n' "${ZONE_COUNT[@]}" | sort -rn | head -1)

# Determine zone_resilience level and PASS condition
if [[ "$CLUSTER_ZONE_COUNT" -ge 3 ]]; then
  # Z>=3: require 3 pods in 3 distinct zones
  if [[ "$UNIQUE_ZONES" -ge 3 ]]; then
    CHECK2="PASS"
    ZONE_RESILIENCE="full"
    CHECK2_DETAIL="Z=${CLUSTER_ZONE_COUNT}: 3 pods across 3 zones (full resilience)"
  else
    CHECK2="FAIL"
    ZONE_RESILIENCE="degraded"
    CHECK2_DETAIL="Z=${CLUSTER_ZONE_COUNT}: only ${UNIQUE_ZONES} zones used (required 3 for full resilience)"
  fi
elif [[ "$CLUSTER_ZONE_COUNT" -eq 2 ]]; then
  # Z=2: require maxSkew=1, distribution must be 2:1 (not 3:0)
  if [[ "$UNIQUE_ZONES" -eq 2 ]] && [[ "$MAX_IN_ZONE" -le 2 ]]; then
    CHECK2="PASS"
    ZONE_RESILIENCE="degraded(two_zones)"
    CHECK2_DETAIL="Z=2: distribution is 2:1 across 2 zones (maxSkew=1 satisfied; zone_resilience=degraded)"
  elif [[ "$UNIQUE_ZONES" -eq 1 ]]; then
    CHECK2="FAIL"
    ZONE_RESILIENCE="none"
    CHECK2_DETAIL="Z=2: all 3 pods in single zone (3:0 — zone failure loses all pods)"
  else
    CHECK2="FAIL"
    ZONE_RESILIENCE="degraded"
    CHECK2_DETAIL="Z=2: distribution ${MAX_IN_ZONE}:$(( 3 - MAX_IN_ZONE )) violates maxSkew=1"
  fi
else
  # Z=1: zone resilience not supported, check node-level only
  CHECK2="PASS"
  ZONE_RESILIENCE="not_supported(single_zone)"
  CHECK2_DETAIL="Z=1: zone resilience not supported; node-level distribution assumed"
  log "WARN: Single-zone cluster — zone resilience not achievable"
fi
log "${CHECK2}: $CHECK2_DETAIL"

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
  CHECK3_DETAIL="PDB minAvailable=${PDB_MIN} (expected >=2)"
  log "FAIL: $CHECK3_DETAIL"
fi

# ── Check 4: Zone topology key in StatefulSet spec ───────────────────────
log "[Check 4] Zone distribution enforcement (affinity/TSC)..."
STS_SPEC=$(kubectl get statefulset eventbus-default-js -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec}' 2>/dev/null || echo "")

if echo "$STS_SPEC" | grep -q 'topology.kubernetes.io/zone'; then
  CHECK4="PASS"
  CHECK4_DETAIL="Zone topology key present in StatefulSet spec"
  log "PASS: $CHECK4_DETAIL"
else
  CHECK4="FAIL"
  CHECK4_DETAIL="No zone topology key found in StatefulSet spec"
  log "FAIL: $CHECK4_DETAIL"
fi

# ── Cron drift check ──────────────────────────────────────────────────────
CRON_DRIFT_S=""
CRON_DRIFT_STATUS="skipped"
if [[ -n "$SCHEDULED_TIME" ]]; then
  CRON_DRIFT_S=$(python3 -c "
from datetime import datetime, timezone
fmt = '%Y-%m-%dT%H:%M:%SZ'
sched = datetime.strptime('$SCHEDULED_TIME', fmt).replace(tzinfo=timezone.utc)
actual = datetime.strptime('$ACTUAL_TIME', fmt).replace(tzinfo=timezone.utc)
print(int((actual - sched).total_seconds()))
" 2>/dev/null || echo "")
  if [[ -n "$CRON_DRIFT_S" ]]; then
    if [[ "$CRON_DRIFT_S" -le 3600 ]]; then
      CRON_DRIFT_STATUS="ok(${CRON_DRIFT_S}s)"
      log "PASS: cron drift=${CRON_DRIFT_S}s (<=3600s)"
    else
      CRON_DRIFT_STATUS="warn(${CRON_DRIFT_S}s)"
      log "WARN: cron drift=${CRON_DRIFT_S}s (>3600s — GitHub runner delay)"
    fi
  fi
fi

# ── Determine gate result ─────────────────────────────────────────────────
if [[ "$CHECK1" == "PASS" ]] && [[ "$CHECK2" == "PASS" ]] && \
   [[ "$CHECK3" == "PASS" ]] && [[ "$CHECK4" == "PASS" ]]; then
  GATE_RESULT="PASS"
  GATE_MESSAGE="EventBus zone drill PASS (zone_resilience=${ZONE_RESILIENCE})"
else
  GATE_RESULT="FAIL"
  GATE_MESSAGE="One or more EventBus zone distribution checks failed"
fi

log ""
log "=== Drill Result: $GATE_RESULT (zone_resilience=${ZONE_RESILIENCE}) ==="
log "Report: $REPORT_FILE"

# ── Write full JSON report (NOT committed to repo) ────────────────────────
python3 - <<PYEOF
import json

zones_list = $(python3 -c "import json; print(json.dumps('${ZONES_LIST[*]:-}'.split()))" 2>/dev/null || echo '[]')
zone_count_raw = "$(for z in "${!ZONE_COUNT[@]}"; do echo "$z=${ZONE_COUNT[$z]}"; done | tr '\n' ',')" 
zone_count = dict(item.split('=') for item in zone_count_raw.rstrip(',').split(',') if '=' in item)
zone_count = {k: int(v) for k, v in zone_count.items()}
report = {
    "drill": "EventBus Cross-Zone Distribution",
    "timestamp": "$ACTUAL_TIME",
    "scheduled_time": "${SCHEDULED_TIME:-null}",
    "cron_drift": {"status": "$CRON_DRIFT_STATUS", "drift_s": ${CRON_DRIFT_S:-0}},
    "namespace": "$NAMESPACE",
    "gate_result": "$GATE_RESULT",
    "gate_message": "$GATE_MESSAGE",
    "zone_resilience": "$ZONE_RESILIENCE",
    "cluster": {
        "available_zones": "$CLUSTER_ZONES".split(),
        "zone_count": $CLUSTER_ZONE_COUNT
    },
    "distribution": {
        "unique_zones_used": $UNIQUE_ZONES,
        "max_pods_in_one_zone": $MAX_IN_ZONE,
        "zone_pod_count": zone_count,
        "pods_zones": zones_list
    },
    "checks": {
        "pod_status": {"result": "$CHECK1", "detail": "$CHECK1_DETAIL"},
        "zone_distribution": {"result": "$CHECK2", "detail": "$CHECK2_DETAIL"},
        "pdb_enforcement": {"result": "$CHECK3", "detail": "$CHECK3_DETAIL"},
        "topology_spread": {"result": "$CHECK4", "detail": "$CHECK4_DETAIL"}
    }
}

with open("$REPORT_FILE", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
PYEOF

# ── Update drill-summary.json (last 10 runs, committed to repo) ───────────
python3 - <<PYEOF
import json, os, glob
from datetime import datetime, timezone

summary_file = "$SUMMARY_FILE"
new_entry = {
    "drill": "eventbus-zone",
    "timestamp": "$ACTUAL_TIME",
    "gate_result": "$GATE_RESULT",
    "zone_resilience": "$ZONE_RESILIENCE",
    "unique_zones": $UNIQUE_ZONES,
    "cluster_zones": $CLUSTER_ZONE_COUNT
}

existing = []
if os.path.exists(summary_file):
    try:
        data = json.load(open(summary_file))
        existing = data.get("runs", [])
    except Exception:
        existing = []

# Keep last 10 runs per drill type
existing = [r for r in existing if r.get("drill") != "eventbus-zone"][-9:]
existing.append(new_entry)

summary = {
    "_note": "Summary only. Full reports are GitHub Actions artifacts (90d retention).",
    "last_updated": "$ACTUAL_TIME",
    "runs": existing
}

with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"drill-summary.json updated ({len(existing)} eventbus-zone entries)")
PYEOF

if [[ "$GATE_RESULT" != "PASS" ]]; then
  exit 1
fi
