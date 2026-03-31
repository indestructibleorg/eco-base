#!/usr/bin/env bash
set -euo pipefail

echo "=== n8n Workflow Sync ==="

: "${N8N_BASE_URL:?N8N_BASE_URL is required}"
: "${N8N_API_KEY:?N8N_API_KEY is required}"

WORKFLOW_DIR="ops/n8n/workflows"

# Health check
echo "Checking n8n health..."
curl -sf "${N8N_BASE_URL}/healthz" >/dev/null || {
  echo "Warning: n8n health check failed (endpoint may not exist)" >&2
}

# Import workflows if they exist
if [[ -d "$WORKFLOW_DIR" ]]; then
  for wf_file in "$WORKFLOW_DIR"/*.json; do
    [[ -f "$wf_file" ]] || continue
    wf_name="$(python3 -c "import json; print(json.load(open('$wf_file')).get('name', 'unknown'))")"
    echo "Importing workflow: $wf_name"
    curl -sf -X POST "${N8N_BASE_URL}/api/v1/workflows" \
      -H "X-N8N-API-KEY: ${N8N_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "@${wf_file}" || {
      echo "Warning: failed to import workflow $wf_name" >&2
    }
  done
else
  echo "No workflow files found in $WORKFLOW_DIR — skipping import"
fi

echo "=== n8n sync complete ==="
