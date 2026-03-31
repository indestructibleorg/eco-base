#!/usr/bin/env bash
set -euo pipefail

echo "=== Supabase Migration ==="

# Validate required env vars
: "${SUPABASE_ACCESS_TOKEN:?SUPABASE_ACCESS_TOKEN is required}"
: "${SUPABASE_PROJECT_REF:?SUPABASE_PROJECT_REF is required}"
: "${SUPABASE_DB_PASSWORD:?SUPABASE_DB_PASSWORD is required}"

# Install Supabase CLI if not present
if ! command -v supabase &>/dev/null; then
  echo "Installing Supabase CLI..."
  npm install -g supabase@latest
fi

echo "Linking project: $SUPABASE_PROJECT_REF"
supabase link --project-ref "$SUPABASE_PROJECT_REF"

echo "Pushing migrations..."
supabase db push

# Deploy edge functions if they exist
if [[ -d "supabase/functions" ]]; then
  for fn_dir in supabase/functions/*/; do
    fn_name="$(basename "$fn_dir")"
    echo "Deploying function: $fn_name"
    supabase functions deploy "$fn_name" --project-ref "$SUPABASE_PROJECT_REF" || {
      echo "Warning: failed to deploy $fn_name" >&2
    }
  done
fi

echo "=== Supabase migration complete ==="
