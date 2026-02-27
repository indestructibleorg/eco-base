#!/usr/bin/env bash
set -euo pipefail

echo "=== Keycloak Realm Sync ==="

: "${KEYCLOAK_ADMIN_URL:?KEYCLOAK_ADMIN_URL is required}"
: "${KEYCLOAK_ADMIN_CLIENT_ID:?KEYCLOAK_ADMIN_CLIENT_ID is required}"
: "${KEYCLOAK_ADMIN_CLIENT_SECRET:?KEYCLOAK_ADMIN_CLIENT_SECRET is required}"

REALM_DIR="ops/keycloak/realms"

# Obtain admin token
echo "Authenticating to Keycloak..."
TOKEN=$(curl -sf -X POST "${KEYCLOAK_ADMIN_URL}/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${KEYCLOAK_ADMIN_CLIENT_ID}" \
  -d "client_secret=${KEYCLOAK_ADMIN_CLIENT_SECRET}" | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

echo "Authenticated successfully"

# Import realm configs if they exist
if [[ -d "$REALM_DIR" ]]; then
  for realm_file in "$REALM_DIR"/*.json; do
    [[ -f "$realm_file" ]] || continue
    realm_name="$(python3 -c "import json; print(json.load(open('$realm_file'))['realm'])")"
    echo "Importing realm: $realm_name"
    curl -sf -X POST "${KEYCLOAK_ADMIN_URL}/admin/realms" \
      -H "Authorization: Bearer ${TOKEN}" \
      -H "Content-Type: application/json" \
      -d "@${realm_file}" || {
      echo "Realm exists, updating..."
      curl -sf -X PUT "${KEYCLOAK_ADMIN_URL}/admin/realms/${realm_name}" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "@${realm_file}"
    }
  done
else
  echo "No realm files found in $REALM_DIR — skipping import"
fi

echo "=== Keycloak sync complete ==="
