#!/usr/bin/env bash
set -e

# GL Validator Script
# Validates GL JSON files against the GL schema
# Updated to support new project-docs directory structure

# Define schema location (check multiple possible paths)
SCHEMA_PATH=""
SCHEMA_LOCATIONS=(
  "project-docs/gl-governance/schemas/GL-schema-singleline.json"
  "GL-schema-singleline.json"
  "schemas/GL-schema-singleline.json"
)

for loc in "${SCHEMA_LOCATIONS[@]}"; do
  if [ -f "$loc" ]; then
    SCHEMA_PATH="$loc"
    echo "Found schema at: $SCHEMA_PATH"
    break
  fi
done

if [ -z "$SCHEMA_PATH" ]; then
  echo "Warning: GL-schema-singleline.json not found, skipping schema validation"
  echo "Searched locations: ${SCHEMA_LOCATIONS[*]}"
  exit 0
fi

# Validate GL files in root directory
for f in GL00-root-semantic-anchor.json GL??-*.json GL99-*.json .gl-index.json; do
  [ -f "$f" ] || continue
  echo "Validating $f..."
  python3 -c "import json,sys; from jsonschema import validate,ValidationError; schema=json.load(open('$SCHEMA_PATH')); data=json.load(open('$f')); validate(instance=data,schema=schema)" || { echo "GL VALIDATION FAILED: $f"; exit 2; }
  echo "GL VALIDATION OK: $f"
done

# Validate GL files in project-docs/gl-governance/schemas directory
for f in project-docs/gl-governance/schemas/GL00-root-semantic-anchor.json project-docs/gl-governance/schemas/GL??-*.json project-docs/gl-governance/schemas/GL99-*.json; do
  [ -f "$f" ] || continue
  echo "Validating $f..."
  python3 -c "import json,sys; from jsonschema import validate,ValidationError; schema=json.load(open('$SCHEMA_PATH')); data=json.load(open('$f')); validate(instance=data,schema=schema)" || { echo "GL VALIDATION FAILED: $f"; exit 2; }
  echo "GL VALIDATION OK: $f"
done

echo "GL Validation completed successfully"
exit 0