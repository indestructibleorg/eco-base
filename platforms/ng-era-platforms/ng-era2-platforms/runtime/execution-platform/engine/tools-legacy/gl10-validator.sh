#!/usr/bin/env bash
set -e

# Validate GL10-GL19 artifacts conform to GL10-schema.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GL10_DIR="$REPO_ROOT/gl10"

cd "$GL10_DIR" || exit 1
for f in GL1*.json .gl10-index.json; do
  [ -f "$f" ] || continue
  # Skip schema file itself
  [[ "$f" == "GL10-schema.json" ]] && continue
  python3 - <<PY
import json,sys
from jsonschema import validate,ValidationError
schema=json.load(open("$GL10_DIR/GL10-schema.json"))
data=json.load(open("$GL10_DIR/$f"))
try:
    validate(instance=data,schema=schema)
except ValidationError as e:
    print("GL10 VALIDATION FAILED:", "$f", e)
    sys.exit(2)
print("GL10 VALIDATION OK:", "$f")
PY
done
exit 0
