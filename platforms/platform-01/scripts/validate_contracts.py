from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    import jsonschema
except Exception as e:
    print("ERROR: missing dependency jsonschema. Install it in requirements-dev.txt", file=sys.stderr)
    raise


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_schema(schemas_dir: str, schema_name: str) -> Dict[str, Any]:
    path = os.path.join(schemas_dir, schema_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"schema not found: {path}")
    return _load_json(path)


def validate_decisions(schema: Dict[str, Any], decision_files: List[str]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    validator = jsonschema.Draft202012Validator(schema)

    for fp in decision_files:
        try:
            data = _load_json(fp)
        except Exception as e:
            errors.append((fp, f"invalid json: {e}"))
            continue

        for err in sorted(validator.iter_errors(data), key=str):
            errors.append((fp, err.message))

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", default="contracts", help="schemas directory, default=contracts")
    ap.add_argument("--artifacts-glob", default="artifacts/**/decision.json", help="glob for decision.json")
    ap.add_argument("--schema-name", default="decision.schema.json", help="schema filename in schemas dir")
    ap.add_argument("--fail-on-missing", action="store_true", help="fail if no files found")
    args = ap.parse_args()

    schema = _load_schema(args.schemas, args.schema_name)
    files = sorted(glob.glob(args.artifacts_glob, recursive=True))

    if args.fail_on_missing and not files:
        print(f"ERROR: no decision.json found via glob: {args.artifacts_glob}", file=sys.stderr)
        return 2

    errors = validate_decisions(schema, files)
    if errors:
        print("CONTRACT VALIDATION FAILED:", file=sys.stderr)
        for fp, msg in errors[:200]:
            print(f"- {fp}: {msg}", file=sys.stderr)
        if len(errors) > 200:
            print(f"... and {len(errors) - 200} more", file=sys.stderr)
        return 1

    print(f"OK: validated {len(files)} decision.json against {args.schema_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
