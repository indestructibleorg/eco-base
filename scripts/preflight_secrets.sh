#!/usr/bin/env bash
set -euo pipefail

CAP_JSON="${1:-artifacts/_tmp/capabilities.enabled.json}"
REQ_FILE="${2:-secrets.required.yaml}"

mkdir -p artifacts/_tmp

if [[ ! -f "$CAP_JSON" ]]; then
  echo "missing $CAP_JSON (run scripts/capabilities.py first)" >&2
  exit 2
fi
if [[ ! -f "$REQ_FILE" ]]; then
  echo "missing $REQ_FILE" >&2
  exit 2
fi

# 解析 required secrets（只取 required: 下的 - NAME）
# 以模組區塊切段：<module>: ... required: ... - SECRET
enabled_modules="$(python3 - "$CAP_JSON" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],'r',encoding='utf-8'))
print("\n".join(sorted(d.keys())))
PY
)"

missing_any=0
missing_report=""

for mod in $enabled_modules; do
  reqs="$(awk -v m="$mod" '
    $0 ~ ("^"m":") {inmod=1; next}
    inmod && $0 ~ "^[a-zA-Z0-9_-]+:" {inmod=0}
    inmod && $0 ~ "^  required:" {inreq=1; next}
    inmod && inreq && $0 ~ "^  [a-zA-Z0-9_-]+:" {inreq=0}
    inmod && inreq && $0 ~ "^    - " {sub("^    - ",""); print; next}
  ' "$REQ_FILE")"

  [[ -z "$reqs" ]] && continue

  while IFS= read -r s; do
    [[ -z "$s" ]] && continue
    v="${!s-}"
    if [[ -z "$v" ]]; then
      missing_any=1
      missing_report+="${mod}: ${s}\n"
    fi
  done <<< "$reqs"
done

if [[ "$missing_any" -eq 1 ]]; then
  echo "Missing required secrets for enabled capabilities:"
  printf "%b" "$missing_report"
  exit 1
fi

echo "Preflight secrets: OK"
