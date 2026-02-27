#!/usr/bin/env python3
import json, sys
from pathlib import Path

def _parse_min_yaml(path: Path) -> dict:
    # 極簡 YAML：僅支援本 repo 需要的 mapping/list/scalar
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.split("#", 1)[0].rstrip()
        if s.strip() == "":
            continue
        lines.append(s)

    root = {}
    stack = [( -1, root, None )]  # (indent, container, last_key)
    def current():
        return stack[-1][1], stack[-1][2]

    def set_last_key(k):
        stack[-1] = (stack[-1][0], stack[-1][1], k)

    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        text = line.lstrip(" ")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        container, last_key = current()

        if text.startswith("- "):
            item = text[2:].strip()
            if not isinstance(container, list):
                # current container was opened as empty mapping but is actually a list
                if len(stack) > 1 and isinstance(container, dict) and len(container) == 0:
                    stack.pop()
                parent = stack[-1][1]
                pk = stack[-1][2]
                if pk is None or not isinstance(parent, dict):
                    raise SystemExit(f"Invalid YAML near: {line}")
                parent[pk] = []
                container = parent[pk]
                stack[-1] = (stack[-1][0], parent, pk)
            container.append(_coerce(item))
            continue

        if ":" in text:
            k, v = text.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v == "":
                # open mapping
                if not isinstance(container, dict):
                    raise SystemExit(f"Invalid YAML mapping near: {line}")
                container[k] = {}
                set_last_key(k)
                stack.append((indent, container[k], None))
            else:
                if not isinstance(container, dict):
                    raise SystemExit(f"Invalid YAML mapping near: {line}")
                container[k] = _coerce(v)
                set_last_key(k)
            continue

        raise SystemExit(f"Unsupported YAML line: {line}")

    return root

def _coerce(v: str):
    if v in ("true", "True"): return True
    if v in ("false", "False"): return False
    if v.startswith('"') and v.endswith('"'): return v[1:-1]
    if v.startswith("'") and v.endswith("'"): return v[1:-1]
    return v

def enabled_modules(cap: dict) -> dict:
    out = {}
    for name, cfg in cap.items():
        if isinstance(cfg, dict) and cfg.get("enabled") is True:
            out[name] = cfg
    return out

if __name__ == "__main__":
    cap_path = Path(sys.argv[1] if len(sys.argv) > 1 else "capabilities.yaml")
    cap = _parse_min_yaml(cap_path)
    print(json.dumps(enabled_modules(cap), indent=2, sort_keys=True))
