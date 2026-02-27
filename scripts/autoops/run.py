#!/usr/bin/env python3
"""AutoOps execution engine — Stage 0 → Stage 1 → Core → Deliver.

Reads inputs/run.yaml (or path from argv[1]), enforces entry contract,
produces artifacts/_internal/* and artifacts/manifests/run.manifest.yaml.

Exit codes:
  0  — all stages completed, deliverables produced
  1  — entry contract incomplete (NEED_INPUT.md written)
  2  — stage gate failed (gaps written)
"""

from __future__ import annotations

import json
import sys
import glob
import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
INTERNAL = ARTIFACTS / "_internal"
MANIFESTS = ARTIFACTS / "manifests"
RUNBOOKS = ARTIFACTS / "runbooks"
PATCHES = ARTIFACTS / "patches"

for d in (INTERNAL, MANIFESTS, RUNBOOKS, PATCHES):
    d.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ── Minimal YAML reader (no PyYAML dependency required) ──────────────

def _read_input_yaml(path: Path) -> dict:
    """Parse flat key: value YAML (only top-level scalars)."""
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        data[k] = v
    return data


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_yaml_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Stage 0: Define problem ──────────────────────────────────────────

REQUIRED_FIELDS = ("TARGET", "REPO_CONTEXT", "SUCCESS_CRITERIA", "CONSTRAINTS", "PRIORITY")


def stage0(inputs: dict) -> bool:
    """Validate entry contract. Write problem/success/constraints artifacts.
    Returns True if gate passed."""
    missing = [f for f in REQUIRED_FIELDS if not inputs.get(f)]
    if missing:
        need = ARTIFACTS / "NEED_INPUT.md"
        lines = ["# NEED_INPUT"]
        for f in REQUIRED_FIELDS:
            val = inputs.get(f, "")
            marker = "  ← MISSING" if f in missing else ""
            lines.append(f"{f}={val}{marker}")
        need.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"STAGE0 FAIL: missing fields: {missing}", file=sys.stderr)
        return False

    _write_json(INTERNAL / "problem.statement.json", {
        "target": inputs["TARGET"],
        "scope": [inputs.get("REPO_CONTEXT", ".")],
        "non_goals": [],
        "known_symptoms": [],
        "suspected_area": [],
    })

    checks = []
    for part in inputs["SUCCESS_CRITERIA"].split(";"):
        part = part.strip()
        if part:
            checks.append({"name": part, "command": part, "expected": "exit 0"})
    if not checks:
        checks.append({"name": "manual", "command": "echo 'manual check'", "expected": "exit 0"})
    _write_json(INTERNAL / "success.criteria.json", {"checks": checks})

    constraints_raw = inputs.get("CONSTRAINTS", "")
    _write_json(INTERNAL / "constraints.json", {
        "network": {"allowed": "network=true" in constraints_raw.lower(), "allowed_domains": []},
        "secrets": {"no_print": True, "no_exfil": True},
        "write_scope": ["."],
        "forbidden_paths": [],
        "raw": constraints_raw,
    })

    print("STAGE0 PASS")
    return True


# ── Stage 1: Repo search ─────────────────────────────────────────────

def stage1(inputs: dict) -> bool:
    """Scan repo for baseline facts. Returns True if gate passed."""
    repo_context = inputs.get("REPO_CONTEXT", ".")

    # Collect workflow files
    wf_dir = REPO_ROOT / ".github" / "workflows"
    wf_files = sorted(str(p.relative_to(REPO_ROOT)) for p in wf_dir.glob("*.y*ml")) if wf_dir.is_dir() else []

    # Collect scripts
    scripts_dir = REPO_ROOT / "scripts"
    script_files = sorted(str(p.relative_to(REPO_ROOT)) for p in scripts_dir.rglob("*") if p.is_file()) if scripts_dir.is_dir() else []

    _write_yaml_lines(INTERNAL / "baseline.facts.yaml", [
        f"run_id: \"{RUN_ID}\"",
        f"repo_context: \"{repo_context}\"",
        "workflow_files:",
        *[f"  - \"{f}\"" for f in wf_files],
        "script_files:",
        *[f"  - \"{f}\"" for f in script_files[:30]],
        "observed_failures: []",
    ])

    _write_yaml_lines(INTERNAL / "hypotheses.yaml", [
        "# Auto-generated hypotheses (to be refined)",
        "- id: H0",
        f"  claim: \"target = {inputs.get('TARGET', '?')}\"",
        "  how_to_test: \"run success criteria commands\"",
        "  expected_signal: \"exit 0\"",
    ])

    _write_yaml_lines(INTERNAL / "gaps.yaml", [
        "# Gaps — blocking items that need resolution",
        "# (empty = no blocking gaps)",
    ])

    _write_yaml_lines(INTERNAL / "security.redlines.yaml", [
        "no_external_network: true",
        "no_secret_echo: true",
        "forbidden_actions:",
        "  - \"curl | bash\"",
        "  - \"upload secrets to artifacts\"",
    ])

    # Gate: check no blocking gaps
    gaps_content = (INTERNAL / "gaps.yaml").read_text()
    has_blocking = "blocking: true" in gaps_content
    if has_blocking:
        print("STAGE1 FAIL: blocking gaps remain", file=sys.stderr)
        return False

    print("STAGE1 PASS")
    return True


# ── Core: Decision + Plan ────────────────────────────────────────────

def core(inputs: dict) -> bool:
    """Produce decision record and plan. Returns True if gate passed."""
    target = inputs.get("TARGET", "")

    _write_yaml_lines(INTERNAL / "decision.record.yaml", [
        "decisions:",
        "  - id: D1",
        f"    problem: \"{target}\"",
        "    root_cause: \"see baseline.facts.yaml\"",
        "    fix:",
        "      type: \"patch\"",
        "      files: []",
        "    rollback:",
        "      type: \"git-revert\"",
        "      command: \"git revert HEAD\"",
        "    verify:",
        "      commands:",
        "        - \"bash artifacts/runbooks/verify.sh\"",
        "risks: []",
    ])

    _write_yaml_lines(INTERNAL / "plan.yaml", [
        "p0:",
        "  - \"apply patch\"",
        "  - \"run verify\"",
        "  - \"confirm rollback works\"",
        "p1:",
        "  - \"add policy gate\"",
        "  - \"update docs\"",
    ])

    # Gate: at least decision record exists
    if not (INTERNAL / "decision.record.yaml").exists():
        print("CORE FAIL: no decision record", file=sys.stderr)
        return False

    print("CORE PASS")
    return True


# ── Manifest ─────────────────────────────────────────────────────────

def write_manifest(inputs: dict) -> None:
    """Write run.manifest.yaml with all produced artifacts."""
    produced = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in ARTIFACTS.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )

    lines = [
        f"run_id: \"{RUN_ID}\"",
        f"target: \"{inputs.get('TARGET', '')}\"",
        f"repo_context: \"{inputs.get('REPO_CONTEXT', '')}\"",
        "inputs:",
        "  success_criteria: \"artifacts/_internal/success.criteria.json\"",
        "  constraints: \"artifacts/_internal/constraints.json\"",
        "artifacts:",
        *[f"  - path: \"{p}\"" for p in produced],
    ]
    _write_yaml_lines(MANIFESTS / "run.manifest.yaml", lines)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "inputs" / "run.yaml"
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    inputs = _read_input_yaml(input_path)

    # Stage 0
    if not stage0(inputs):
        write_manifest(inputs)
        return 1

    # Stage 1
    if not stage1(inputs):
        write_manifest(inputs)
        return 2

    # Core
    if not core(inputs):
        write_manifest(inputs)
        return 2

    # Final manifest
    write_manifest(inputs)
    print(f"AutoOps run complete: {RUN_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
