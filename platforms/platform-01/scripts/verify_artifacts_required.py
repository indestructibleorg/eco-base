from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple


REQUIRED_FILES = [
    "manifest.json",
    "evidence.json",
    "topology_snapshot.json",
    "decision.json",
    "execution_log.jsonl",
    "verification_result.json",
    "final_report.md",
]


def list_runs(artifacts_root: str) -> List[str]:
    if not os.path.isdir(artifacts_root):
        return []
    runs = []
    for name in os.listdir(artifacts_root):
        p = os.path.join(artifacts_root, name)
        if os.path.isdir(p):
            runs.append(name)
    return sorted(runs)


def check_run_dir(run_dir: str) -> List[str]:
    missing = []
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(run_dir, f)):
            missing.append(f)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default="artifacts", help="artifacts root directory")
    ap.add_argument("--fail-on-empty", action="store_true", help="fail if no run dirs found")
    args = ap.parse_args()

    runs = list_runs(args.artifacts_root)
    if args.fail_on_empty and not runs:
        print(f"ERROR: no run dirs found under {args.artifacts_root}", file=sys.stderr)
        return 2

    failures: List[Tuple[str, List[str]]] = []
    for run_id in runs:
        run_dir = os.path.join(args.artifacts_root, run_id)
        missing = check_run_dir(run_dir)
        if missing:
            failures.append((run_id, missing))

    if failures:
        print("ARTIFACTS REQUIRED CHECK FAILED:", file=sys.stderr)
        for run_id, missing in failures:
            print(f"- run_id={run_id} missing={missing}", file=sys.stderr)
        return 1

    print(f"OK: {len(runs)} run(s) have all required artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
