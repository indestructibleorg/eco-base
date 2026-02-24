from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple


def sha3_512_file(path: str) -> str:
    h = hashlib.sha3_512()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha3_512_join_hex(hex_digests: Iterable[str]) -> str:
    h = hashlib.sha3_512()
    for d in hex_digests:
        h.update(bytes.fromhex(d))
    return h.hexdigest()


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_root(files_map: Dict[str, str]) -> str:
    # stable: sort by relative path
    ordered = [files_map[k] for k in sorted(files_map.keys())]
    return sha3_512_join_hex(ordered)


def verify_one_run(run_dir: str, algo: str) -> List[str]:
    errors: List[str] = []
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return [f"missing manifest.json in {run_dir}"]

    try:
        manifest = _load_json(manifest_path)
    except Exception as e:
        return [f"invalid manifest.json: {e}"]

    m_algo = manifest.get("algo")
    if m_algo != algo:
        errors.append(f"algo mismatch: manifest={m_algo} expected={algo}")

    files_map = manifest.get("files")
    if not isinstance(files_map, dict) or not files_map:
        errors.append("manifest.files must be a non-empty object")
        return errors

    # verify each file digest
    for rel, expected_hex in files_map.items():
        if not isinstance(rel, str) or not isinstance(expected_hex, str):
            errors.append(f"invalid entry types: {rel} -> {expected_hex}")
            continue

        fp = os.path.join(run_dir, rel)
        if not os.path.exists(fp):
            errors.append(f"missing file referenced by manifest: {rel}")
            continue

        actual = sha3_512_file(fp)
        if actual != expected_hex:
            errors.append(f"digest mismatch: {rel} expected={expected_hex} actual={actual}")

    # verify root
    expected_root = manifest.get("root")
    if not isinstance(expected_root, str) or len(expected_root) < 16:
        errors.append("manifest.root missing/invalid")
    else:
        actual_root = _compute_root(files_map)
        if actual_root != expected_root:
            errors.append(f"root mismatch: expected={expected_root} actual={actual_root}")

    return errors


def list_runs(artifacts_root: str) -> List[str]:
    if not os.path.isdir(artifacts_root):
        return []
    return sorted([d for d in os.listdir(artifacts_root) if os.path.isdir(os.path.join(artifacts_root, d))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--algo", default="sha3-512")
    ap.add_argument("--fail-on-empty", action="store_true")
    args = ap.parse_args()

    if args.algo != "sha3-512":
        print("ERROR: only sha3-512 is supported in this minimal verifier", file=sys.stderr)
        return 2

    runs = list_runs(args.artifacts_root)
    if args.fail_on_empty and not runs:
        print(f"ERROR: no run dirs found under {args.artifacts_root}", file=sys.stderr)
        return 2

    all_errors: List[Tuple[str, List[str]]] = []
    for run_id in runs:
        run_dir = os.path.join(args.artifacts_root, run_id)
        errs = verify_one_run(run_dir, args.algo)
        if errs:
            all_errors.append((run_id, errs))

    if all_errors:
        print("MANIFEST VERIFICATION FAILED:", file=sys.stderr)
        for run_id, errs in all_errors:
            print(f"- run_id={run_id}", file=sys.stderr)
            for e in errs:
                print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"OK: verified manifest for {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
