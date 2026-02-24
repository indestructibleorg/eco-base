import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest


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


def _write(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_root(files_map):
    ordered = [files_map[k] for k in sorted(files_map.keys())]
    return sha3_512_join_hex(ordered)


# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture()
def repo_tmp(tmp_path: Path):
    """
    建一個最小 repo 結構在 tmp_path，讓 scripts 可對 artifacts/contracts 運作。
    """
    # contracts
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    schema_src = PROJECT_ROOT / "contracts" / "decision.schema.json"
    if schema_src.exists():
        (contracts / "decision.schema.json").write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        pytest.skip("contracts/decision.schema.json not found in repo; add it from the kit")
    
    # 复制 scripts 到临时目录
    scripts_src = PROJECT_ROOT / "scripts"
    scripts_dst = tmp_path / "scripts"
    scripts_dst.mkdir(exist_ok=True)
    for script_file in ["validate_contracts.py", "verify_artifacts_required.py", "verify_manifest.py"]:
        src = scripts_src / script_file
        if src.exists():
            (scripts_dst / script_file).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # artifacts/run-001
    run_dir = tmp_path / "artifacts" / "run-001"
    run_dir.mkdir(parents=True)

    _write_json(run_dir / "evidence.json", {"k": "v"})
    _write_json(run_dir / "topology_snapshot.json", {"services": []})
    _write(run_dir / "execution_log.jsonl", '{"ts":1,"msg":"ok"}\n')
    _write_json(run_dir / "verification_result.json", {"passed": True})
    _write(run_dir / "final_report.md", "# ok\n")

    decision = {
        "trace_id": "trace-00000001",
        "run_id": "run-001",
        "risk": {"level": "LOW", "score": 0.1},
        "actions": [
            {"action_id": "action-00000001", "apply": {"type": "noop"}, "verify": {"type": "noop"}, "rollback": {"type": "noop"}}
        ],
        "evidence": {"source": "unit-test"},
        "versions": {"contracts": "v0.1.0"}
    }
    _write_json(run_dir / "decision.json", decision)

    # manifest
    files_map = {}
    for rel in [
        "evidence.json",
        "topology_snapshot.json",
        "decision.json",
        "execution_log.jsonl",
        "verification_result.json",
        "final_report.md",
    ]:
        files_map[rel] = sha3_512_file(str(run_dir / rel))

    manifest = {
        "algo": "sha3-512",
        "files": files_map,
        "root": _compute_root(files_map)
    }
    _write_json(run_dir / "manifest.json", manifest)

    return tmp_path


def _run(cmd, cwd: Path):
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def test_validate_contracts_ok(repo_tmp: Path):
    code, out, err = _run(
        [sys.executable, "scripts/validate_contracts.py", "--schemas", "contracts", "--artifacts-glob", "artifacts/**/decision.json", "--fail-on-missing"],
        cwd=repo_tmp,
    )
    assert code == 0, (out, err)


def test_verify_artifacts_required_ok(repo_tmp: Path):
    code, out, err = _run(
        [sys.executable, "scripts/verify_artifacts_required.py", "--artifacts-root", "artifacts", "--fail-on-empty"],
        cwd=repo_tmp,
    )
    assert code == 0, (out, err)


def test_verify_manifest_ok(repo_tmp: Path):
    code, out, err = _run(
        [sys.executable, "scripts/verify_manifest.py", "--artifacts-root", "artifacts", "--algo", "sha3-512", "--fail-on-empty"],
        cwd=repo_tmp,
    )
    assert code == 0, (out, err)


def test_verify_manifest_detects_tamper(repo_tmp: Path):
    # 篡改一個檔案，manifest 應該 fail
    fp = repo_tmp / "artifacts" / "run-001" / "final_report.md"
    fp.write_text("# tampered\n", encoding="utf-8")

    code, out, err = _run(
        [sys.executable, "scripts/verify_manifest.py", "--artifacts-root", "artifacts", "--algo", "sha3-512", "--fail-on-empty"],
        cwd=repo_tmp,
    )
    assert code != 0
    assert "digest mismatch" in err or "root mismatch" in err
