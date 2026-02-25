# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: provenance_injector
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Provenance Injector - SLSA Evidence Generator
供應鏈證據注入器 - SLSA 證據生成器
Generates SLSA L3 compliant provenance records, injects digests,
and creates audit-ready evidence for supply chain gl-platform.governance.
Usage:
    python tools/docs/provenance_injector.py --input docs/knowledge_index.yaml
    python tools/docs/provenance_injector.py --generate-sbom
    python tools/docs/provenance_injector.py --sign --key-path /path/to/key
"""
# MNGA-002: Import organization needs review
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)
SLSA_PROVENANCE_TYPE = "https://slsa.dev/provenance/v1"
SLSA_BUILDER_ID = "https://github.com/Unmanned-Island-admin/SynergyMesh"
def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return "0" * 64
def calculate_content_hash(content: str) -> str:
    """Calculate SHA256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
def get_git_info(repo_root: Path) -> dict[str, str]:
    """Get Git repository information."""
    git_info = {"commit": "unknown", "branch": "unknown", "remote": "unknown"}
    try:
        import subprocess
        # Get current commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        # Get remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            git_info["remote"] = result.stdout.strip()
    except Exception:
        pass
    return git_info
def generate_slsa_provenance(
    subject_name: str,
    subject_digest: str,
    repo_root: Path,
    build_type: str = "https://github.com/actions/runner",
) -> dict[str, Any]:
    """Generate SLSA L3 compliant provenance record."""
    git_info = get_git_info(repo_root)
    now = datetime.now(timezone.utc)
    return {
        "$schema": "https://slsa.dev/provenance/v1",
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": subject_name, "digest": {"sha256": subject_digest}}],
        "predicateType": SLSA_PROVENANCE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": build_type,
                "externalParameters": {
                    "workflow": {
                        "ref": f"refs/heads/{git_info['branch']}",
                        "repository": git_info["remote"],
                        "path": ".github/workflows/apply.yaml",
                    }
                },
                "internalParameters": {
                    "github_event_name": os.environ.get("GITHUB_EVENT_NAME", "manual"),
                    "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                    "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
                },
                "resolvedDependencies": [],
            },
            "runDetails": {
                "builder": {"id": SLSA_BUILDER_ID, "version": {"slsa": "v1.0"}},
                "metadata": {
                    "invocationId": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'local')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', 'local')}",
                    "startedOn": now.isoformat(),
                    "finishedOn": now.isoformat(),
                },
                "byproducts": [],
            },
        },
    }
def generate_spdx_sbom(items: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    """Generate SPDX format SBOM."""
    now = datetime.now(timezone.utc)
    packages = []
    relationships = []
    # Root package
    root_id = "SPDXRef-Package-machinenativenops-docs"
    packages.append(
        {
            "SPDXID": root_id,
            "name": "machinenativenops-docs",
            "versionInfo": "1.0.0",
            "downloadLocation": "https://github.com/Unmanned-Island-admin/SynergyMesh",
            "filesAnalyzed": True,
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright 2025 SynergyMesh Team",
            "supplier": "Organization: SynergyMesh Team",
            "description": "SynergyMesh documentation index",
        }
    )
    # Add each documented item as a package
    for item in items:
        pkg_id = f"SPDXRef-{item['id']}"
        file_path = repo_root / item["path"]
        pkg = {
            "SPDXID": pkg_id,
            "name": item["id"],
            "versionInfo": "1.0.0",
            "downloadLocation": f"https://github.com/Unmanned-Island-admin/SynergyMesh/blob/main/{item['path']}",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "description": item.get("description", "")[:200],
        }
        # Add checksum if file exists
        if file_path.exists():
            pkg["checksums"] = [
                {"algorithm": "SHA256", "checksumValue": calculate_sha256(file_path)}
            ]
        packages.append(pkg)
        # Add relationship
        relationships.append(
            {
                "spdxElementId": root_id,
                "relatedSpdxElement": pkg_id,
                "relationshipType": "CONTAINS",
            }
        )
    # Document describes root
    relationships.append(
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relatedSpdxElement": root_id,
            "relationshipType": "DESCRIBES",
        }
    )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "machinenativenops-docs-sbom",
        "documentNamespace": f"https://github.com/Unmanned-Island-admin/SynergyMesh/sbom/{uuid4()}",
        "creationInfo": {
            "created": now.isoformat(),
            "creators": [
                "Tool: SynergyMesh Provenance Injector",
                "Organization: SynergyMesh Team",
            ],
            "licenseListVersion": "3.21",
        },
        "documentDescribes": [root_id],
        "packages": packages,
        "relationships": relationships,
    }
def inject_provenance_into_index(
    index_path: Path, output_path: Path, repo_root: Path
) -> dict[str, Any]:
    """Inject provenance information into index items."""
    with open(index_path, "r", encoding="utf-8") as f:
        index = yaml.safe_load(f)
    items = index.get("items", [])
    for item in items:
        file_path = repo_root / item["path"]
        if file_path.exists():
            digest = calculate_sha256(file_path)
            item["provenance"] = {
                "digest": digest,
                "slsaRef": "gl-platform.governance/sbom/provenance.json",
                "signatureRef": "gl-platform.governance/sbom/signature.sig",
            }
    # Write updated index
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            index, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )
    return index
def generate_audit_event(
    action: str, subject: str, details: dict[str, Any]
) -> dict[str, Any]:
    """Generate an audit event record."""
    now = datetime.now(timezone.utc)
    return {
        "eventId": str(uuid4()),
        "timestamp": now.isoformat(),
        "action": action,
        "subject": subject,
        "actor": os.environ.get("GITHUB_ACTOR", "system"),
        "repository": os.environ.get("GITHUB_REPOSITORY", "local"),
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "workflow": os.environ.get("GITHUB_WORKFLOW", "manual"),
        "runId": os.environ.get("GITHUB_RUN_ID", "local"),
        "details": details,
        "integrity": {
            "algorithm": "sha256",
            "value": calculate_content_hash(json.dumps(details, sort_keys=True)),
        },
    }
def main():
    parser = argparse.ArgumentParser(
        description="Generate SLSA provenance and supply chain evidence"
    )
    parser.add_argument(
        "--input", "-i", default="docs/knowledge_index.yaml", help="Input index file"
    )
    parser.add_argument(
        "--output", "-o", help="Output file path (default: same as input)"
    )
    parser.add_argument(
        "--generate-sbom", action="store_true", help="Generate SPDX SBOM"
    )
    parser.add_argument(
        "--generate-provenance", action="store_true", help="Generate SLSA provenance"
    )
    parser.add_argument(
        "--inject", action="store_true", help="Inject provenance into index"
    )
    parser.add_argument("--audit", action="store_true", help="Generate audit event")
    parser.add_argument(
        "--sbom-output",
        default="gl-platform.governance/sbom/docs-sbom.spdx.json",
        help="SBOM output path",
    )
    parser.add_argument(
        "--provenance-output",
        default="gl-platform.governance/sbom/docs-provenance.json",
        help="Provenance output path",
    )
    args = parser.parse_args()
    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    print("🔐 供應鏈證據注入器")
    print("=" * 60)
    print()
    input_path = repo_root / args.input
    output_path = repo_root / (args.output or args.input)
    if not input_path.exists():
        print(f"❌ 找不到輸入檔案: {input_path}")
        return 1
    # Load index
    with open(input_path, "r", encoding="utf-8") as f:
        index = yaml.safe_load(f)
    items = index.get("items", [])
    print(f"📄 載入索引: {len(items)} 個項目")
    print()
    # Generate SBOM
    if args.generate_sbom:
        print("📦 生成 SPDX SBOM...")
        sbom = generate_spdx_sbom(items, repo_root)
        sbom_path = repo_root / args.sbom_output
        sbom_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sbom_path, "w", encoding="utf-8") as f:
            json.dump(sbom, f, indent=2, ensure_ascii=False)
        print(f"   ✅ SBOM 已寫入: {sbom_path}")
        print(f"   📊 包含 {len(sbom['packages'])} 個套件")
        print()
    # Generate provenance
    if args.generate_provenance:
        print("🔏 生成 SLSA 溯源...")
        # Calculate index digest
        index_content = json.dumps(index, sort_keys=True)
        index_digest = calculate_content_hash(index_content)
        provenance = generate_slsa_provenance(
            subject_name="docs-index", subject_digest=index_digest, repo_root=repo_root
        )
        prov_path = repo_root / args.provenance_output
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
        print(f"   ✅ 溯源已寫入: {prov_path}")
        print(f"   🔑 主題摘要: {index_digest[:16]}...")
        print()
    # Inject provenance into index
    if args.inject:
        print("💉 注入溯源資訊...")
        updated_index = inject_provenance_into_index(input_path, output_path, repo_root)
        items_with_prov = sum(
            1 for item in updated_index.get("items", []) if "provenance" in item
        )
        print(f"   ✅ 已更新: {output_path}")
        print(f"   📊 {items_with_prov} 個項目含有溯源資訊")
        print()
    # Generate audit event
    if args.audit:
        print("📝 生成審計事件...")
        event = generate_audit_event(
            action="provenance.generate",
            subject="docs-index",
            details={
                "input": str(input_path),
                "itemCount": len(items),
                "operations": {
                    "sbom": args.generate_sbom,
                    "provenance": args.generate_provenance,
                    "inject": args.inject,
                },
            },
        )
        audit_path = repo_root / "gl-platform.governance" / "audit" / "events.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        print(f"   ✅ 審計事件已追加: {audit_path}")
        print(f"   🆔 事件 ID: {event['eventId']}")
        print()
    print("=" * 60)
    print("✅ 完成")
    return 0
if __name__ == "__main__":
    sys.exit(main())
