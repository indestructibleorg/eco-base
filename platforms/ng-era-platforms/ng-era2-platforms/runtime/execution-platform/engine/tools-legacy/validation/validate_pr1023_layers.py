# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate_pr1023_layers
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Validate PR #1023 layer coverage (L4–L8).
Checks the presence of key artifacts introduced in PR #1023:
 - L4: monitor modules and quantum tests
 - L5: quantum dashboard UI source
 - L6: Kubernetes manifests for quantum stack and validation system
 - L7: validation tools and evidence chains
 - L8: five-layer quantum security configs
Exit codes:
  0 -> all checks passed
  1 -> one or more checks failed
"""
# MNGA-002: Import organization needs review
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
REPO_ROOT = Path(__file__).resolve().parents[2]
@dataclass
class Check:
    name: str
    patterns: List[str]
    min_count: int = 1
    description: str | None = None
    def run(self) -> dict:
        matches: List[str] = []
        for pattern in self.patterns:
            matches.extend(str(p) for p in REPO_ROOT.glob(pattern) if p.exists())
        return {
            "name": self.name,
            "patterns": self.patterns,
            "found": matches,
            "passed": len(matches) >= self.min_count,
            "expected_min": self.min_count,
            "description": self.description or "",
        }
def build_checks(evidence_min: int) -> Iterable[Check]:
    return [
        Check(
            name="L4 monitor modules",
            patterns=["workspace/src/quantum/monitor/*.py"],
            description="監控/效能模組應存在且可讀取",
        ),
        Check(
            name="L4 quantum tests",
            patterns=["workspace/tests/quantum/*.py"],
            description="量子測試覆蓋需存在",
        ),
        Check(
            name="L5 quantum dashboard",
            patterns=["workspace/tools/apps/quantum-dashboard/src/*"],
            description="量子前端 UI 與指標板原始碼",
        ),
        Check(
            name="L6 k8s quantum stack",
            patterns=["workspace/tools/infrastructure/kubernetes/quantum/*.yaml"],
            description="量子堆疊 K8s 組件 (backend/frontend/HPA/ingress/secret)",
        ),
        Check(
            name="L6 k8s validation system",
            patterns=["workspace/tools/infrastructure/kubernetes/validation/*.yaml"],
            description="驗證系統 K8s 部署",
        ),
        Check(
            name="L7 validation tools",
            patterns=["tools/validation/*"],
            description="量子增強驗證腳本與配置",
        ),
        Check(
            name="L7 evidence chains",
            patterns=["workspace/docs/validation/evidence-chains/EV-*.json"],
            min_count=evidence_min,
            description=f"{evidence_min} 項以上證據鏈 (EV-*)",
        ),
        Check(
            name="L8 security configs",
            patterns=["workspace/tools/security/*.yaml"],
            description="五層量子安全策略",
        ),
    ]
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PR #1023 layers (L4–L8).")
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print JSON output (machine-readable).",
    )
    parser.add_argument(
        "--evidence-min",
        type=int,
        default=23,
        help="Minimum expected EV-* evidence chain files (default: 23).",
    )
    args = parser.parse_args()
    results = [check.run() for check in build_checks(args.evidence_min)]
    all_passed = all(r["passed"] for r in results)
    if args.json_output:
        print(
            json.dumps(
                {"passed": all_passed, "results": results}, indent=2, ensure_ascii=False
            )
        )
    else:
        for result in results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {result['name']} (expected ≥{result['expected_min']})")
            if result["description"]:
                print(f"      {result['description']}")
            print(f"      patterns: {', '.join(result['patterns'])}")
            print(f"      found: {len(result['found'])} item(s)")
        print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1
if __name__ == "__main__":
    raise SystemExit(main())
