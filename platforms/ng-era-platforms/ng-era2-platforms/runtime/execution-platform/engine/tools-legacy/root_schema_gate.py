# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: root_schema_gate
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
# tools/root_schema_gate.py
# Phase 1：驗證 root/registry/modules.yaml 是否符合 root/schemas/modules.schema.json
"""
Module docstring
================

This module is part of the GL governance framework.
Please add specific module documentation here.
"""
# MNGA-002: Import organization needs review
import json
import sys
from pathlib import Path
import jsonschema
import yaml
def main():
    repo = Path(".")
    schema_path = repo / "root/schemas/modules.schema.json"
    registry_path = repo / "root/registry/root.registry.modules.yaml"
    report_path = repo / "dist/reports/root-schema.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    report = {
        "gate": "gate-root-schema",
        "schema": str(schema_path),
        "target": str(registry_path),
        "result": "pass",
        "errors": [],
    }
    try:
        jsonschema.validate(instance=registry, schema=schema)
    except jsonschema.ValidationError as e:
        report["result"] = "fail"
        report["errors"].append(str(e))
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["result"] == "pass" else 2
if __name__ == "__main__":
    sys.exit(main())
