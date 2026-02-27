#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: validate-module-manifests
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# ECO-Layer: GL30-49 (Execution)
#!/usr/bin/env python3
"""
Module Manifest Validator
Validates module manifests against JSON schema
"""
# MNGA-002: Import organization needs review
import json
import yaml
import sys
from pathlib import Path
from jsonschema import validate, ValidationError
def main():
    # Load schema
    schema_path = Path("controlplane/baseline/modules/module-manifest.schema.json")
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    # Validate each module manifest
    modules_dir = Path("controlplane/baseline/modules")
    failed = False
    validated = 0
    for module_dir in sorted(modules_dir.glob("[0-9][0-9]-*")):
        if module_dir.is_dir():
            manifest_path = module_dir / "module-manifest.yaml"
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = yaml.safe_load(f)
                try:
                    validate(instance=manifest, schema=schema)
                    print(f"✅ {module_dir.name}: Valid")
                    validated += 1
                except ValidationError as e:
                    print(f"❌ {module_dir.name}: {e.message}")
                    failed = True
    print(f"\nValidated {validated} module manifest(s)")
    if failed:
        sys.exit(1)
    else:
        print("✅ All module manifests are valid!")
        sys.exit(0)
if __name__ == "__main__":
    main()
