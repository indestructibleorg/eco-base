# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: fix-yaml-structure
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Fix multi-document YAML files by separating schemas from main content
"""
from pathlib import Path
import yaml
def fix_multi_document_yaml(file_path: Path):
    """Fix multi-document YAML by extracting first document only"""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            documents = list(yaml.safe_load_all(f))
        if len(documents) <= 1:
            print(f"✅ {file_path.name}: Already single document")
            return True
        # Extract first document (main content)
        main_doc = documents[0]
        # Create backup
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        with open(backup_path, "w", encoding='utf-8') as f:
            f.write(file_path.read_text())
        # Write back only first document
        with open(file_path, "w", encoding='utf-8') as f:
            yaml.dump(main_doc, f, default_flow_style=False, sort_keys=False)
        # Save additional documents as schema files if needed
        if len(documents) > 1:
            schemas_dir = file_path.parent / "schemas"
            schemas_dir.mkdir(exist_ok=True)
            for i, doc in enumerate(documents[1:], 1):
                schema_file = schemas_dir / f"{file_path.stem}.schema{i}.yaml"
                with open(schema_file, "w", encoding='utf-8') as f:
                    yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
        print(
            f"🔧 {file_path.name}: Fixed - {len(documents)} docs -> 1 main + {len(documents)-1} schemas"
        )
        return True
    except Exception as e:
        print(f"❌ {file_path.name}: Error - {str(e)}")
        return False
def fix_metadata_issues(file_path: Path):
    """Fix missing metadata fields"""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            return False
        # Fix missing apiVersion
        if "apiVersion" not in content:
            if "kind" in content:
                content["apiVersion"] = "machinenativeops.io/v1"
            else:
                content["apiVersion"] = "v1"
        # Fix metadata
        if "metadata" not in content:
            content["metadata"] = {}
        metadata = content["metadata"]
        # Add missing required fields
        if "name" not in metadata:
            metadata["name"] = file_path.stem
        if "version" not in metadata:
            metadata["version"] = "v1.0.0"
        if "urn" not in metadata:
            kind = content.get("kind", "config").lower().replace(" ", "-")
            metadata["urn"] = f"urn:machinenativeops:root:{kind}:v1"
        # Write back
        with open(file_path, "w", encoding='utf-8') as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        print(f"📝 {file_path.name}: Fixed metadata")
        return True
    except Exception as e:
        print(f"❌ {file_path.name}: Metadata fix error - {str(e)}")
        return False
def main():
    root_dir = Path(".")
    # Files to fix
    multi_doc_files = [
        "root.config.yaml",
        "root.modules.yaml",
        "root.super-execution.yaml",
        "root.trust.yaml",
        "root.provenance.yaml",
        "root.integrity.yaml",
        "root.bootstrap.yaml",
    ]
    metadata_fix_files = [
        "root.naming-policy.yaml",
        "root.specs.context.yaml",
        "root.specs.logic.yaml",
        "root.specs.mapping.yaml",
        "root.specs.naming.yaml",
        "root.specs.references.yaml",
        "root.registry.modules.yaml",
        "root.registry.urns.yaml",
    ]
    print("🔧 Fixing multi-document YAML files...")
    for file_name in multi_doc_files:
        file_path = root_dir / file_name
        if file_path.exists():
            fix_multi_document_yaml(file_path)
    print("\n📝 Fixing metadata issues...")
    for file_name in metadata_fix_files:
        file_path = root_dir / file_name
        if file_path.exists():
            fix_metadata_issues(file_path)
    print("\n✅ YAML structure fixes completed!")
if __name__ == "__main__":
    main()
