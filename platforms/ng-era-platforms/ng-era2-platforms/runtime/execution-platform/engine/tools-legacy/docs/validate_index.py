# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate_index
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Knowledge Index Validator
知識索引驗證工具
Validates the docs/knowledge_index.yaml file to ensure:
- Schema validation against docs-index.schema.json
- All referenced files exist
- Required fields are present
- IDs are unique
- Relationships reference valid documents
Usage:
    python tools/docs/validate_index.py
    python tools/docs/validate_index.py --verbose
"""
# MNGA-002: Import organization needs review
import argparse
import json
import sys
from pathlib import Path
from typing import Any
# Try to import yaml, provide helpful error if not available
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)
# Try to import jsonschema for schema validation
try:
    from jsonschema import ValidationError, validate
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
def load_index(index_path: Path) -> dict[str, Any]:
    """Load and parse the knowledge index YAML file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def load_json_schema(schema_path: Path) -> dict[str, Any]:
    """Load JSON schema file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)
def validate_against_schema(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate data against JSON schema."""
    errors = []
    if not HAS_JSONSCHEMA:
        return ["⚠️  Schema validation skipped (jsonschema not installed)"]
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
        if e.path:
            errors.append(f"  At path: {'.'.join(str(p) for p in e.path)}")
    return errors
def validate_required_fields(
    doc: dict[str, Any], doc_id: str, schema: dict[str, Any] | None = None
) -> list[str]:
    """Check that all required fields are present in a document entry.
    If schema is provided, required fields are extracted from the schema.
    Otherwise, falls back to default required fields.
    """
    # Extract required fields from schema if available
    if schema and "properties" in schema:
        items_schema = schema.get("properties", {}).get("items", {})
        if items_schema.get("type") == "array" and "items" in items_schema:
            item_schema = items_schema["items"]
            required_fields = item_schema.get("required", [])
        else:
            required_fields = [
                "id",
                "path",
                "title",
                "domain",
                "layer",
                "type",
                "tags",
                "owner",
                "status",
                "description",
            ]
    else:
        # Default required fields for backward compatibility
        required_fields = [
            "id",
            "path",
            "title",
            "domain",
            "layer",
            "type",
            "tags",
            "owner",
            "status",
            "description",
        ]
    errors = []
    for field in required_fields:
        if field not in doc:
            errors.append(f"Document '{doc_id}' missing required field: {field}")
    return errors
def validate_file_exists(doc: dict[str, Any], repo_root: Path) -> list[str]:
    """Check that the referenced file exists."""
    errors = []
    file_path = repo_root / doc["path"]
    if not file_path.exists():
        errors.append(
            f"Document '{doc['id']}' references non-existent file: {doc['path']}"
        )
    return errors
def validate_unique_ids(documents: list[dict[str, Any]]) -> list[str]:
    """Check that all document IDs are unique."""
    errors = []
    seen_ids = set()
    for doc in documents:
        doc_id = doc.get("id", "<unknown>")
        if doc_id in seen_ids:
            errors.append(f"Duplicate document ID: {doc_id}")
        seen_ids.add(doc_id)
    return errors
def validate_relationships(
    relationships: list[dict[str, Any]], doc_ids: set[str]
) -> list[str]:
    """Check that all relationships reference valid document IDs."""
    errors = []
    for rel in relationships:
        if rel.get("from") not in doc_ids:
            errors.append(
                f"Relationship references unknown document: {rel.get('from')}"
            )
        if rel.get("to") not in doc_ids:
            errors.append(f"Relationship references unknown document: {rel.get('to')}")
    return errors
def validate_domains(
    documents: list[dict[str, Any]], categories: dict[str, Any]
) -> list[str]:
    """Check that all documents use valid domain categories."""
    errors = []
    valid_domains = set(categories.keys())
    for doc in documents:
        domain = doc.get("domain")
        if domain and domain not in valid_domains:
            errors.append(f"Document '{doc.get('id')}' uses unknown domain: {domain}")
    return errors
def validate_layers(documents: list[dict[str, Any]], layers: list[str]) -> list[str]:
    """Check that all documents use valid layers."""
    errors = []
    valid_layers = set(layers)
    for doc in documents:
        layer = doc.get("layer")
        if layer and layer not in valid_layers:
            errors.append(f"Document '{doc.get('id')}' uses unknown layer: {layer}")
    return errors
def main():
    parser = argparse.ArgumentParser(description="Validate knowledge index YAML file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--skip-schema", action="store_true", help="Skip JSON schema validation"
    )
    args = parser.parse_args()
    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    index_path = repo_root / "docs" / "knowledge_index.yaml"
    schema_path = repo_root / "gl-platform.governance" / "schemas" / "docs-index.schema.json"
    if not index_path.exists():
        print(f"Error: Knowledge index not found at {index_path}")
        sys.exit(1)
    if args.verbose:
        print(f"Validating: {index_path}")
        print(f"Repo root: {repo_root}")
        print()
    # Load the index
    try:
        index = load_index(index_path)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in {index_path}")
        print(str(e))
        sys.exit(1)
    # Collect all errors
    all_errors = []
    schema = None
    # JSON Schema validation
    if not args.skip_schema and schema_path.exists():
        if args.verbose:
            print("🔍 Validating against JSON Schema...")
        try:
            schema = load_json_schema(schema_path)
            schema_errors = validate_against_schema(index, schema)
            all_errors.extend(schema_errors)
            if args.verbose and not schema_errors:
                print("  ✅ Schema validation passed")
        except Exception as e:
            all_errors.append(f"Failed to load schema: {e}")
    elif args.verbose:
        print("⚠️  Schema validation skipped")
    documents = index.get("items", [])
    categories = index.get("categories", {})
    layers = index.get("layers", [])
    relationships = index.get("relationships", [])
    if args.verbose:
        print(f"Found {len(documents)} documents")
        print(f"Found {len(categories)} categories")
        print(f"Found {len(layers)} layers")
        print(f"Found {len(relationships)} relationships")
        print()
    # Validate unique IDs
    all_errors.extend(validate_unique_ids(documents))
    # Get all valid document IDs
    doc_ids = {doc.get("id") for doc in documents if doc.get("id")}
    # Validate each document
    for doc in documents:
        doc_id = doc.get("id", "<unknown>")
        # Check required fields (using schema if available)
        all_errors.extend(validate_required_fields(doc, doc_id, schema))
        # Check file exists
        if "path" in doc:
            file_errors = validate_file_exists(doc, repo_root)
            all_errors.extend(file_errors)
    # Validate domains
    all_errors.extend(validate_domains(documents, categories))
    # Validate layers
    all_errors.extend(validate_layers(documents, layers))
    # Validate relationships
    all_errors.extend(validate_relationships(relationships, doc_ids))
    # Output results
    if all_errors:
        print("❌ Validation FAILED")
        print()
        print(f"Found {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  • {error}")
        sys.exit(1)
    else:
        print("✅ Validation PASSED")
        print()
        print("Summary:")
        print(f"  • {len(documents)} documents validated")
        print(f"  • {len(relationships)} relationships validated")
        print("  • All referenced files exist")
        print("  • All IDs are unique")
        sys.exit(0)
if __name__ == "__main__":
    main()
