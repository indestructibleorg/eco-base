# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate_vectors
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Test Vectors Validation Tool
This script validates test vectors against their JSON schemas with SLSA provenance support.
"""
# MNGA-002: Import organization needs review
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
try:
    from jsonschema import ValidationError, validate
except ImportError:
    print("⚠️  Warning: jsonschema not installed. Install with: pip install jsonschema")
    sys.exit(1)
def load_data_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON or YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Try to determine file type by extension
            if file_path.suffix in [".yaml", ".yml"]:
                if not HAS_YAML:
                    raise ValueError("YAML support requires PyYAML: pip install pyyaml")
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load file: {e}")
def load_schema(schema_path: Path) -> Dict[str, Any]:
    """Load JSON schema file."""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load schema: {e}")
def validate_vector(vector_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate a test vector against a schema."""
    try:
        validate(instance=vector_data, schema=schema)
        return True
    except ValidationError as e:
        print(f"❌ Validation error: {e.message}", file=sys.stderr)
        if e.path:
            print(f"   Path: {' -> '.join(str(p) for p in e.path)}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return False
def generate_slsa_evidence(
    file_path: Path, validation_result: bool, evidence_dir: Optional[Path]
) -> None:
    """Generate SLSA-format evidence for validation."""
    if not evidence_dir:
        return
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "type": "test-vector-validation",
        "subject": str(file_path),
        "result": "passed" if validation_result else "failed",
        "timestamp": None,  # Would be set by actual implementation
        "validator": "validate_vectors.py",
        "version": "1.0.0",
    }
    output_file = evidence_dir / f"{file_path.stem}-validation.json"
    with open(output_file, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"📊 Evidence written to: {output_file}")
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate test vectors against JSON schema"
    )
    parser.add_argument("file_path", type=Path, help="Path to the test vector file")
    parser.add_argument(
        "--schema", type=Path, required=True, help="Path to JSON schema file"
    )
    parser.add_argument(
        "--output-format",
        choices=["slsa", "json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--evidence-dir", type=Path, help="Directory to store validation evidence"
    )
    args = parser.parse_args()
    # Load files
    try:
        vector_data = load_data_file(args.file_path)
        schema = load_schema(args.schema)
    except ValueError as e:
        print(f"❌ Error loading files: {e}", file=sys.stderr)
        sys.exit(1)
    # Validate
    print(f"🔍 Validating: {args.file_path}")
    print(f"📋 Schema: {args.schema}")
    print("-" * 60)
    is_valid = validate_vector(vector_data, schema)
    if is_valid:
        print("✅ Validation passed!")
    else:
        print("❌ Validation failed!")
    # Generate evidence if requested
    if args.output_format == "slsa" or args.evidence_dir:
        generate_slsa_evidence(args.file_path, is_valid, args.evidence_dir)
    sys.exit(0 if is_valid else 1)
if __name__ == "__main__":
    main()
