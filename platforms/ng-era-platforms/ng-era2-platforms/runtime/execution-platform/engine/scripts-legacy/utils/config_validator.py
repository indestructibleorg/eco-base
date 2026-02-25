#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: config_validator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Machine-Native Configuration Validator
Validates YAML/JSON configurations against unified schema.
"""
# MNGA-002: Import organization needs review
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
class ConfigValidator:
    """Validates configuration files against machine-native standards."""
    REQUIRED_FIELDS = ["apiVersion", "kind", "metadata"]
    METADATA_REQUIRED = ["name", "version"]
    NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
    VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(-[a-z0-9]+)?$")
    def __init__(self, strict: bool = True):
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
    def validate_file(self, file_path: str) -> Tuple[bool, List[str], List[str]]:
        """Validate a configuration file."""
        self.errors = []
        self.warnings = []
        path = Path(file_path)
        # Check file extension
        if path.suffix not in [".yaml", ".yml", ".json"]:
            self.errors.append(f"Invalid file extension: {path.suffix}")
            return False, self.errors, self.warnings
        # Check filename pattern
        if not self.NAME_PATTERN.match(path.stem.replace(".", "-").replace("_", "-")):
            self.warnings.append(f"Filename should be lowercase kebab-case: {path.name}")
        # Load and validate content
        try:
            with open(path, "r") as f:
                if path.suffix == ".json":
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to parse file: {e}")
            return False, self.errors, self.warnings
        if config is None:
            self.errors.append("Empty configuration file")
            return False, self.errors, self.warnings
        # Validate structure
        self._validate_structure(config)
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    def _validate_structure(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                if self.strict:
                    self.errors.append(f"Missing required field: {field}")
                else:
                    self.warnings.append(f"Missing recommended field: {field}")
        # Validate metadata
        if "metadata" in config:
            self._validate_metadata(config["metadata"])
        # Validate apiVersion
        if "apiVersion" in config:
            if not isinstance(config["apiVersion"], str):
                self.errors.append("apiVersion must be a string")
        # Validate kind
        if "kind" in config:
            if not isinstance(config["kind"], str):
                self.errors.append("kind must be a string")
    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata section."""
        if not isinstance(metadata, dict):
            self.errors.append("metadata must be an object")
            return
        # Check required metadata fields
        for field in self.METADATA_REQUIRED:
            if field not in metadata:
                self.errors.append(f"Missing required metadata field: {field}")
        # Validate name
        if "name" in metadata:
            name = metadata["name"]
            if not isinstance(name, str):
                self.errors.append("metadata.name must be a string")
            elif not self.NAME_PATTERN.match(name):
                self.errors.append(f"Invalid name format: {name} (must be lowercase kebab-case)")
            elif len(name) > 63:
                self.errors.append(f"Name too long: {len(name)} chars (max 63)")
        # Validate version
        if "version" in metadata:
            version = str(metadata["version"])
            if not self.VERSION_PATTERN.match(version):
                self.warnings.append(f"Version should follow semver: {version}")
    def validate_directory(self, dir_path: str, pattern: str = "*.yaml") -> Dict[str, Any]:
        """Validate all configuration files in a directory."""
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "files": []
        }
        path = Path(dir_path)
        for file_path in path.rglob(pattern):
            if ".git" in str(file_path) or "archive" in str(file_path):
                continue
            results["total"] += 1
            is_valid, errors, warnings = self.validate_file(str(file_path))
            if is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
            results["files"].append({
                "path": str(file_path),
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings
            })
        return results
def main():
    validator = ConfigValidator(strict=False)
    if len(sys.argv) > 1:
        # Validate specific file
        file_path = sys.argv[1]
        is_valid, errors, warnings = validator.validate_file(file_path)
        print(f"Validating: {file_path}")
        print(f"Valid: {is_valid}")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ❌ {e}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠️ {w}")
        sys.exit(0 if is_valid else 1)
    else:
        # Validate all configs
        results = validator.validate_directory(".", "*.yaml")
        print("=" * 60)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 60)
        print(f"Total files: {results['total']}")
        print(f"Valid: {results['valid']}")
        print(f"Invalid: {results['invalid']}")
        print("=" * 60)
        for file_info in results["files"]:
            if not file_info["valid"] or file_info["warnings"]:
                status = "✅" if file_info["valid"] else "❌"
                print(f"\n{status} {file_info['path']}")
                for e in file_info["errors"]:
                    print(f"    ❌ {e}")
                for w in file_info["warnings"]:
                    print(f"    ⚠️ {w}")
if __name__ == "__main__":
    main()