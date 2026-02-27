# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: gl-platform.governance_agent
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Machine-Native Governance Agent
A machine-executable agent that reads gl-platform.governance-manifest.yaml and performs automated gl-platform.governance tasks.
"""
# MNGA-002: Import organization needs review
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import jsonschema
import yaml
class GovernanceAgent:
    """
    Machine-Native Governance Agent
    Reads gl-platform.governance-manifest.yaml and provides machine-executable gl-platform.governance functions.
    """
    def __init__(self, manifest_path: str = "gl-platform.governance-manifest.yaml"):
        """Initialize the gl-platform.governance agent."""
        self.manifest_path = Path(manifest_path)
        self.manifest = None
        self.schemas = {}
        self.policies = {}
        self._load_manifest()
        self._load_schemas()
    def _load_manifest(self):
        """Load the gl-platform.governance manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Governance manifest not found: {self.manifest_path}"
            )
        with open(self.manifest_path, "r", encoding='utf-8') as f:
            self.manifest = yaml.safe_load(f)
        print(f"[INFO] Loaded gl-platform.governance manifest: {self.manifest['metadata']['name']}")
        print(f"[INFO] Version: {self.manifest['metadata']['version']}")
    def _load_schemas(self):
        """Load all schemas defined in the manifest."""
        schema_dir = Path("schemas")
        if schema_dir.exists():
            for schema_file in schema_dir.glob("*.yaml"):
                with open(schema_file, "r", encoding='utf-8') as f:
                    schema = yaml.safe_load(f)
                    # Extract schema name without ".schema" suffix
                    schema_name = schema_file.stem.replace(".schema", "")
                    self.schemas[schema_name] = schema
                    print(f"[INFO] Loaded schema: {schema_name}")
    def validate_request(
        self, request_data: Dict[str, Any], request_type: str = "validation"
    ) -> Dict[str, Any]:
        """
        Validate a request against its schema.
        Args:
            request_data: The request data to validate
            request_type: Type of request (validation, generation, change, exception, report)
        Returns:
            Validation result with valid flag and any errors
        """
        schema_name = f"{request_type}-request"
        if schema_name not in self.schemas:
            return {
                "valid": False,
                "error": f"Schema not found: {schema_name}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        try:
            jsonschema.validate(instance=request_data, schema=self.schemas[schema_name])
            return {
                "valid": True,
                "message": "Request is valid",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except jsonschema.ValidationError as e:
            return {
                "valid": False,
                "error": str(e),
                "path": list(e.path) if e.path else [],
                "timestamp": datetime.utcnow().isoformat(),
            }
    def validate_name(
        self, name: str, resource_type: str, environment: str, **context
    ) -> Dict[str, Any]:
        """
        Validate a resource name against gl-platform.governance policies.
        Args:
            name: The name to validate
            resource_type: Type of resource
            environment: Deployment environment
#*context: Additional context (team, service, etc.)
        Returns:
            Validation response
        """
        # Create request
        request = {
            "resource_name": name,
            "resource_type": resource_type,
            "environment": environment,
#*context,
        }
        # Validate against schema
        schema_validation = self.validate_request(request, "validation")
        if not schema_validation["valid"]:
            return {
                "valid": False,
                "resource_name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "violations": [
                    {
                        "severity": "critical",
                        "code": "SCHEMA_VALIDATION_FAILED",
                        "message": schema_validation["error"],
                    }
                ],
            }
        # Basic format validation
        violations = []
        # Check length
        if len(name) < 3:
            violations.append(
                {
                    "severity": "critical",
                    "code": "NAME_TOO_SHORT",
                    "message": "Name must be at least 3 characters",
                    "suggestion": f"{name}-{environment}",
                }
            )
        elif len(name) > 63:
            violations.append(
                {
                    "severity": "critical",
                    "code": "NAME_TOO_LONG",
                    "message": "Name must be at most 63 characters",
                }
            )
        # Check pattern
        pattern = r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$"
        if not re.match(pattern, name):
            violations.append(
                {
                    "severity": "critical",
                    "code": "INVALID_PATTERN",
                    "message": "Name must contain only lowercase alphanumeric characters and hyphens",
                    "message": "Name must not start or end with a hyphen",
                })
        # Check environment prefix
        if not name.startswith(environment):
            violations.append({"severity": "high",
                               "code": "MISSING_ENVIRONMENT_PREFIX",
                               "message": f"Name should start with environment prefix: {environment}",
                               "suggestion": f"{environment}-{name}",
                               })
        # Load naming policy for detailed validation
        policy_path = Path(
            "workspace/src/gl-platform.governance/10-policy/naming-gl-platform.governance-policy.yaml"
        )
        if policy_path.exists():
            try:
                with open(policy_path, "r", encoding='utf-8') as f:
                    yaml.safe_load(f)
                    # Add policy-specific validation here
                    # This is a simplified version
            except Exception:
                # If policy file has syntax errors, skip policy-specific
                # validation
                pass
        return {
            "valid": len(violations) == 0,
            "resource_name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "violations": violations,
            "suggestions": (
                self._generate_suggestions(name, resource_type, environment)
                if violations
                else []
            ),
        }
    def _generate_suggestions(
        self, name: str, resource_type: str, environment: str
    ) -> List[str]:
        """Generate suggested valid names."""
        suggestions = []
        # Normalize the name
        normalized = re.sub(r"[^a-z0-9-]", "-", name.lower())
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        # Generate variations
        suggestions.append(f"{environment}-{normalized}")
        if "team" in name or "service" in name:
            suggestions.append(
                f"{environment}-platform-{resource_type.replace('k8s-', '')}"
            )
        return suggestions[:3]
    def generate_name(
        self,
        resource_type: str,
        environment: str,
        team: str = None,
        service: str = None,
        version: str = None,
#*options,
    ) -> Dict[str, Any]:
        """
        Generate a compliant resource name.
        Args:
            resource_type: Type of resource
            environment: Deployment environment
            team: Team name
            service: Service name
            version: Version identifier
#*options: Additional generation options
        Returns:
            Generated name response
        """
        # Create request
        request = {"resource_type": resource_type, "environment": environment}
        if team:
            request["team"] = team
        if service:
            request["service"] = service
        if version:
            request["version"] = version
        if options:
            request["options"] = options
        # Validate request
        schema_validation = self.validate_request(request, "generation")
        if not schema_validation["valid"]:
            return {
                "success": False,
                "error": schema_validation["error"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        # Load naming patterns
        patterns_path = Path("workspace/src/gl-platform.governance/35-scripts/naming-patterns.yaml")
        if patterns_path.exists():
            with open(patterns_path, "r", encoding='utf-8') as f:
                patterns_data = yaml.safe_load(f)
                patterns_data.get("patterns", {})
        # Get resource type suffix
        type_suffixes = {
            "k8s-deployment": "deploy",
            "k8s-service": "svc",
            "k8s-ingress": "ing",
            "k8s-configmap": "cm",
            "k8s-secret": "secret",
            "k8s-pvc": "pvc",
            "k8s-pv": "pv",
            "aws-s3-bucket": "s3",
            "aws-lambda": "lambda",
            "azure-storage-account": "st",
            "gcp-storage-bucket": "gcs",
        }
        # Build name components
        components = [environment]
        if team:
            components.append(team)
        elif service:
            # Use first part of service name as team
            components.append(service.split("-")[0])
        if service:
            components.append(service)
        type_suffix = type_suffixes.get(resource_type, resource_type.replace("-", "-"))
        components.append(type_suffix)
        if version:
            # Remove 'v' prefix if present
            clean_version = version.lstrip("v")
            components.append(clean_version)
        elif options.get("include_timestamp"):
            components.append(str(int(datetime.utcnow().timestamp())))
        # Join components
        generated_name = "-".join(components)
        # Ensure name doesn't exceed 63 characters
        if len(generated_name) > 63:
            # Truncate service name if too long
            service_idx = 2 if team else 1
            excess = len(generated_name) - 63
            components[service_idx] = components[service_idx][:-excess]
            generated_name = "-".join(components)
        return {
            "success": True,
            "generated_name": generated_name,
            "resource_type": resource_type,
            "environment": environment,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "components": components,
                "pattern": f"{environment}-{'{team}-' if team else ''}{service}-{type_suffix}{'-{version}' if version else ''}",
            },
        }
    def create_change_request(self, change_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a change request.
        Args:
            change_data: Change request data
        Returns:
            Created change request with ID
        """
        # Validate request
        schema_validation = self.validate_request(change_data, "change")
        if not schema_validation["valid"]:
            return {
                "success": False,
                "error": schema_validation["error"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        # Generate change ID if not provided
        if "change_id" not in change_data:
            year = datetime.utcnow().year
            # In production, this would be a sequential number
            change_data["change_id"] = f"CHG-{year}-001"
        # Set default values
        change_data.setdefault("approval", {}).setdefault("status", "pending")
        change_data.setdefault("approval", {}).setdefault("method", "auto")
        return {
            "success": True,
            "change_request": change_data,
            "timestamp": datetime.utcnow().isoformat(),
            "next_steps": [
                "Submit change request for approval",
                "Implement according to plan",
                "Execute rollback if needed",
            ],
        }
    def create_exception_request(
        self, exception_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create an exception request.
        Args:
            exception_data: Exception request data
        Returns:
            Created exception request with ID
        """
        # Validate request
        schema_validation = self.validate_request(exception_data, "exception")
        if not schema_validation["valid"]:
            return {
                "success": False,
                "error": schema_validation["error"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        # Generate exception ID if not provided
        if "exception_id" not in exception_data:
            year = datetime.utcnow().year
            exception_data["exception_id"] = f"EXC-{year}-001"
        # Set default values
        exception_data.setdefault("approval", {}).setdefault("status", "under_review")
        return {
            "success": True,
            "exception_request": exception_data,
            "timestamp": datetime.utcnow().isoformat(),
            "next_steps": [
                "Submit exception request for review",
                "Provide additional documentation if requested",
                "Implement mitigation measures",
            ],
        }
    def get_manifest_info(self) -> Dict[str, Any]:
        """Get information about the gl-platform.governance manifest."""
        return {
            "name": self.manifest["metadata"]["name"],
            "version": self.manifest["metadata"]["version"],
            "owner": self.manifest["metadata"]["owner"],
            "modules": len(self.manifest["spec"]["modules"]),
            "api_version": self.manifest["apiVersion"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all gl-platform.governance modules."""
        return [
            {
                "id": module["id"],
                "name": module["name"],
                "status": module["status"],
                "location": module.get("location"),
                "functions": module.get("functions", []),
            }
            for module in self.manifest["spec"]["modules"]
        ]
    def get_module_info(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific module."""
        for module in self.manifest["spec"]["modules"]:
            if module["id"] == module_id:
                return module
        return None
def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python gl-platform.governance_agent.py <command> [args]")
        print("Commands:")
        print("  validate <name> <type> <env>")
        print("  generate <type> <env> [team] [service] [version]")
        print("  info")
        print("  modules")
        sys.exit(1)
    command = sys.argv[1]
    agent = GovernanceAgent()
    if command == "validate":
        if len(sys.argv) < 4:
            print("Usage: python gl-platform.governance_agent.py validate <name> <type> <env>")
            sys.exit(1)
        name = sys.argv[2]
        resource_type = sys.argv[3]
        environment = sys.argv[4]
        result = agent.validate_name(name, resource_type, environment)
        print(json.dumps(result, indent=2))
    elif command == "generate":
        if len(sys.argv) < 4:
            print(
                "Usage: python gl-platform.governance_agent.py generate <type> <env> [team] [service] [version]"
            )
            sys.exit(1)
        resource_type = sys.argv[2]
        environment = sys.argv[3]
        team = sys.argv[4] if len(sys.argv) > 4 else None
        service = sys.argv[5] if len(sys.argv) > 5 else None
        version = sys.argv[6] if len(sys.argv) > 6 else None
        result = agent.generate_name(resource_type, environment, team, service, version)
        print(json.dumps(result, indent=2))
    elif command == "info":
        info = agent.get_manifest_info()
        print(json.dumps(info, indent=2))
    elif command == "modules":
        modules = agent.list_modules()
        print(json.dumps(modules, indent=2))
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
if __name__ == "__main__":
    main()
