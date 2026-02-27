# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: baseline_validation_engine
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
SynergyMesh Baseline Validation Engine
======================================
Purpose: Validate baseline configurations and system health
Extracted from legacy validate-all-baselines.v1.0.sh and adapted for SynergyMesh
"""
# MNGA-002: Import organization needs review
import json
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
class Status:
    """Status constants for validation results"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
STATUS_EMOJI = {Status.PASS: "✅", Status.FAIL: "❌", Status.WARN: "⚠️"}
def get_status_emoji(status: str) -> str:
    """Get emoji for status"""
    return STATUS_EMOJI.get(status, "")
@dataclass
class ValidationResult:
    """Data class for validation results with auto-evolution support"""
    check_name: str
    status: str
    message: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    auto_fix_suggestion: str = ""
    remediation_command: str = ""
class BaselineValidationEngine:
    """Engine for validating SynergyMesh baseline configurations"""
    @staticmethod
    def validate_namespace_name(namespace: str) -> bool:
        """Validate namespace name matches RFC 1123 label"""
        # Kubernetes namespace must match RFC 1123 label: lowercase
        # alphanumerics and '-', 1-63 chars
        return bool(re.fullmatch(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?", namespace))
    def __init__(self, namespace: str = "machinenativenops-system"):
        self.namespace = namespace
        self.validation_results: list[ValidationResult] = []
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        self.log_file = f"/tmp/baseline-validation-{timestamp_str}-{unique_id}.log"
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except IOError as e:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Failed to write log: {e}",
                file=sys.stderr,
            )
    def add_result(
        self,
        check_name: str,
        status: str,
        message: str,
        details: dict = None,
        auto_fix_suggestion: str = "",
        remediation_command: str = "",
    ):
        """Add validation result with auto-evolution metadata"""
        result = ValidationResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {},
            auto_fix_suggestion=auto_fix_suggestion,
            remediation_command=remediation_command,
        )
        self.validation_results.append(result)
        emoji = get_status_emoji(status)
        self.log(f"{emoji} {check_name}: {message}", level=status)
        if auto_fix_suggestion:
            self.log(f"  💡 Auto-fix suggestion: {auto_fix_suggestion}", level="INFO")
    def run_kubectl(self, args: list[str]) -> tuple[bool, str]:
        """Execute kubectl command after validating subcommand"""
        # Allowlist of safe kubectl subcommands
        allowed_subcommands = {
            "get",
            "describe",
            "apply",
            "delete",
            "logs",
            "exec",
            "create",
            "edit",
            "replace",
            "patch",
            "scale",
            "cordon",
            "uncordon",
            "drain",
            "rollout",
            "top",
            "expose",
            "run",
            "explain",
            "version",
            "cluster-info",
            "api-resources",
            "api-versions",
            "config",
            "label",
            "annotate",
            "attach",
            "port-forward",
            "cp",
            "auth",
        }
        if not args or args[0] not in allowed_subcommands:
            error_msg = f"Disallowed or missing kubectl subcommand: {args[0] if args else '(none)'}"
            self.log(error_msg, level="FAIL")
            return False, error_msg
        try:
            result = subprocess.run(
                ["kubectl"] + args, capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0, result.stdout
        except subprocess.TimeoutExpired as e:
            return False, f"Timeout: {str(e)}"
        except subprocess.CalledProcessError as e:
            return False, f"Process error: {str(e)}"
        except Exception as e:
            return False, str(e)
    def check_prerequisites(self) -> bool:
        """Check if required tools are available"""
        self.log("Checking prerequisites...")
        # Check kubectl
        success, _ = self.run_kubectl(["version", "--client"])
        if not success:
            self.add_result(
                "prerequisites",
                Status.FAIL,
                "kubectl not found",
                auto_fix_suggestion="Install kubectl: https://kubernetes.io/docs/tasks/tools/",
                remediation_command="curl -LO https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl",
            )
            return False
        # Check cluster connectivity
        success, _ = self.run_kubectl(["cluster-info"])
        if not success:
            self.add_result(
                "prerequisites",
                Status.FAIL,
                "Cannot connect to Kubernetes cluster",
                auto_fix_suggestion="Verify kubeconfig and cluster accessibility",
                remediation_command="kubectl config view && kubectl config current-context",
            )
            return False
        self.add_result("prerequisites", Status.PASS, "All prerequisites met")
        return True
    def validate_namespace(self) -> bool:
        """Validate namespace configuration"""
        self.log(f"Validating namespace: {self.namespace}")
        # Check namespace exists
        success, output = self.run_kubectl(["get", "namespace", self.namespace])
        if not success:
            self.add_result(
                "namespace_exists",
                Status.FAIL,
                f"Namespace not found: {self.namespace}",
                auto_fix_suggestion=f"Create namespace: kubectl create namespace {self.namespace}",
                remediation_command=f"kubectl create namespace {self.namespace}",
            )
            return False
        self.add_result(
            "namespace_exists", Status.PASS, f"Namespace exists: {self.namespace}"
        )
        # Check namespace labels
        success, output = self.run_kubectl(
            ["get", "namespace", self.namespace, "-o", "jsonpath={.metadata.labels}"]
        )
        if success:
            try:
                labels = json.loads(output) if output else {}
                required_labels = ["app.kubernetes.io/name"]
                missing_labels = [
                    label for label in required_labels if label not in labels
                ]
                if missing_labels:
                    self.add_result(
                        "namespace_labels",
                        Status.WARN,
                        f"Missing recommended labels: {missing_labels}",
                        {"labels": labels},
                        auto_fix_suggestion=f"Add labels to namespace {self.namespace}",
                        remediation_command=f"kubectl label namespace {self.namespace} app.kubernetes.io/name={self.namespace}",
                    )
                else:
                    self.add_result(
                        "namespace_labels", Status.PASS, "All required labels present"
                    )
            except json.JSONDecodeError:
                self.add_result(
                    "namespace_labels", Status.WARN, "Could not parse namespace labels"
                )
        return True
    def validate_configmaps(self) -> bool:
        """Validate ConfigMaps in namespace"""
        self.log("Validating ConfigMaps...")
        success, output = self.run_kubectl(
            ["get", "configmap", "-n", self.namespace, "-o", "json"]
        )
        if not success:
            self.add_result("configmaps", Status.FAIL, "Could not retrieve ConfigMaps")
            return False
        try:
            data = json.loads(output)
            configmaps = data.get("items", [])
            if len(configmaps) == 0:
                self.add_result("configmaps", Status.WARN, "No ConfigMaps found")
            else:
                self.add_result(
                    "configmaps",
                    Status.PASS,
                    f"Found {len(configmaps)} ConfigMaps",
                    {"count": len(configmaps)},
                )
            return True
        except json.JSONDecodeError:
            self.add_result("configmaps", Status.FAIL, "Could not parse ConfigMap data")
            return False
    def validate_deployments(self) -> bool:
        """Validate Deployments in namespace"""
        self.log("Validating Deployments...")
        success, output = self.run_kubectl(
            ["get", "deployment", "-n", self.namespace, "-o", "json"]
        )
        if not success:
            self.add_result(
                "deployments", Status.WARN, "Could not retrieve Deployments"
            )
            return True
        try:
            data = json.loads(output)
            deployments = data.get("items", [])
            if len(deployments) == 0:
                self.add_result("deployments", Status.WARN, "No Deployments found")
                return True
            ready_deployments = 0
            for deployment in deployments:
                status = deployment.get("status", {})
                ready_replicas = status.get("readyReplicas", 0)
                replicas = status.get("replicas", 0)
                if ready_replicas == replicas and replicas > 0:
                    ready_deployments += 1
            if ready_deployments == len(deployments):
                self.add_result(
                    "deployments",
                    Status.PASS,
                    f"All {len(deployments)} Deployments are ready",
                )
            else:
                self.add_result(
                    "deployments",
                    Status.WARN,
                    f"Only {ready_deployments}/{len(deployments)} Deployments are ready",
                )
            return True
        except json.JSONDecodeError:
            self.add_result(
                "deployments", Status.FAIL, "Could not parse Deployment data"
            )
            return False
    def validate_services(self) -> bool:
        """Validate Services in namespace"""
        self.log("Validating Services...")
        success, output = self.run_kubectl(
            ["get", "service", "-n", self.namespace, "-o", "json"]
        )
        if not success:
            self.add_result("services", Status.WARN, "Could not retrieve Services")
            return True
        try:
            data = json.loads(output)
            services = data.get("items", [])
            if len(services) == 0:
                self.add_result("services", Status.WARN, "No Services found")
            else:
                self.add_result(
                    "services",
                    Status.PASS,
                    f"Found {len(services)} Services",
                    {"count": len(services)},
                )
            return True
        except json.JSONDecodeError:
            self.add_result("services", Status.FAIL, "Could not parse Service data")
            return False
    def validate_network_policies(self) -> bool:
        """Validate NetworkPolicies in namespace"""
        self.log("Validating NetworkPolicies...")
        success, output = self.run_kubectl(
            ["get", "networkpolicy", "-n", self.namespace, "-o", "json"]
        )
        if not success:
            self.add_result(
                "network_policies", Status.WARN, "Could not retrieve NetworkPolicies"
            )
            return True
        try:
            data = json.loads(output)
            policies = data.get("items", [])
            if len(policies) == 0:
                self.add_result(
                    "network_policies", Status.WARN, "No NetworkPolicies found"
                )
            else:
                self.add_result(
                    "network_policies",
                    Status.PASS,
                    f"Found {len(policies)} NetworkPolicies",
                )
            return True
        except json.JSONDecodeError:
            self.add_result(
                "network_policies", Status.FAIL, "Could not parse NetworkPolicy data"
            )
            return False
    def generate_report(self) -> dict:
        """Generate validation report with auto-evolution insights"""
        total_checks = len(self.validation_results)
        passed_checks = sum(
            1 for r in self.validation_results if r.status == Status.PASS
        )
        failed_checks = sum(
            1 for r in self.validation_results if r.status == Status.FAIL
        )
        warned_checks = sum(
            1 for r in self.validation_results if r.status == Status.WARN
        )
        # Collect auto-remediation suggestions
        remediation_plan = [
            {
                "check": r.check_name,
                "issue": r.message,
                "suggestion": r.auto_fix_suggestion,
                "command": r.remediation_command,
            }
            for r in self.validation_results
            if r.status in (Status.FAIL, Status.WARN) and r.auto_fix_suggestion
        ]
        report = {
            "timestamp": datetime.now().isoformat(),
            "namespace": self.namespace,
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
            "warnings": warned_checks,
            "health_score": round(
                (passed_checks / total_checks * 100) if total_checks > 0 else 0, 2
            ),
            "auto_evolution": {
                "remediations_available": len(remediation_plan),
                "remediation_plan": remediation_plan,
            },
            "results": [
                {
                    "check": r.check_name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp,
                    "auto_fix_suggestion": r.auto_fix_suggestion,
                    "remediation_command": r.remediation_command,
                }
                for r in self.validation_results
            ],
        }
        return report
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        self.log("=" * 60)
        self.log("SynergyMesh Baseline Validation")
        self.log("=" * 60)
        self.log(f"Namespace: {self.namespace}")
        self.log(f"Log File: {self.log_file}")
        self.log("")
        if not self.check_prerequisites():
            return False
        # Run all validations
        self.validate_namespace()
        self.validate_configmaps()
        self.validate_deployments()
        self.validate_services()
        self.validate_network_policies()
        # Generate and display report
        report = self.generate_report()
        self.log("")
        self.log("=" * 60)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 60)
        self.log(f"Total Checks: {report['total_checks']}")
        self.log(f"✅ Passed: {report['passed']}")
        self.log(f"❌ Failed: {report['failed']}")
        self.log(f"⚠️  Warnings: {report['warnings']}")
        self.log(f"🎯 Health Score: {report['health_score']}%")
        self.log("")
        # Display auto-evolution remediation plan
        if report["auto_evolution"]["remediations_available"] > 0:
            self.log("=" * 60)
            self.log("🔧 AUTO-EVOLUTION REMEDIATION PLAN")
            self.log("=" * 60)
            self.log(
                f"💡 {report['auto_evolution']['remediations_available']} auto-fix suggestion(s) available:"
            )
            self.log("")
            for idx, remediation in enumerate(
                report["auto_evolution"]["remediation_plan"], 1
            ):
                self.log(f"{idx}. {remediation['check']}")
                self.log(f"   Issue: {remediation['issue']}")
                self.log(f"   💡 Suggestion: {remediation['suggestion']}")
                if remediation["command"]:
                    self.log(f"   🔧 Command: {remediation['command']}")
                self.log("")
            self.log("=" * 60)
            self.log("")
        # Save JSON report
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        json_file = f"/tmp/baseline-validation-{timestamp_str}-{unique_id}.json"
        with open(json_file, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        self.log(f"JSON report saved to: {json_file}")
        if report["failed"] == 0:
            self.log("🎉 All validations passed!")
            return True
        else:
            self.log("⚠️  Some validations failed")
            return False
def main():
    """Main entry point with auto-evolution support"""
    import argparse
    parser = argparse.ArgumentParser(
        description="SynergyMesh Baseline Validation Engine with Auto-Evolution"
    )
    parser.add_argument(
        "--namespace",
        default="machinenativenops-system",
        help="Kubernetes namespace to validate",
    )
    parser.add_argument(
        "--auto-evolve",
        action="store_true",
        help="Enable automatic evolution mode (display remediation suggestions)",
    )
    args = parser.parse_args()
    if not BaselineValidationEngine.validate_namespace_name(args.namespace):
        print(
            f"ERROR: Invalid namespace '{args.namespace}'. Must match RFC 1123 label: "
            "[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?"
        )
        sys.exit(2)
    engine = BaselineValidationEngine(namespace=args.namespace)
    success = engine.run_all_validations()
    if args.auto_evolve and not success:
        print("\n" + "=" * 60)
        print("🤖 AUTO-EVOLUTION MODE: Remediation suggestions displayed above")
        print("   Review the remediation plan in the JSON report for details")
        print("=" * 60)
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()
