#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: health_check
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Machine-Native Health Check System
Performs comprehensive health checks on repository components.
"""
# MNGA-002: Import organization needs review
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }
class HealthChecker:
    """Performs health checks on repository components."""
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.results: List[HealthCheckResult] = []
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        self.results = [
            self._check_git_status(),
            self._check_workflows(),
            self._check_dependencies(),
            self._check_documentation(),
            self._check_security_config(),
            self._check_tests(),
        ]
        return self.results
    def _check_git_status(self) -> HealthCheckResult:
        """Check git repository status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            uncommitted = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            if uncommitted == 0:
                return HealthCheckResult(
                    name="git_status",
                    status=HealthStatus.HEALTHY,
                    message="Working directory clean",
                    details={"uncommitted_changes": 0}
                )
            else:
                return HealthCheckResult(
                    name="git_status",
                    status=HealthStatus.DEGRADED,
                    message=f"{uncommitted} uncommitted changes",
                    details={"uncommitted_changes": uncommitted}
                )
        except Exception as e:
            return HealthCheckResult(
                name="git_status",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )
    def _check_workflows(self) -> HealthCheckResult:
        """Check GitHub workflows configuration."""
        workflows_dir = self.repo_root / ".github" / "workflows"
        if not workflows_dir.exists():
            return HealthCheckResult(
                name="workflows",
                status=HealthStatus.UNHEALTHY,
                message="No workflows directory found"
            )
        workflows = list(workflows_dir.glob("*.yml"))
        required = ["policy-gate.yml", "pr-quality-check.yml"]
        missing = [r for r in required if not (workflows_dir / r).exists()]
        if not missing:
            return HealthCheckResult(
                name="workflows",
                status=HealthStatus.HEALTHY,
                message=f"{len(workflows)} workflows configured",
                details={"workflow_count": len(workflows), "workflows": [w.name for w in workflows]}
            )
        else:
            return HealthCheckResult(
                name="workflows",
                status=HealthStatus.DEGRADED,
                message=f"Missing required workflows: {missing}",
                details={"missing": missing}
            )
    def _check_dependencies(self) -> HealthCheckResult:
        """Check dependency configuration."""
        has_requirements = (self.repo_root / "requirements.txt").exists()
        has_package_json = (self.repo_root / "package.json").exists()
        has_pyproject = (self.repo_root / "pyproject.toml").exists()
        if has_requirements or has_pyproject or has_package_json:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="Dependency files found",
                details={
                    "requirements_txt": has_requirements,
                    "package_json": has_package_json,
                    "pyproject_toml": has_pyproject
                }
            )
        else:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.DEGRADED,
                message="No dependency files found"
            )
    def _check_documentation(self) -> HealthCheckResult:
        """Check documentation completeness."""
        docs = {
            "README.md": (self.repo_root / "README.md").exists(),
            "QUICKSTART.md": (self.repo_root / "QUICKSTART.md").exists(),
            "PROJECT_STATUS.md": (self.repo_root / "PROJECT_STATUS.md").exists(),
            "SECURITY.md": (self.repo_root / "SECURITY.md").exists(),
        }
        present = sum(docs.values())
        total = len(docs)
        if present == total:
            status = HealthStatus.HEALTHY
            message = "All documentation present"
        elif present >= total // 2:
            status = HealthStatus.DEGRADED
            message = f"{present}/{total} documentation files present"
        else:
            status = HealthStatus.UNHEALTHY
            message = "Missing critical documentation"
        return HealthCheckResult(
            name="documentation",
            status=status,
            message=message,
            details=docs
        )
    def _check_security_config(self) -> HealthCheckResult:
        """Check security configuration."""
        checks = {
            "security_policy": (self.repo_root / "SECURITY.md").exists(),
            "secrets_baseline": (self.repo_root / ".secrets.baseline").exists(),
            "pre_commit": (self.repo_root / ".pre-commit-config.yaml").exists(),
            "gitignore": (self.repo_root / ".gitignore").exists(),
        }
        passed = sum(checks.values())
        total = len(checks)
        if passed == total:
            status = HealthStatus.HEALTHY
            message = "All security configurations present"
        elif passed >= total // 2:
            status = HealthStatus.DEGRADED
            message = f"{passed}/{total} security configurations present"
        else:
            status = HealthStatus.UNHEALTHY
            message = "Missing critical security configurations"
        return HealthCheckResult(
            name="security_config",
            status=status,
            message=message,
            details=checks
        )
    def _check_tests(self) -> HealthCheckResult:
        """Check test configuration."""
        tests_dir = self.repo_root / "tests"
        pytest_ini = self.repo_root / "pytest.ini"
        if not tests_dir.exists():
            return HealthCheckResult(
                name="tests",
                status=HealthStatus.DEGRADED,
                message="No tests directory found"
            )
        test_files = list(tests_dir.rglob("test_*.py"))
        return HealthCheckResult(
            name="tests",
            status=HealthStatus.HEALTHY if test_files else HealthStatus.DEGRADED,
            message=f"{len(test_files)} test files found",
            details={
                "test_file_count": len(test_files),
                "pytest_ini_exists": pytest_ini.exists()
            }
        )
    def get_overall_status(self) -> HealthStatus:
        """Get overall health status."""
        if not self.results:
            return HealthStatus.UNKNOWN
        statuses = [r.status for r in self.results]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    def export_json(self, output_path: str) -> None:
        """Export health check results to JSON."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": self.get_overall_status().value,
            "checks": [r.to_dict() for r in self.results]
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
def main():
    checker = HealthChecker()
    results = checker.run_all_checks()
    # Export to JSON
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    checker.export_json(str(output_dir / "health_check.json"))
    # Print results
    print("=" * 60)
    print("HEALTH CHECK REPORT")
    print("=" * 60)
    for result in results:
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓"
        }.get(result.status, "❓")
        print(f"{status_icon} {result.name}: {result.message}")
    print("=" * 60)
    overall = checker.get_overall_status()
    print(f"Overall Status: {overall.value.upper()}")
    print("=" * 60)
    # Exit with appropriate code
    sys.exit(0 if overall == HealthStatus.HEALTHY else 1)
if __name__ == "__main__":
    main()
