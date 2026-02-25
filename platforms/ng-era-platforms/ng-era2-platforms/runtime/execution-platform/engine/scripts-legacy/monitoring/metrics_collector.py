#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: metrics_collector
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Machine-Native Observability Metrics Collector
Collects and exports metrics for monitoring dashboards.
"""
# MNGA-002: Import organization needs review
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
class MetricsCollector:
    """Collects observability metrics from the repository."""
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.metrics: Dict[str, Any] = {}
    def collect_all(self) -> Dict[str, Any]:
        """Collect all metrics."""
        self.metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "repository": self._collect_repo_metrics(),
            "code_quality": self._collect_code_quality_metrics(),
            "ci_cd": self._collect_ci_metrics(),
            "security": self._collect_security_metrics(),
            "documentation": self._collect_doc_metrics(),
        }
        return self.metrics
    def _collect_repo_metrics(self) -> Dict[str, Any]:
        """Collect repository structure metrics."""
        metrics = {
            "total_files": 0,
            "python_files": 0,
            "yaml_files": 0,
            "markdown_files": 0,
            "total_lines": 0,
        }
        # Use pathlib for safe file counting instead of shell commands
        exclude_dirs = {'archive', '.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        for ext, key in [(".py", "python_files"), (".yaml", "yaml_files"), 
                         (".yml", "yaml_files"), (".md", "markdown_files")]:
            try:
                count = 0
                for file_path in self.repo_root.rglob(f"*{ext}"):
                    # Skip excluded directories
                    if not any(excluded in file_path.parts for excluded in exclude_dirs):
                        if file_path.is_file():
                            count += 1
                metrics[key] += count
            except (OSError, PermissionError):
                pass
        metrics["total_files"] = sum([
            metrics["python_files"], 
            metrics["yaml_files"], 
            metrics["markdown_files"]
        ])
        return metrics
    def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {
            "test_files": 0,
            "test_coverage_estimate": 0,
            "linting_enabled": True,
            "type_hints_usage": "partial",
        }
        # Use pathlib for safe file counting instead of shell commands
        exclude_dirs = {'archive', '.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        try:
            count = 0
            for file_path in self.repo_root.rglob("*.py"):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                # Count test files
                if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
                    if file_path.is_file():
                        count += 1
            metrics["test_files"] = count
        except (OSError, PermissionError):
            pass
        return metrics
    def _collect_ci_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD metrics."""
        workflows_dir = self.repo_root / ".github" / "workflows"
        metrics = {
            "workflow_count": 0,
            "workflows": [],
        }
        if workflows_dir.exists():
            workflows = list(workflows_dir.glob("*.yml"))
            metrics["workflow_count"] = len(workflows)
            metrics["workflows"] = [w.stem for w in workflows]
        return metrics
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        return {
            "security_policy_exists": (self.repo_root / "SECURITY.md").exists(),
            "secrets_baseline_exists": (self.repo_root / ".secrets.baseline").exists(),
            "pre_commit_enabled": (self.repo_root / ".pre-commit-config.yaml").exists(),
        }
    def _collect_doc_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        docs = {
            "readme_exists": (self.repo_root / "README.md").exists(),
            "quickstart_exists": (self.repo_root / "QUICKSTART.md").exists(),
            "project_status_exists": (self.repo_root / "PROJECT_STATUS.md").exists(),
        }
        return docs
    def export_json(self, output_path: str) -> None:
        """Export metrics to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# HELP repo_files_total Total number of files by type")
        lines.append("# TYPE repo_files_total gauge")
        repo = self.metrics.get("repository", {})
        for key in ["python_files", "yaml_files", "markdown_files"]:
            lines.append(f'repo_files_total{{type="{key}"}} {repo.get(key, 0)}')
        lines.append("# HELP code_quality_test_files Number of test files")
        lines.append("# TYPE code_quality_test_files gauge")
        lines.append(f'code_quality_test_files {self.metrics.get("code_quality", {}).get("test_files", 0)}')
        lines.append("# HELP ci_workflow_count Number of CI workflows")
        lines.append("# TYPE ci_workflow_count gauge")
        lines.append(f'ci_workflow_count {self.metrics.get("ci_cd", {}).get("workflow_count", 0)}')
        return "\n".join(lines)
def main():
    collector = MetricsCollector()
    metrics = collector.collect_all()
    # Export to JSON
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    collector.export_json(str(output_dir / "metrics.json"))
    # Print Prometheus format
    print(collector.export_prometheus())
    # Print summary
    print("\n--- Metrics Summary ---")
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()
