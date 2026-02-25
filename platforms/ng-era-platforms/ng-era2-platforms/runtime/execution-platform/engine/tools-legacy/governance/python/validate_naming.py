# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate_naming
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
资源命名验证工具
用途: 验证资源命名是否符合组织规范
使用: python validate_naming.py --files <file1> <file2> --policies <policy_dir> --schemas <schema_dir>
"""
# MNGA-002: Import organization needs review
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
try:
    import yaml
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    print("Error: Required packages not installed")
    print("Install with: pip install pyyaml jsonschema")
    sys.exit(1)
class Colors:
    """ANSI color codes"""
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
class NamingValidator:
    """命名规范验证器"""
    def __init__(self, policies_dir: Path, schemas_dir: Path):
        self.policies_dir = policies_dir
        self.schemas_dir = schemas_dir
        self.policies = {}
        self.schemas = {}
        self.load_policies()
        self.load_schemas()
    def load_policies(self):
        """加载命名策略"""
        if not self.policies_dir.exists():
            print(
                f"{Colors.RED}Error: Policies directory not found: {self.policies_dir}{Colors.RESET}"
            )
            sys.exit(1)
        for policy_file in self.policies_dir.rglob("*.yaml"):
            try:
                with open(policy_file) as f:
                    policy = yaml.safe_load(f)
                    if policy and policy.get("kind") == "NamingPolicy":
                        policy_name = policy["metadata"]["name"]
                        self.policies[policy_name] = policy
                        print(f"Loaded policy: {policy_name}")
            except Exception as e:
                print(
                    f"{Colors.YELLOW}Warning: Failed to load policy {policy_file}: {e}{Colors.RESET}"
                )
    def load_schemas(self):
        """加载验证 schema"""
        if not self.schemas_dir.exists():
            print(
                f"{Colors.RED}Error: Schemas directory not found: {self.schemas_dir}{Colors.RESET}"
            )
            sys.exit(1)
        for schema_file in self.schemas_dir.glob("*.yaml"):
            try:
                with open(schema_file) as f:
                    schema = yaml.safe_load(f)
                    schema_name = schema_file.stem
                    self.schemas[schema_name] = schema
                    print(f"Loaded schema: {schema_name}")
            except Exception as e:
                print(
                    f"{Colors.YELLOW}Warning: Failed to load schema {schema_file}: {e}{Colors.RESET}"
                )
    def validate_k8s_deployment(self, resource: Dict, file_path: Path) -> List[Dict]:
        """验证 Kubernetes Deployment 命名"""
        violations = []
        if resource.get("kind") != "Deployment":
            return violations
        name = resource.get("metadata", {}).get("name", "")
        namespace = resource.get("metadata", {}).get("namespace", "default")
        # 使用 k8s-deployment-standard 策略
        policy = self.policies.get("k8s-deployment-standard")
        if not policy:
            return violations
        # 提取命名模式
        pattern = policy["spec"].get("pattern", "")
        # 转换模板为正则表达式
        regex_pattern = (
            r"^(dev|staging|prod)-[a-z0-9-]+-deploy-v\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        )
        if not re.match(regex_pattern, name):
            violations.append(
                {
                    "file": str(file_path),
                    "resource": "Deployment",
                    "name": name,
                    "namespace": namespace,
                    "violation": "Invalid naming pattern",
                    "expected": pattern,
                    "actual": name,
                    "severity": "error",
                    "policy": "k8s-deployment-standard",
                }
            )
        # 检查长度
        if len(name) > 63:
            violations.append(
                {
                    "file": str(file_path),
                    "resource": "Deployment",
                    "name": name,
                    "violation": "Name exceeds maximum length",
                    "expected": "<= 63 characters",
                    "actual": f"{len(name)} characters",
                    "severity": "error",
                }
            )
        return violations
    def validate_file(self, file_path: Path) -> List[Dict]:
        """验证单个文件"""
        violations = []
        if file_path.suffix not in [".yaml", ".yml"]:
            return violations
        try:
            with open(file_path) as f:
                docs = yaml.safe_load_all(f)
                for doc in docs:
                    if not doc:
                        continue
                    # Kubernetes 资源验证
                    if "apiVersion" in doc and "kind" in doc:
                        violations.extend(self.validate_k8s_deployment(doc, file_path))
        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning: Failed to parse {file_path}: {e}{Colors.RESET}"
            )
        return violations
    def validate_files(self, files: List[Path]) -> Tuple[List[Dict], int, int]:
        """验证多个文件"""
        all_violations = []
        total_files = 0
        compliant_files = 0
        for file_path in files:
            if not file_path.exists():
                print(
                    f"{Colors.YELLOW}Warning: File not found: {file_path}{Colors.RESET}"
                )
                continue
            total_files += 1
            violations = self.validate_file(file_path)
            if violations:
                all_violations.extend(violations)
            else:
                compliant_files += 1
        return all_violations, total_files, compliant_files
def generate_report(
    violations: List[Dict], total_files: int, compliant_files: int, format: str = "text"
):
    """生成验证报告"""
    if format == "json":
        report = {
            "status": "passed" if not violations else "failed",
            "summary": {
                "total_files": total_files,
                "compliant": compliant_files,
                "violations": len(violations),
            },
            "violation_details": violations,
        }
        print(json.dumps(report, indent=2))
    elif format == "text":
        print(f"\n{Colors.BOLD}=== Naming Compliance Report ==={Colors.RESET}\n")
        # 摘要
        compliance_rate = (
            (compliant_files / total_files * 100) if total_files > 0 else 0
        )
        status_color = Colors.GREEN if not violations else Colors.RED
        status = "PASSED" if not violations else "FAILED"
        print(f"Status: {status_color}{status}{Colors.RESET}")
        print(f"Compliance Rate: {compliance_rate:.1f}%")
        print(f"Files Checked: {total_files}")
        print(f"Compliant: {Colors.GREEN}{compliant_files}{Colors.RESET}")
        print(f"Violations: {Colors.RED}{len(violations)}{Colors.RESET}\n")
        # 违规详情
        if violations:
            print(f"{Colors.BOLD}Violations:{Colors.RESET}\n")
            for i, v in enumerate(violations, 1):
                print(f"{i}. {Colors.RED}✗{Colors.RESET} {v['file']}")
                print(f"   Resource: {v['resource']} '{v['name']}'")
                print(f"   Violation: {v['violation']}")
                print(f"   Expected: {v.get('expected', 'N/A')}")
                print(f"   Actual: {v.get('actual', 'N/A')}")
                if "policy" in v:
                    print(f"   Policy: {v['policy']}")
                print()
        else:
            print(
                f"{Colors.GREEN}✓ All resources comply with naming standards{Colors.RESET}\n"
            )
def main():
    parser = argparse.ArgumentParser(
        description="Validate resource naming against gl-platform.governance policies"
    )
    parser.add_argument("--files", nargs="+", type=Path, help="Files to validate")
    parser.add_argument(
        "--files-list",
        type=Path,
        help="File containing list of files to validate (one per line)",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        required=True,
        help="Directory containing naming policies",
    )
    parser.add_argument(
        "--schemas",
        type=Path,
        required=True,
        help="Directory containing validation schemas",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file for report (JSON format)"
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument(
        "--changed-files-only",
        action="store_true",
        help="Only validate changed files (requires git)",
    )
    parser.add_argument(
        "--base-ref",
        help="Base git reference for diff (used with --changed-files-only)",
    )
    parser.add_argument(
        "--head-ref",
        help="Head git reference for diff (used with --changed-files-only)",
    )
    args = parser.parse_args()
    # 收集要验证的文件
    files_to_validate = []
    if args.files:
        files_to_validate.extend(args.files)
    if args.files_list:
        with open(args.files_list) as f:
            for line in f:
                file_path = Path(line.strip())
                if file_path.exists():
                    files_to_validate.append(file_path)
    if args.changed_files_only:
        # 获取 git 变更的文件
        import subprocess
        try:
            base = args.base_ref or "origin/main"
            head = args.head_ref or "HEAD"
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base}...{head}"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                file_path = Path(line.strip())
                if file_path.suffix in [".yaml", ".yml", ".json"]:
                    files_to_validate.append(file_path)
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}Error: Failed to get git diff: {e}{Colors.RESET}")
            sys.exit(1)
    if not files_to_validate:
        print(f"{Colors.YELLOW}No files to validate{Colors.RESET}")
        sys.exit(0)
    # 初始化验证器
    validator = NamingValidator(args.policies, args.schemas)
    # 执行验证
    violations, total_files, compliant_files = validator.validate_files(
        files_to_validate
    )
    # 生成报告
    if args.output:
        # 写入 JSON 报告
        report = {
            "status": "passed" if not violations else "failed",
            "summary": {
                "total_files": total_files,
                "compliant": compliant_files,
                "violations": len(violations),
            },
            "violation_details": violations,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.output}")
    # 输出到控制台
    generate_report(violations, total_files, compliant_files, args.format)
    # 退出码
    sys.exit(1 if violations else 0)
if __name__ == "__main__":
    main()
