#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: auto-remediate-policy
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# ECO-Layer: GL30-49 (Execution)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Violation Auto-Remediation Script for MachineNativeOps
Automatically detects and fixes common policy violations including:
- Naming convention violations (kebab-case enforcement)
- Semantic health issues (missing mappings, duplicates)
- Security policy gaps (missing documentation)
Usage:
    python3 scripts/auto-remediate-policy.py --policy naming --scan
    python3 scripts/auto-remediate-policy.py --policy naming --fix
    python3 scripts/auto-remediate-policy.py --policy semantic --suggest
    python3 scripts/auto-remediate-policy.py --all --report
"""
# MNGA-002: Import organization needs review
import re
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
class NamingPolicyRemediator:
    """Handles naming convention violations."""
    # Patterns that should be kebab-case
    KEBAB_CASE_PATTERN = re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+)*$')
    # Files/directories to skip
    SKIP_PATTERNS = [
        r'^README\.md$',
        r'^LICENSE$',
        r'^CHANGELOG\.md$',
        r'^CONTRIBUTING\.md$',
        r'^CODE_OF_CONDUCT\.md$',
        r'^SECURITY\.md$',
        r'^\.git',
        r'^\.github',
        r'^node_modules',
        r'^__pycache__',
        r'\.pyc$',
        r'^HANDOFF_NEXT_PR\.md$',
        r'^PHASE1_.*\.md$',
        r'^PENDING_INTEGRATIONS.*\.md$',
        r'^AUTONOMY_SUMMARY\.',
        r'^LANGUAGE_GOVERNANCE_DASHBOARD\.',
        r'^DOCUMENTATION_PORTAL\.md$',
        r'^AUTONOMY_CLASSIFICATION_FRAMEWORK\.md$',
    ]
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path)
        self.violations = []
        self.fixes = []
    def should_skip(self, name):
        """Check if file/directory should be skipped."""
        for pattern in self.SKIP_PATTERNS:
            if re.match(pattern, name):
                return True
        return False
    def to_kebab_case(self, name):
        """Convert a name to kebab-case."""
        # Handle file extension
        parts = name.rsplit('.', 1)
        base_name = parts[0]
        extension = f".{parts[1]}" if len(parts) > 1 else ""
        # Convert camelCase to kebab-case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', base_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
        # Replace underscores and spaces with hyphens
        s3 = s2.replace('_', '-').replace(' ', '-')
        # Convert to lowercase and remove multiple hyphens
        s4 = re.sub(r'-+', '-', s3.lower())
        # Remove leading/trailing hyphens
        s5 = s4.strip('-')
        return s5 + extension
    def scan(self, directories=None):
        """Scan for naming violations."""
        self.violations = []
        if directories is None:
            directories = ['controlplane', 'scripts', 'docs']
        for dir_name in directories:
            dir_path = self.root_path / dir_name
            if not dir_path.exists():
                continue
            for item in dir_path.rglob('*'):
                name = item.name
                if self.should_skip(name):
                    continue
                # Check for violations
                has_space = ' ' in name
                has_underscore = '_' in name and not name.endswith('.py')  # Allow Python files
                has_uppercase = bool(re.search(r'[A-Z]', name.rsplit('.', 1)[0]))
                if has_space or has_underscore or has_uppercase:
                    suggested_name = self.to_kebab_case(name)
                    self.violations.append({
                        'path': str(item.relative_to(self.root_path)),
                        'current_name': name,
                        'suggested_name': suggested_name,
                        'issues': {
                            'has_space': has_space,
                            'has_underscore': has_underscore,
                            'has_uppercase': has_uppercase
                        },
                        'type': 'directory' if item.is_dir() else 'file'
                    })
        return self.violations
    def fix(self, dry_run=True):
        """Fix naming violations."""
        if not self.violations:
            self.scan()
        self.fixes = []
        for violation in self.violations:
            old_path = self.root_path / violation['path']
            new_name = violation['suggested_name']
            new_path = old_path.parent / new_name
            if dry_run:
                self.fixes.append({
                    'action': 'rename',
                    'from': str(old_path),
                    'to': str(new_path),
                    'status': 'dry-run'
                })
            else:
                try:
                    old_path.rename(new_path)
                    self.fixes.append({
                        'action': 'rename',
                        'from': str(old_path),
                        'to': str(new_path),
                        'status': 'success'
                    })
                except Exception as e:
                    self.fixes.append({
                        'action': 'rename',
                        'from': str(old_path),
                        'to': str(new_path),
                        'status': 'error',
                        'error': str(e)
                    })
        return self.fixes
class SemanticPolicyRemediator:
    """Handles semantic health issues."""
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path)
        self.issues = []
        self.suggestions = []
    def load_registry(self):
        """Load module registry."""
        registry_path = self.root_path / 'controlplane/baseline/modules/REGISTRY.yaml'
        if not registry_path.exists():
            return None
        with open(registry_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    def scan(self):
        """Scan for semantic health issues."""
        self.issues = []
        registry = self.load_registry()
        if not registry or 'modules' not in registry:
            self.issues.append({
                'type': 'missing_registry',
                'message': 'Module registry not found or empty',
                'severity': 'high'
            })
            return self.issues
        modules = registry['modules']
        namespaces = {}
        for module in modules:
            module_id = module.get('module_id', 'unknown')
            namespace = module.get('namespace', '')
            semantic_health = module.get('semantic_health_score', 0)
            # Check for low semantic health
            if semantic_health < 90:
                self.issues.append({
                    'type': 'low_semantic_health',
                    'module': module_id,
                    'current_score': semantic_health,
                    'target_score': 90,
                    'severity': 'medium' if semantic_health >= 80 else 'high'
                })
            # Check for namespace conflicts
            if namespace in namespaces:
                self.issues.append({
                    'type': 'namespace_conflict',
                    'namespace': namespace,
                    'modules': [namespaces[namespace], module_id],
                    'severity': 'high'
                })
            else:
                namespaces[namespace] = module_id
            # Check for missing semantic mappings
            if not module.get('semantic_mappings'):
                self.issues.append({
                    'type': 'missing_semantic_mappings',
                    'module': module_id,
                    'severity': 'medium'
                })
        return self.issues
    def suggest(self):
        """Generate suggestions for semantic improvements."""
        if not self.issues:
            self.scan()
        self.suggestions = []
        for issue in self.issues:
            if issue['type'] == 'low_semantic_health':
                self.suggestions.append({
                    'issue': issue,
                    'suggestion': f"Review module {issue['module']} semantic mappings. "
                                  f"Current score: {issue['current_score']}%, target: {issue['target_score']}%. "
                                  f"Consider adding more semantic mappings or fixing namespace consistency.",
                    'action': 'manual_review'
                })
            elif issue['type'] == 'namespace_conflict':
                self.suggestions.append({
                    'issue': issue,
                    'suggestion': f"Namespace '{issue['namespace']}' is used by multiple modules: {issue['modules']}. "
                                  f"Each module should have a unique namespace.",
                    'action': 'rename_namespace'
                })
            elif issue['type'] == 'missing_semantic_mappings':
                self.suggestions.append({
                    'issue': issue,
                    'suggestion': f"Module {issue['module']} has no semantic mappings. "
                                  f"Add semantic_mappings to the module manifest.",
                    'action': 'add_mappings',
                    'template': {
                        'semantic_mappings': [
                            {'concept': 'example-concept', 'mapping': 'mno-example'}
                        ]
                    }
                })
        return self.suggestions
class SecurityPolicyRemediator:
    """Handles security policy gaps."""
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path)
        self.gaps = []
        self.suggestions = []
    def scan(self):
        """Scan for security policy gaps."""
        self.gaps = []
        # Check for SECURITY.md
        security_md = self.root_path / 'SECURITY.md'
        if not security_md.exists():
            self.gaps.append({
                'type': 'missing_security_policy',
                'file': 'SECURITY.md',
                'severity': 'high'
            })
        # Check for supply chain documentation
        supply_chain_doc = self.root_path / 'docs/supply-chain-security.md'
        if not supply_chain_doc.exists():
            self.gaps.append({
                'type': 'missing_supply_chain_doc',
                'file': 'docs/supply-chain-security.md',
                'severity': 'medium'
            })
        # Check for security workflow
        security_workflow = self.root_path / '.github/workflows/supply-chain-security.yml'
        if not security_workflow.exists():
            self.gaps.append({
                'type': 'missing_security_workflow',
                'file': '.github/workflows/supply-chain-security.yml',
                'severity': 'high'
            })
        return self.gaps
    def suggest(self):
        """Generate suggestions for security improvements."""
        if not self.gaps:
            self.scan()
        self.suggestions = []
        for gap in self.gaps:
            if gap['type'] == 'missing_security_policy':
                self.suggestions.append({
                    'gap': gap,
                    'suggestion': "Create SECURITY.md with vulnerability reporting guidelines.",
                    'action': 'create_file',
                    'template': self._get_security_md_template()
                })
            elif gap['type'] == 'missing_supply_chain_doc':
                self.suggestions.append({
                    'gap': gap,
                    'suggestion': "Create supply chain security documentation.",
                    'action': 'create_file'
                })
        return self.suggestions
    def _get_security_md_template(self):
        return """# Security Policy
## Supported Versions
| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
## Reporting a Vulnerability
Please report security vulnerabilities by emailing security@example.com.
We will respond within 48 hours and provide a timeline for fixes.
## Security Measures
- All artifacts are signed with Cosign
- SBOM generated for all releases
- SLSA Level 3 provenance
- Regular vulnerability scanning with Trivy
"""
def generate_report(naming_violations, semantic_issues, security_gaps, output_path=None):
    """Generate a comprehensive remediation report."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Policy Remediation Report
#*Generated**: {timestamp}
---
## Summary
| Policy | Issues Found | Severity |
|--------|--------------|----------|
| Naming | {len(naming_violations)} | {'High' if naming_violations else 'None'} |
| Semantic | {len(semantic_issues)} | {'High' if any(i.get('severity') == 'high' for i in semantic_issues) else 'Medium' if semantic_issues else 'None'} |
| Security | {len(security_gaps)} | {'High' if any(g.get('severity') == 'high' for g in security_gaps) else 'Medium' if security_gaps else 'None'} |
---
## Naming Policy Violations
"""
    if naming_violations:
        report += "| Current Name | Suggested Name | Issues |\n"
        report += "|--------------|----------------|--------|\n"
        for v in naming_violations[:20]:  # Limit to 20
            issues = []
            if v['issues']['has_space']:
                issues.append('space')
            if v['issues']['has_underscore']:
                issues.append('underscore')
            if v['issues']['has_uppercase']:
                issues.append('uppercase')
            report += f"| `{v['current_name']}` | `{v['suggested_name']}` | {', '.join(issues)} |\n"
        if len(naming_violations) > 20:
            report += f"\n*... and {len(naming_violations) - 20} more violations*\n"
    else:
        report += "✅ No naming violations found.\n"
    report += """
---
## Semantic Health Issues
"""
    if semantic_issues:
        for issue in semantic_issues:
            report += f"### {issue['type'].replace('_', ' ').title()}\n\n"
            if issue['type'] == 'low_semantic_health':
                report += f"- **Module**: {issue['module']}\n"
                report += f"- **Current Score**: {issue['current_score']}%\n"
                report += f"- **Target Score**: {issue['target_score']}%\n"
            elif issue['type'] == 'namespace_conflict':
                report += f"- **Namespace**: {issue['namespace']}\n"
                report += f"- **Conflicting Modules**: {', '.join(issue['modules'])}\n"
            report += f"- **Severity**: {issue['severity']}\n\n"
    else:
        report += "✅ No semantic health issues found.\n"
    report += """
---
## Security Policy Gaps
"""
    if security_gaps:
        for gap in security_gaps:
            report += f"- **{gap['type'].replace('_', ' ').title()}**: `{gap['file']}` (Severity: {gap['severity']})\n"
    else:
        report += "✅ No security policy gaps found.\n"
    report += """
---
## Recommendations
1. **Naming**: Run `python3 scripts/auto-remediate-policy.py --policy naming --fix` to auto-fix naming violations
2. **Semantic**: Review modules with low semantic health scores and add semantic mappings
3. **Security**: Ensure all security documentation and workflows are in place
---
#Report generated by auto-remediate-policy.py*
"""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Report saved to: {output_path}")
    return report
def main():
    parser = argparse.ArgumentParser(
        description="Policy Violation Auto-Remediation for MachineNativeOps"
    )
    parser.add_argument(
        '--policy', '-p',
        choices=['naming', 'semantic', 'security', 'all'],
        default='all',
        help='Policy to check/remediate'
    )
    parser.add_argument(
        '--scan', '-s',
        action='store_true',
        help='Scan for violations (default action)'
    )
    parser.add_argument(
        '--fix', '-f',
        action='store_true',
        help='Apply fixes (naming policy only)'
    )
    parser.add_argument(
        '--suggest',
        action='store_true',
        help='Generate suggestions'
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate comprehensive report'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for report'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Dry run mode (default: True)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply changes (disables dry-run)'
    )
    args = parser.parse_args()
    # Default to scan if no action specified
    if not (args.scan or args.fix or args.suggest or args.report):
        args.scan = True
    naming_violations = []
    semantic_issues = []
    security_gaps = []
    # Naming policy
    if args.policy in ['naming', 'all']:
        print("🔍 Checking naming policy...")
        naming_remediator = NamingPolicyRemediator()
        naming_violations = naming_remediator.scan()
        print(f"   Found {len(naming_violations)} naming violations")
        if args.fix:
            dry_run = not args.apply
            fixes = naming_remediator.fix(dry_run=dry_run)
            mode = "DRY RUN" if dry_run else "APPLIED"
            print(f"   [{mode}] {len(fixes)} fixes prepared")
            if not dry_run:
                for fix in fixes:
                    print(f"   - {fix['from']} → {fix['to']}: {fix['status']}")
    # Semantic policy
    if args.policy in ['semantic', 'all']:
        print("🔍 Checking semantic policy...")
        semantic_remediator = SemanticPolicyRemediator()
        semantic_issues = semantic_remediator.scan()
        print(f"   Found {len(semantic_issues)} semantic issues")
        if args.suggest:
            suggestions = semantic_remediator.suggest()
            print(f"   Generated {len(suggestions)} suggestions")
            for s in suggestions:
                print(f"   - {s['suggestion'][:80]}...")
    # Security policy
    if args.policy in ['security', 'all']:
        print("🔍 Checking security policy...")
        security_remediator = SecurityPolicyRemediator()
        security_gaps = security_remediator.scan()
        print(f"   Found {len(security_gaps)} security gaps")
        if args.suggest:
            suggestions = security_remediator.suggest()
            print(f"   Generated {len(suggestions)} suggestions")
    # Generate report
    if args.report:
        output_path = args.output or 'docs/POLICY_REMEDIATION_REPORT.md'
        generate_report(naming_violations, semantic_issues, security_gaps, output_path)
        print(f"\n📊 Report generated: {output_path}")
    # Summary
    total_issues = len(naming_violations) + len(semantic_issues) + len(security_gaps)
    print(f"\n📋 Total issues found: {total_issues}")
    if total_issues == 0:
        print("✅ All policies are compliant!")
        return 0
    else:
        print("⚠️ Some policy violations found. Review the report for details.")
        return 1
if __name__ == "__main__":
    sys.exit(main())