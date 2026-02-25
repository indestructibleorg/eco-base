#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: gl_reporter
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Reporter - Governance Layer Report Generator
MachineNativeOps GL Architecture Implementation
This module provides comprehensive reporting capabilities for GL gl-platform.gl-platform.governance,
including dashboards, compliance reports, layer analysis, and trend tracking.
"""
# MNGA-002: Import organization needs review
import yaml
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GLReporter')
class ReportType(Enum):
    """Report type enumeration."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    LAYER = "layer"
    COMPLIANCE = "compliance"
    DASHBOARD = "dashboard"
    TREND = "trend"
    HEALTH = "health"
class OutputFormat(Enum):
    """Output format enumeration."""
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
@dataclass
class LayerMetrics:
    """Metrics for a single GL layer."""
    layer_id: str
    layer_name: str
    artifact_count: int = 0
    compliant_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    avg_age_days: float = 0.0
    oldest_artifact_days: int = 0
    newest_artifact_days: int = 0
    owners: List[str] = field(default_factory=list)
    kinds: Dict[str, int] = field(default_factory=dict)
    @property
    def compliance_rate(self) -> float:
        if self.artifact_count == 0:
            return 100.0
        return (self.compliant_count / self.artifact_count) * 100
    @property
    def health_score(self) -> float:
        """Calculate health score (0-100)."""
        score = 100.0
        # Deduct for errors
        score -= min(self.error_count * 10, 50)
        # Deduct for warnings
        score -= min(self.warning_count * 2, 20)
        # Deduct for old artifacts
        if self.avg_age_days > 180:
            score -= 10
        elif self.avg_age_days > 90:
            score -= 5
        return max(0, score)
@dataclass
class GovernanceMetrics:
    """Overall gl-platform.gl-platform.governance metrics."""
    total_artifacts: int = 0
    total_layers: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    overall_compliance: float = 0.0
    overall_health: float = 0.0
    layer_metrics: Dict[str, LayerMetrics] = field(default_factory=dict)
    artifact_by_kind: Dict[str, int] = field(default_factory=dict)
    artifact_by_owner: Dict[str, int] = field(default_factory=dict)
    recent_changes: List[Dict[str, Any]] = field(default_factory=list)
class GLReporter:
    """GL Report Generator."""
    LAYER_INFO = {
        'GL00-09': {'name': 'Strategic Layer', 'name_zh': '戰略層', 'color': '#1E3A8A'},
        'GL10-29': {'name': 'Operational Layer', 'name_zh': '運營層', 'color': '#059669'},
        'GL30-49': {'name': 'Execution Layer', 'name_zh': '執行層', 'color': '#DC2626'},
        'GL50-59': {'name': 'Observability Layer', 'name_zh': '觀測層', 'color': '#7C3AED'},
        'GL60-80': {'name': 'Advanced/Feedback Layer', 'name_zh': '進階/回饋層', 'color': '#EA580C'},
        'GL81-83': {'name': 'Extended Layer', 'name_zh': '擴展層', 'color': '#0891B2'},
        'GL90-99': {'name': 'Meta-Specification Layer', 'name_zh': '元規範層', 'color': '#4F46E5'},
    }
    def __init__(self, workspace_path: str = '.'):
        self.workspace_path = Path(workspace_path).resolve()
        self.gl-platform.gl-platform.governance_path = self.workspace_path / 'workspace' / 'gl-platform.gl-platform.governance'
    def collect_metrics(self) -> GovernanceMetrics:
        """Collect gl-platform.gl-platform.governance metrics from workspace."""
        metrics = GovernanceMetrics()
        if not self.gl-platform.gl-platform.governance_path.exists():
            logger.warning(f"Governance path not found: {self.gl-platform.gl-platform.governance_path}")
            return metrics
        # Initialize layer metrics
        for layer_id, info in self.LAYER_INFO.items():
            metrics.layer_metrics[layer_id] = LayerMetrics(
                layer_id=layer_id,
                layer_name=info['name']
            )
        # Scan artifacts
        yaml_files = list(self.gl-platform.gl-platform.governance_path.rglob('*.yaml')) + \
                     list(self.gl-platform.gl-platform.governance_path.rglob('*.yml'))
        now = datetime.utcnow()
        for file_path in yaml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artifact = yaml.safe_load(f)
                if not artifact or 'apiVersion' not in artifact:
                    continue
                metrics.total_artifacts += 1
                metadata = artifact.get('metadata', {})
                layer = metadata.get('layer', 'unknown')
                kind = artifact.get('kind', 'unknown')
                owner = metadata.get('owner', 'unknown')
                # Update kind counts
                metrics.artifact_by_kind[kind] = metrics.artifact_by_kind.get(kind, 0) + 1
                # Update owner counts
                metrics.artifact_by_owner[owner] = metrics.artifact_by_owner.get(owner, 0) + 1
                # Update layer metrics
                if layer in metrics.layer_metrics:
                    layer_metric = metrics.layer_metrics[layer]
                    layer_metric.artifact_count += 1
                    layer_metric.kinds[kind] = layer_metric.kinds.get(kind, 0) + 1
                    if owner not in layer_metric.owners:
                        layer_metric.owners.append(owner)
                    # Calculate age
                    updated_at = metadata.get('updated_at', metadata.get('created_at', ''))
                    if updated_at:
                        try:
                            update_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                            age_days = (now.replace(tzinfo=update_date.tzinfo) - update_date).days
                            # Track oldest/newest
                            if layer_metric.oldest_artifact_days == 0 or age_days > layer_metric.oldest_artifact_days:
                                layer_metric.oldest_artifact_days = age_days
                            if layer_metric.newest_artifact_days == 0 or age_days < layer_metric.newest_artifact_days:
                                layer_metric.newest_artifact_days = age_days
                            # Update average
                            current_total = layer_metric.avg_age_days * (layer_metric.artifact_count - 1)
                            layer_metric.avg_age_days = (current_total + age_days) / layer_metric.artifact_count
                        except Exception:
                            pass
                    # Check compliance (basic check)
                    has_required = all([
                        metadata.get('name'),
                        metadata.get('version'),
                        metadata.get('owner'),
                        artifact.get('spec')
                    ])
                    if has_required:
                        layer_metric.compliant_count += 1
                    else:
                        layer_metric.error_count += 1
                        metrics.total_errors += 1
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
        # Calculate overall metrics
        metrics.total_layers = len([l for l in metrics.layer_metrics.values() if l.artifact_count > 0])
        if metrics.total_artifacts > 0:
            compliant = sum(l.compliant_count for l in metrics.layer_metrics.values())
            metrics.overall_compliance = (compliant / metrics.total_artifacts) * 100
            health_scores = [l.health_score for l in metrics.layer_metrics.values() if l.artifact_count > 0]
            if health_scores:
                metrics.overall_health = sum(health_scores) / len(health_scores)
        return metrics
    def generate_report(self, report_type: ReportType, output_format: OutputFormat,
                       layer_filter: Optional[str] = None) -> str:
        """Generate a report."""
        metrics = self.collect_metrics()
        if report_type == ReportType.SUMMARY:
            return self._generate_summary_report(metrics, output_format)
        elif report_type == ReportType.DETAILED:
            return self._generate_detailed_report(metrics, output_format)
        elif report_type == ReportType.LAYER:
            return self._generate_layer_report(metrics, output_format, layer_filter)
        elif report_type == ReportType.COMPLIANCE:
            return self._generate_compliance_report(metrics, output_format)
        elif report_type == ReportType.DASHBOARD:
            return self._generate_dashboard(metrics, output_format)
        elif report_type == ReportType.HEALTH:
            return self._generate_health_report(metrics, output_format)
        else:
            return self._generate_summary_report(metrics, output_format)
    def _generate_summary_report(self, metrics: GovernanceMetrics, format: OutputFormat) -> str:
        """Generate summary report."""
        if format == OutputFormat.JSON:
            return self._to_json(metrics)
        elif format == OutputFormat.YAML:
            return self._to_yaml(metrics)
        elif format == OutputFormat.HTML:
            return self._summary_to_html(metrics)
        else:
            return self._summary_to_markdown(metrics)
    def _summary_to_markdown(self, metrics: GovernanceMetrics) -> str:
        """Generate markdown summary."""
        report = []
        report.append("# GL Governance Summary Report\n")
        report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z\n")
        # Overall metrics
        report.append("## Overview\n")
        health_emoji = "🟢" if metrics.overall_health >= 80 else "🟡" if metrics.overall_health >= 60 else "🔴"
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Total Artifacts | {metrics.total_artifacts} |")
        report.append(f"| Active Layers | {metrics.total_layers} |")
        report.append(f"| Overall Compliance | {metrics.overall_compliance:.1f}% |")
        report.append(f"| Overall Health | {health_emoji} {metrics.overall_health:.1f}% |")
        report.append(f"| Total Errors | {metrics.total_errors} |")
        report.append(f"| Total Warnings | {metrics.total_warnings} |")
        report.append("")
        # Layer summary
        report.append("## Layer Summary\n")
        report.append("| Layer | Name | Artifacts | Compliance | Health |")
        report.append("|-------|------|-----------|------------|--------|")
        for layer_id in sorted(self.LAYER_INFO.keys()):
            layer = metrics.layer_metrics.get(layer_id)
            if layer and layer.artifact_count > 0:
                health_emoji = "🟢" if layer.health_score >= 80 else "🟡" if layer.health_score >= 60 else "🔴"
                report.append(
                    f"| {layer_id} | {layer.layer_name} | {layer.artifact_count} | "
                    f"{layer.compliance_rate:.1f}% | {health_emoji} {layer.health_score:.0f}% |"
                )
        report.append("")
        # Top artifact kinds
        report.append("## Artifact Distribution\n")
        report.append("### By Kind\n")
        report.append("| Kind | Count |")
        report.append("|------|-------|")
        for kind, count in sorted(metrics.artifact_by_kind.items(), key=lambda x: -x[1])[:10]:
            report.append(f"| {kind} | {count} |")
        report.append("")
        # Top owners
        report.append("### By Owner\n")
        report.append("| Owner | Count |")
        report.append("|-------|-------|")
        for owner, count in sorted(metrics.artifact_by_owner.items(), key=lambda x: -x[1])[:10]:
            report.append(f"| {owner} | {count} |")
        report.append("")
        return '\n'.join(report)
    def _generate_detailed_report(self, metrics: GovernanceMetrics, format: OutputFormat) -> str:
        """Generate detailed report."""
        if format != OutputFormat.MARKDOWN:
            return self._to_json(metrics)
        report = []
        report.append("# GL Governance Detailed Report\n")
        report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z\n")
        for layer_id in sorted(self.LAYER_INFO.keys()):
            layer = metrics.layer_metrics.get(layer_id)
            info = self.LAYER_INFO[layer_id]
            report.append(f"## {layer_id}: {info['name']} ({info['name_zh']})\n")
            if not layer or layer.artifact_count == 0:
                report.append("*No artifacts found for this layer*\n")
                continue
            # Layer metrics
            report.append("### Metrics\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Artifact Count | {layer.artifact_count} |")
            report.append(f"| Compliant | {layer.compliant_count} |")
            report.append(f"| Compliance Rate | {layer.compliance_rate:.1f}% |")
            report.append(f"| Health Score | {layer.health_score:.0f}% |")
            report.append(f"| Errors | {layer.error_count} |")
            report.append(f"| Warnings | {layer.warning_count} |")
            report.append(f"| Average Age | {layer.avg_age_days:.0f} days |")
            report.append(f"| Oldest Artifact | {layer.oldest_artifact_days} days |")
            report.append(f"| Newest Artifact | {layer.newest_artifact_days} days |")
            report.append("")
            # Artifact kinds
            report.append("### Artifact Types\n")
            report.append("| Kind | Count |")
            report.append("|------|-------|")
            for kind, count in sorted(layer.kinds.items(), key=lambda x: -x[1]):
                report.append(f"| {kind} | {count} |")
            report.append("")
            # Owners
            report.append("### Owners\n")
            for owner in sorted(layer.owners):
                report.append(f"- {owner}")
            report.append("")
        return '\n'.join(report)
    def _generate_layer_report(self, metrics: GovernanceMetrics, format: OutputFormat,
                              layer_filter: Optional[str] = None) -> str:
        """Generate layer-specific report."""
        if not layer_filter or layer_filter not in self.LAYER_INFO:
            return self._generate_summary_report(metrics, format)
        layer = metrics.layer_metrics.get(layer_filter)
        info = self.LAYER_INFO[layer_filter]
        if format != OutputFormat.MARKDOWN:
            return json.dumps({
                'layer_id': layer_filter,
                'layer_name': info['name'],
                'metrics': {
                    'artifact_count': layer.artifact_count if layer else 0,
                    'compliance_rate': layer.compliance_rate if layer else 100,
                    'health_score': layer.health_score if layer else 100,
                }
            }, indent=2)
        report = []
        report.append(f"# {layer_filter}: {info['name']} ({info['name_zh']})\n")
        report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z\n")
        if not layer or layer.artifact_count == 0:
            report.append("## Status: No Artifacts\n")
            report.append("This layer has no gl-platform.gl-platform.governance artifacts defined.\n")
            report.append("### Recommended Actions\n")
            report.append("1. Review the GL architecture specification")
            report.append("2. Create required artifacts for this layer")
            report.append("3. Run validation after creating artifacts")
            return '\n'.join(report)
        # Health indicator
        health_emoji = "🟢" if layer.health_score >= 80 else "🟡" if layer.health_score >= 60 else "🔴"
        report.append(f"## Health Status: {health_emoji} {layer.health_score:.0f}%\n")
        # Metrics table
        report.append("## Metrics\n")
        report.append("| Metric | Value | Status |")
        report.append("|--------|-------|--------|")
        compliance_status = "✅" if layer.compliance_rate >= 90 else "⚠️" if layer.compliance_rate >= 70 else "❌"
        report.append(f"| Compliance Rate | {layer.compliance_rate:.1f}% | {compliance_status} |")
        error_status = "✅" if layer.error_count == 0 else "❌"
        report.append(f"| Errors | {layer.error_count} | {error_status} |")
        warning_status = "✅" if layer.warning_count <= 2 else "⚠️"
        report.append(f"| Warnings | {layer.warning_count} | {warning_status} |")
        age_status = "✅" if layer.avg_age_days <= 90 else "⚠️" if layer.avg_age_days <= 180 else "❌"
        report.append(f"| Average Age | {layer.avg_age_days:.0f} days | {age_status} |")
        report.append("")
        # Artifact breakdown
        report.append("## Artifact Breakdown\n")
        report.append("| Kind | Count | Percentage |")
        report.append("|------|-------|------------|")
        for kind, count in sorted(layer.kinds.items(), key=lambda x: -x[1]):
            pct = (count / layer.artifact_count) * 100
            report.append(f"| {kind} | {count} | {pct:.1f}% |")
        report.append("")
        # Recommendations
        report.append("## Recommendations\n")
        if layer.error_count > 0:
            report.append(f"1. ⚠️ Fix {layer.error_count} validation errors")
        if layer.avg_age_days > 90:
            report.append(f"2. 📅 Review and update artifacts (avg age: {layer.avg_age_days:.0f} days)")
        if layer.compliance_rate < 90:
            report.append("3. 📋 Improve compliance rate to 90%+")
        if layer.health_score >= 80:
            report.append("✅ Layer is in good health!")
        return '\n'.join(report)
    def _generate_compliance_report(self, metrics: GovernanceMetrics, format: OutputFormat) -> str:
        """Generate compliance report."""
        if format != OutputFormat.MARKDOWN:
            compliance_data = {
                'overall_compliance': metrics.overall_compliance,
                'total_artifacts': metrics.total_artifacts,
                'total_errors': metrics.total_errors,
                'by_layer': {
                    layer_id: {
                        'compliance_rate': layer.compliance_rate,
                        'error_count': layer.error_count
                    }
                    for layer_id, layer in metrics.layer_metrics.items()
                    if layer.artifact_count > 0
                }
            }
            return json.dumps(compliance_data, indent=2)
        report = []
        report.append("# GL Compliance Report\n")
        report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z\n")
        # Overall compliance
        status = "✅ COMPLIANT" if metrics.overall_compliance >= 90 else \
                 "⚠️ NEEDS ATTENTION" if metrics.overall_compliance >= 70 else \
                 "❌ NON-COMPLIANT"
        report.append(f"## Overall Status: {status}\n")
        report.append(f"**Compliance Rate**: {metrics.overall_compliance:.1f}%\n")
        # Compliance by layer
        report.append("## Compliance by Layer\n")
        report.append("| Layer | Artifacts | Compliant | Rate | Status |")
        report.append("|-------|-----------|-----------|------|--------|")
        for layer_id in sorted(self.LAYER_INFO.keys()):
            layer = metrics.layer_metrics.get(layer_id)
            if layer and layer.artifact_count > 0:
                status = "✅" if layer.compliance_rate >= 90 else "⚠️" if layer.compliance_rate >= 70 else "❌"
                report.append(
                    f"| {layer_id} | {layer.artifact_count} | {layer.compliant_count} | "
                    f"{layer.compliance_rate:.1f}% | {status} |"
                )
        report.append("")
        # Non-compliant layers
        non_compliant = [
            (layer_id, layer) for layer_id, layer in metrics.layer_metrics.items()
            if layer.artifact_count > 0 and layer.compliance_rate < 90
        ]
        if non_compliant:
            report.append("## Layers Requiring Attention\n")
            for layer_id, layer in sorted(non_compliant, key=lambda x: x[1].compliance_rate):
                report.append(f"### {layer_id}")
                report.append(f"- Compliance: {layer.compliance_rate:.1f}%")
                report.append(f"- Errors: {layer.error_count}")
                report.append("- Action: Review and fix non-compliant artifacts")
                report.append("")
        return '\n'.join(report)
    def _generate_dashboard(self, metrics: GovernanceMetrics, format: OutputFormat) -> str:
        """Generate dashboard view."""
        if format == OutputFormat.HTML:
            return self._dashboard_to_html(metrics)
        elif format != OutputFormat.MARKDOWN:
            return self._to_json(metrics)
        report = []
        report.append("# GL Governance Dashboard\n")
        report.append(f"**Last Updated**: {datetime.utcnow().isoformat()}Z\n")
        # Key metrics cards
        report.append("## Key Metrics\n")
        report.append("```")
        report.append("┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")
        report.append("│   ARTIFACTS     │   COMPLIANCE    │     HEALTH      │     ERRORS      │")
        report.append(f"│      {metrics.total_artifacts:^5}       │     {metrics.overall_compliance:>5.1f}%      │     {metrics.overall_health:>5.1f}%      │      {metrics.total_errors:^5}       │")
        report.append("└─────────────────┴─────────────────┴─────────────────┴─────────────────┘")
        report.append("```\n")
        # Layer health bars
        report.append("## Layer Health\n")
        report.append("```")
        for layer_id in sorted(self.LAYER_INFO.keys()):
            layer = metrics.layer_metrics.get(layer_id)
            if layer and layer.artifact_count > 0:
                bar_length = int(layer.health_score / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                report.append(f"{layer_id}: [{bar}] {layer.health_score:.0f}%")
        report.append("```\n")
        # Quick stats
        report.append("## Quick Stats\n")
        report.append(f"- **Active Layers**: {metrics.total_layers}/7")
        report.append(f"- **Artifact Types**: {len(metrics.artifact_by_kind)}")
        report.append(f"- **Contributors**: {len(metrics.artifact_by_owner)}")
        report.append("")
        # Status indicators
        report.append("## Status Indicators\n")
        indicators = []
        if metrics.overall_compliance >= 90:
            indicators.append("🟢 Compliance: Excellent")
        elif metrics.overall_compliance >= 70:
            indicators.append("🟡 Compliance: Needs Attention")
        else:
            indicators.append("🔴 Compliance: Critical")
        if metrics.overall_health >= 80:
            indicators.append("🟢 Health: Good")
        elif metrics.overall_health >= 60:
            indicators.append("🟡 Health: Fair")
        else:
            indicators.append("🔴 Health: Poor")
        if metrics.total_errors == 0:
            indicators.append("🟢 Errors: None")
        elif metrics.total_errors <= 5:
            indicators.append("🟡 Errors: Some")
        else:
            indicators.append("🔴 Errors: Many")
        for indicator in indicators:
            report.append(f"- {indicator}")
        return '\n'.join(report)
    def _generate_health_report(self, metrics: GovernanceMetrics, format: OutputFormat) -> str:
        """Generate health report."""
        if format != OutputFormat.MARKDOWN:
            health_data = {
                'overall_health': metrics.overall_health,
                'by_layer': {
                    layer_id: layer.health_score
                    for layer_id, layer in metrics.layer_metrics.items()
                    if layer.artifact_count > 0
                }
            }
            return json.dumps(health_data, indent=2)
        report = []
        report.append("# GL Governance Health Report\n")
        report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z\n")
        # Overall health
        health_emoji = "🟢" if metrics.overall_health >= 80 else "🟡" if metrics.overall_health >= 60 else "🔴"
        report.append(f"## Overall Health: {health_emoji} {metrics.overall_health:.1f}%\n")
        # Health factors
        report.append("## Health Factors\n")
        report.append("| Factor | Impact | Status |")
        report.append("|--------|--------|--------|")
        error_impact = min(metrics.total_errors * 10, 50)
        error_status = "✅" if error_impact == 0 else "⚠️" if error_impact < 20 else "❌"
        report.append(f"| Validation Errors | -{error_impact}% | {error_status} |")
        warning_impact = min(metrics.total_warnings * 2, 20)
        warning_status = "✅" if warning_impact == 0 else "⚠️"
        report.append(f"| Warnings | -{warning_impact}% | {warning_status} |")
        report.append("")
        # Layer health details
        report.append("## Layer Health Details\n")
        for layer_id in sorted(self.LAYER_INFO.keys()):
            layer = metrics.layer_metrics.get(layer_id)
            info = self.LAYER_INFO[layer_id]
            if not layer or layer.artifact_count == 0:
                continue
            health_emoji = "🟢" if layer.health_score >= 80 else "🟡" if layer.health_score >= 60 else "🔴"
            report.append(f"### {layer_id}: {info['name']} {health_emoji}\n")
            report.append(f"**Health Score**: {layer.health_score:.0f}%\n")
            # Health breakdown
            issues = []
            if layer.error_count > 0:
                issues.append(f"- ❌ {layer.error_count} validation errors")
            if layer.warning_count > 0:
                issues.append(f"- ⚠️ {layer.warning_count} warnings")
            if layer.avg_age_days > 180:
                issues.append(f"- 📅 Artifacts are outdated (avg: {layer.avg_age_days:.0f} days)")
            if issues:
                report.append("**Issues**:")
                report.extend(issues)
            else:
                report.append("✅ No issues detected")
            report.append("")
        return '\n'.join(report)
    def _dashboard_to_html(self, metrics: GovernanceMetrics) -> str:
        """Generate HTML dashboard."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("  <title>GL Governance Dashboard</title>")
        html.append("  <style>")
        html.append("    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }")
        html.append("    .dashboard { max-width: 1200px; margin: 0 auto; }")
        html.append("    .header { text-align: center; margin-bottom: 30px; }")
        html.append("    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }")
        html.append("    .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }")
        html.append("    .metric-value { font-size: 2.5em; font-weight: bold; color: #333; }")
        html.append("    .metric-label { color: #666; margin-top: 5px; }")
        html.append("    .layer-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }")
        html.append("    .layer-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
        html.append("    .layer-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }")
        html.append("    .health-bar { height: 10px; background: #e0e0e0; border-radius: 5px; overflow: hidden; }")
        html.append("    .health-fill { height: 100%; transition: width 0.3s; }")
        html.append("    .health-good { background: #22c55e; }")
        html.append("    .health-fair { background: #eab308; }")
        html.append("    .health-poor { background: #ef4444; }")
        html.append("    .timestamp { text-align: center; color: #999; margin-top: 30px; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        html.append("  <div class='dashboard'>")
        html.append("    <div class='header'>")
        html.append("      <h1>🏛️ GL Governance Dashboard</h1>")
        html.append("    </div>")
        # Metrics cards
        html.append("    <div class='metrics-grid'>")
        html.append(f"      <div class='metric-card'><div class='metric-value'>{metrics.total_artifacts}</div><div class='metric-label'>Total Artifacts</div></div>")
        html.append(f"      <div class='metric-card'><div class='metric-value'>{metrics.overall_compliance:.1f}%</div><div class='metric-label'>Compliance</div></div>")
        html.append(f"      <div class='metric-card'><div class='metric-value'>{metrics.overall_health:.1f}%</div><div class='metric-label'>Health</div></div>")
        html.append(f"      <div class='metric-card'><div class='metric-value'>{metrics.total_errors}</div><div class='metric-label'>Errors</div></div>")
        html.append("    </div>")
        # Layer cards
        html.append("    <div class='layer-grid'>")
        for layer_id in sorted(self.LAYER_INFO.keys()):
            layer = metrics.layer_metrics.get(layer_id)
            info = self.LAYER_INFO[layer_id]
            if not layer or layer.artifact_count == 0:
                continue
            health_class = "health-good" if layer.health_score >= 80 else "health-fair" if layer.health_score >= 60 else "health-poor"
            html.append("      <div class='layer-card'>")
            html.append("        <div class='layer-header'>")
            html.append(f"          <strong>{layer_id}</strong>")
            html.append(f"          <span>{layer.artifact_count} artifacts</span>")
            html.append("        </div>")
            html.append(f"        <div>{info['name']}</div>")
            html.append(f"        <div class='health-bar'><div class='health-fill {health_class}' style='width: {layer.health_score}%'></div></div>")
            html.append(f"        <div style='margin-top: 10px; font-size: 0.9em; color: #666;'>Health: {layer.health_score:.0f}% | Compliance: {layer.compliance_rate:.0f}%</div>")
            html.append("      </div>")
        html.append("    </div>")
        html.append(f"    <div class='timestamp'>Last updated: {datetime.utcnow().isoformat()}Z</div>")
        html.append("  </div>")
        html.append("</body>")
        html.append("</html>")
        return '\n'.join(html)
    def _summary_to_html(self, metrics: GovernanceMetrics) -> str:
        """Generate HTML summary."""
        return self._dashboard_to_html(metrics)
    def _to_json(self, metrics: GovernanceMetrics) -> str:
        """Convert metrics to JSON."""
        data = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_artifacts': metrics.total_artifacts,
            'total_layers': metrics.total_layers,
            'overall_compliance': metrics.overall_compliance,
            'overall_health': metrics.overall_health,
            'total_errors': metrics.total_errors,
            'total_warnings': metrics.total_warnings,
            'by_layer': {
                layer_id: {
                    'name': layer.layer_name,
                    'artifact_count': layer.artifact_count,
                    'compliance_rate': layer.compliance_rate,
                    'health_score': layer.health_score,
                    'error_count': layer.error_count,
                    'warning_count': layer.warning_count,
                    'avg_age_days': layer.avg_age_days,
                    'kinds': layer.kinds,
                    'owners': layer.owners
                }
                for layer_id, layer in metrics.layer_metrics.items()
                if layer.artifact_count > 0
            },
            'by_kind': metrics.artifact_by_kind,
            'by_owner': metrics.artifact_by_owner
        }
        return json.dumps(data, indent=2)
    def _to_yaml(self, metrics: GovernanceMetrics) -> str:
        """Convert metrics to YAML."""
        data = json.loads(self._to_json(metrics))
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GL Reporter - Governance Layer Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--workspace', '-w', default='.',
                        help='Workspace path')
    parser.add_argument('--type', '-t', dest='report_type',
                        choices=[t.value for t in ReportType],
                        default='summary',
                        help='Report type')
    parser.add_argument('--format', '-f', dest='output_format',
                        choices=[f.value for f in OutputFormat],
                        default='markdown',
                        help='Output format')
    parser.add_argument('--layer', '-l',
                        help='Filter by layer (for layer report)')
    parser.add_argument('--output', '-o',
                        help='Output file path')
    args = parser.parse_args()
    # Create reporter
    reporter = GLReporter(args.workspace)
    # Generate report
    report_type = ReportType(args.report_type)
    output_format = OutputFormat(args.output_format)
    report = reporter.generate_report(report_type, output_format, args.layer)
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)
if __name__ == '__main__':
    main()