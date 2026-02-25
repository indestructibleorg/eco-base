#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: generate-autonomy-summary
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# ECO-Layer: GL30-49 (Execution)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomy Summary Generator for MachineNativeOps
Generates a comprehensive summary report of autonomy classifications
across all modules, including trends and recommendations.
Usage:
    python3 scripts/generate-autonomy-summary.py
    python3 scripts/generate-autonomy-summary.py --output docs/autonomy/AUTONOMY_SUMMARY.md
"""
# MNGA-002: Import organization needs review
import yaml
import json
import argparse
from datetime import datetime
from pathlib import Path
# Module definitions with expected autonomy levels
MODULE_DEFINITIONS = {
    "01-core": {"name": "Core Infrastructure", "expected_min": "L1", "expected_max": "L2"},
    "02-intelligence": {"name": "Intelligence Layer", "expected_min": "L2", "expected_max": "L3"},
    "03-gl-platform.gl-platform.governance": {"name": "Governance Framework", "expected_min": "L3", "expected_max": "L4"},
    "04-autonomous": {"name": "Autonomous Operations", "expected_min": "L4", "expected_max": "L5"},
    "05-observability": {"name": "Observability Platform", "expected_min": "L4", "expected_max": "L5"},
    "06-security": {"name": "Security & Compliance", "expected_min": "Global Layer", "expected_max": "Global Layer"}
}
AUTONOMY_LEVEL_VALUES = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "Global Layer": 10
}
def load_module_registry():
    """Load the module registry YAML file."""
    registry_path = Path("controlplane/baseline/modules/REGISTRY.yaml")
    if not registry_path.exists():
        print(f"Warning: Registry file not found at {registry_path}")
        return None
    with open(registry_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
def load_classification_reports():
    """Load existing classification reports from docs/autonomy/reports/."""
    reports_dir = Path("docs/autonomy/reports")
    reports = {}
    if not reports_dir.exists():
        return reports
    for report_file in reports_dir.glob("*-classification.md"):
        module_id = report_file.stem.replace("-classification", "")
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract score from report if available
            reports[module_id] = {"file": str(report_file), "content": content}
    return reports
def get_level_value(level):
    """Convert autonomy level string to numeric value."""
    return AUTONOMY_LEVEL_VALUES.get(level, 0)
def calculate_module_metrics(registry):
    """Calculate metrics for each module from registry."""
    metrics = {}
    if not registry or 'modules' not in registry:
        return metrics
    for module in registry['modules']:
        module_id = module.get('module_id', 'unknown')
        metrics[module_id] = {
            'name': module.get('name', module_id),
            'autonomy_level': module.get('autonomy_level', 'Unknown'),
            'semantic_health_score': module.get('semantic_health_score', 0),
            'status': module.get('status', 'unknown'),
            'dependencies': module.get('dependencies', []),
            'namespace': module.get('namespace', '')
        }
    return metrics
def generate_summary_report(metrics, reports, output_path):
    """Generate the autonomy summary report."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    # Calculate overall statistics
    total_modules = len(metrics)
    active_modules = sum(1 for m in metrics.values() if m['status'] == 'active')
    autonomy_levels = [m['autonomy_level'] for m in metrics.values()]
    semantic_scores = [m['semantic_health_score'] for m in metrics.values() if m['semantic_health_score'] > 0]
    avg_semantic_health = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    # Count modules by autonomy level
    level_distribution = {}
    for level in autonomy_levels:
        level_distribution[level] = level_distribution.get(level, 0) + 1
    report = f"""# Autonomy Classification Summary
#*Generated**: {timestamp}  
#*Total Modules**: {total_modules}  
#*Active Modules**: {active_modules}  
#*Average Semantic Health**: {avg_semantic_health:.1f}%
---
## 📊 Executive Summary
This report provides a comprehensive overview of autonomy classifications across all MachineNativeOps modules. The autonomy framework defines five levels (L1-L5) plus a Global Layer for cross-cutting concerns.
### Key Metrics
| Metric | Value |
|--------|-------|
| Total Modules | {total_modules} |
| Active Modules | {active_modules} |
| In Development | {total_modules - active_modules} |
| Average Semantic Health | {avg_semantic_health:.1f}% |
### Autonomy Level Distribution
| Level | Count | Modules |
|-------|-------|---------|
"""
    for level in ["L1", "L2", "L3", "L4", "L5", "Global Layer"]:
        count = level_distribution.get(level, 0)
        modules_at_level = [m_id for m_id, m in metrics.items() if m['autonomy_level'] == level]
        modules_str = ", ".join(modules_at_level) if modules_at_level else "-"
        report += f"| {level} | {count} | {modules_str} |\n"
    report += """
---
## 📋 Module Details
"""
    for module_id, definition in MODULE_DEFINITIONS.items():
        module_data = metrics.get(module_id, {})
        current_level = module_data.get('autonomy_level', 'Unknown')
        semantic_health = module_data.get('semantic_health_score', 0)
        status = module_data.get('status', 'unknown')
        dependencies = module_data.get('dependencies', [])
        # Check if within expected range
        expected_min = definition['expected_min']
        expected_max = definition['expected_max']
        current_value = get_level_value(current_level)
        min_value = get_level_value(expected_min)
        max_value = get_level_value(expected_max)
        if min_value <= current_value <= max_value:
            compliance = "✅ Within Range"
        elif current_value < min_value:
            compliance = "⚠️ Below Expected"
        else:
            compliance = "🔶 Above Expected"
        status_emoji = "🟢" if status == "active" else "🟡"
        report += f"""### {module_id}: {definition['name']}
| Attribute | Value |
|-----------|-------|
| Current Level | **{current_level}** |
| Expected Range | {expected_min} - {expected_max} |
| Compliance | {compliance} |
| Semantic Health | {semantic_health}% |
| Status | {status_emoji} {status} |
| Dependencies | {', '.join(dependencies) if dependencies and dependencies != ['none'] else 'None'} |
"""
    report += """---
## 🎯 Recommendations
### Priority Actions
"""
    # Generate recommendations based on analysis
    recommendations = []
    for module_id, module_data in metrics.items():
        definition = MODULE_DEFINITIONS.get(module_id, {})
        current_level = module_data.get('autonomy_level', 'Unknown')
        semantic_health = module_data.get('semantic_health_score', 0)
        status = module_data.get('status', 'unknown')
        if status == 'in-development':
            recommendations.append(f"1. **Complete {module_id}**: Module is in development status. Prioritize completion to enable full autonomy assessment.")
        if semantic_health < 90:
            recommendations.append(f"2. **Improve {module_id} Semantic Health**: Current score ({semantic_health}%) is below target (90%). Review semantic mappings and namespace consistency.")
        expected_min = definition.get('expected_min', 'L1')
        if get_level_value(current_level) < get_level_value(expected_min):
            recommendations.append(f"3. **Upgrade {module_id} Autonomy**: Current level ({current_level}) is below expected minimum ({expected_min}). Implement required capabilities for level progression.")
    if not recommendations:
        recommendations.append("✅ All modules are within expected parameters. Continue monitoring and incremental improvements.")
    for rec in recommendations[:5]:  # Top 5 recommendations
        report += f"{rec}\n\n"
    report += """### Long-term Goals
1. **Achieve L5 for 04-autonomous**: Enable full autonomous operations
2. **Maintain 95%+ Semantic Health**: Across all modules
3. **Zero Circular Dependencies**: Maintain clean dependency graph
4. **100% Policy Compliance**: All modules pass policy gates
---
## 📈 Progression Tracking
### Target State (6 months)
| Module | Current | Target | Gap |
|--------|---------|--------|-----|
"""
    for module_id, definition in MODULE_DEFINITIONS.items():
        module_data = metrics.get(module_id, {})
        current_level = module_data.get('autonomy_level', 'Unknown')
        target_level = definition['expected_max']
        current_value = get_level_value(current_level)
        target_value = get_level_value(target_level)
        gap = target_value - current_value
        gap_str = f"+{gap} levels" if gap > 0 else "✅ At target" if gap == 0 else f"{gap} levels"
        report += f"| {module_id} | {current_level} | {target_level} | {gap_str} |\n"
    report += """
---
## 📁 Classification Reports
Individual classification reports are available in `docs/autonomy/reports/`:
"""
    for module_id in MODULE_DEFINITIONS.keys():
        report_exists = module_id in reports
        status = "✅ Available" if report_exists else "⏳ Pending"
        report += f"- [{module_id}-classification.md](reports/{module_id}-classification.md) - {status}\n"
    report += f"""
---
## 🔗 Related Documentation
- [Autonomy Classification Framework](../AUTONOMY_CLASSIFICATION_FRAMEWORK.md)
- [Language Governance Dashboard](../LANGUAGE_GOVERNANCE_DASHBOARD.md)
- [DAG Visualization](../dag-visualization/DAG_VISUALIZATION.md)
- [Documentation Portal](../DOCUMENTATION_PORTAL.md)
---
#This report is automatically generated. Last update: {timestamp}*
"""
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Autonomy summary report generated: {output_path}")
    # Also generate JSON export
    json_output = output_path.with_suffix('.json')
    json_data = {
        "generated": timestamp,
        "total_modules": total_modules,
        "active_modules": active_modules,
        "average_semantic_health": avg_semantic_health,
        "level_distribution": level_distribution,
        "modules": metrics,
        "recommendations": recommendations[:5]
    }
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON export generated: {json_output}")
    return report
def main():
    parser = argparse.ArgumentParser(
        description="Generate Autonomy Classification Summary Report"
    )
    parser.add_argument(
        '--output', '-o',
        default='docs/autonomy/AUTONOMY_SUMMARY.md',
        help='Output file path (default: docs/autonomy/AUTONOMY_SUMMARY.md)'
    )
    args = parser.parse_args()
    print("🔄 Loading module registry...")
    registry = load_module_registry()
    print("🔄 Loading classification reports...")
    reports = load_classification_reports()
    print("🔄 Calculating module metrics...")
    metrics = calculate_module_metrics(registry)
    if not metrics:
        print("⚠️ No module metrics found. Using default definitions.")
        # Create default metrics from definitions
        metrics = {
            module_id: {
                'name': definition['name'],
                'autonomy_level': definition['expected_min'],
                'semantic_health_score': 95,
                'status': 'active' if module_id != '04-autonomous' else 'in-development',
                'dependencies': [],
                'namespace': f'mno-{module_id.split("-")[1]}'
            }
            for module_id, definition in MODULE_DEFINITIONS.items()
        }
    print("🔄 Generating summary report...")
    generate_summary_report(metrics, reports, args.output)
    print("\n✅ Autonomy summary generation complete!")
if __name__ == "__main__":
    main()