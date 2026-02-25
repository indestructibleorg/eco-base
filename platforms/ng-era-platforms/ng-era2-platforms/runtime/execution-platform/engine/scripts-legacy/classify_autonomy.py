#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: classify-autonomy
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Autonomy Level Classification Script
Classifies modules and components according to L1-L5 autonomy framework
"""
# MNGA-002: Import organization needs review
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
# Scoring criteria weights
CRITERIA_WEIGHTS = {
    'decision_making': 30,
    'self_configuration': 25,
    'monitoring_response': 20,
    'learning_adaptation': 15,
    'error_handling': 10
}
# Level thresholds
LEVEL_THRESHOLDS = [
    (0, 20, 'L1'),
    (21, 40, 'L2'),
    (41, 60, 'L3'),
    (61, 80, 'L4'),
    (81, 100, 'L5')
]
def load_module_manifest(module_id: str) -> Dict:
    """Load module manifest"""
    manifest_path = Path(f"controlplane/baseline/modules/{module_id}/module-manifest.yaml")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Module manifest not found: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
def score_component(app.kubernetes.io/component: Dict) -> Tuple[int, Dict[str, int]]:
    """
    Score a component based on autonomy criteria
    Returns (total_score, breakdown)
    """
    # For demonstration, we'll use heuristics based on component type
    # In production, this would analyze actual code or metadata
    component_type = component.get('type', 'unknown')
    component.get('name', 'unknown')
    # Default scores (can be overridden by metadata)
    scores = {
        'decision_making': 0,
        'self_configuration': 0,
        'monitoring_response': 0,
        'learning_adaptation': 0,
        'error_handling': 0
    }
    # Heuristic scoring based on component type
    if component_type == 'service':
        scores['decision_making'] = 12  # L2-L3
        scores['self_configuration'] = 10  # L2
        scores['monitoring_response'] = 8  # L2
        scores['learning_adaptation'] = 3  # L1
        scores['error_handling'] = 4  # L2
    elif component_type == 'library':
        scores['decision_making'] = 6  # L1-L2
        scores['self_configuration'] = 5  # L1
        scores['monitoring_response'] = 4  # L1
        scores['learning_adaptation'] = 0  # L1
        scores['error_handling'] = 2  # L1
    elif component_type == 'tool':
        scores['decision_making'] = 9  # L2
        scores['self_configuration'] = 8  # L2
        scores['monitoring_response'] = 6  # L2
        scores['learning_adaptation'] = 2  # L1
        scores['error_handling'] = 3  # L1-L2
    elif component_type == 'framework':
        scores['decision_making'] = 18  # L3
        scores['self_configuration'] = 15  # L3
        scores['monitoring_response'] = 12  # L3
        scores['learning_adaptation'] = 9  # L3
        scores['error_handling'] = 6  # L3
    # Check for autonomy-enhancing keywords in description
    description = component.get('description', '').lower()
    if 'ai' in description or 'ml' in description or 'machine learning' in description:
        scores['decision_making'] += 6
        scores['learning_adaptation'] += 6
    if 'autonomous' in description or 'self-' in description:
        scores['self_configuration'] += 5
        scores['learning_adaptation'] += 3
    if 'monitor' in description or 'observ' in description:
        scores['monitoring_response'] += 4
    if 'heal' in description or 'recover' in description:
        scores['error_handling'] += 3
    # Ensure scores don't exceed maximum for each criterion
    for key in scores:
        scores[key] = min(scores[key], CRITERIA_WEIGHTS[key])
    total_score = sum(scores.values())
    return total_score, scores
def score_to_level(score: int) -> str:
    """Convert score to autonomy level"""
    for min_score, max_score, level in LEVEL_THRESHOLDS:
        if min_score <= score <= max_score:
            return level
    return 'Unknown'
def classify_module(module_id: str) -> Dict:
    """Classify a module and all its components"""
    manifest = load_module_manifest(module_id)
    components = manifest.get('components', [])
    component_results = []
    total_weighted_score = 0
    total_weight = 0
    for component in components:
        score, breakdown = score_component(component)
        level = score_to_level(score)
        # Weight based on component type (core components weighted more)
        component_type = component.get('type', 'unknown')
        weight = 3 if 'core' in component.get('name', '').lower() else 2 if component_type == 'service' else 1
        component_results.append({
            'name': component.get('name', 'unknown'),
            'type': component_type,
            'level': level,
            'score': score,
            'weight': weight,
            'breakdown': breakdown
        })
        total_weighted_score += score * weight
        total_weight += weight
    # Calculate module-level score
    module_score = int(total_weighted_score / total_weight) if total_weight > 0 else 0
    module_level = score_to_level(module_score)
    # Get declared autonomy level from manifest
    declared_level = manifest.get('autonomy_level', 'Unknown')
    # Generate recommendations
    recommendations = generate_recommendations(module_score, component_results)
    return {
        'module_id': module_id,
        'module_name': manifest.get('module_name', 'Unknown'),
        'declared_autonomy_level': declared_level,
        'calculated_autonomy_level': module_level,
        'total_score': module_score,
        'max_score': 100,
        'component_count': len(components),
        'components': component_results,
        'recommendations': recommendations,
        'classified_at': datetime.now().isoformat()
    }
def generate_recommendations(score: int, components: List[Dict]) -> List[str]:
    """Generate improvement recommendations"""
    recommendations = []
    current_level = score_to_level(score)
    # Analyze component scores to find weak areas
    total_scores = {key: 0 for key in CRITERIA_WEIGHTS}
    for comp in components:
        for key, value in comp['breakdown'].items():
            total_scores[key] += value
    avg_scores = {key: total_scores[key] / len(components) if components and len(components) > 0 else 0 for key in total_scores}
    # Find lowest scoring areas
    sorted_criteria = sorted(avg_scores.items(), key=lambda x: x[1])
    # Generate specific recommendations
    for criterion, avg_score in sorted_criteria[:2]:  # Focus on 2 lowest
        max_score = CRITERIA_WEIGHTS[criterion]
        if avg_score < max_score * 0.5:  # Less than 50% of max
            if criterion == 'decision_making':
                recommendations.append("Enhance decision-making with contextual awareness and rule-based logic")
            elif criterion == 'self_configuration':
                recommendations.append("Implement adaptive configuration management based on environment")
            elif criterion == 'monitoring_response':
                recommendations.append("Add proactive monitoring and automated response capabilities")
            elif criterion == 'learning_adaptation':
                recommendations.append("Integrate pattern learning and adaptive behavior")
            elif criterion == 'error_handling':
                recommendations.append("Improve error recovery with intelligent retry strategies")
    # Level-specific recommendations
    if current_level == 'L1':
        recommendations.append("Start with basic conditional logic and environment-based configuration")
    elif current_level == 'L2':
        recommendations.append("Focus on context-aware operations and proactive monitoring")
    elif current_level == 'L3':
        recommendations.append("Integrate ML/AI capabilities for predictive operations")
    elif current_level == 'L4':
        recommendations.append("Enable full autonomous behavior and multi-agent collaboration")
    return recommendations
def generate_report(classification: Dict, output_format: str = 'json') -> str:
    """Generate classification report"""
    if output_format == 'json':
        return json.dumps(classification, indent=2)
    elif output_format == 'yaml':
        return yaml.dump(classification, default_flow_style=False)
    elif output_format == 'markdown':
        md = f"""# Autonomy Classification Report: {classification['module_id']}
#*Module**: {classification['module_name']}  
#*Declared Level**: {classification['declared_autonomy_level']}  
#*Calculated Level**: {classification['calculated_autonomy_level']}  
#*Score**: {classification['total_score']}/100  
#*Classified**: {classification['classified_at']}
---
## Component Breakdown
| Component | Type | Level | Score | Weight |
|-----------|------|-------|-------|--------|
"""
        for comp in classification['components']:
            md += f"| {comp['name']} | {comp['type']} | {comp['level']} | {comp['score']} | {comp['weight']} |\n"
        md += "\n## Recommendations\n\n"
        for i, rec in enumerate(classification['recommendations'], 1):
            md += f"{i}. {rec}\n"
        return md
    return str(classification)
def main():
    parser = argparse.ArgumentParser(description='Classify module autonomy levels')
    parser.add_argument('--module', '-m', required=True, help='Module ID (e.g., 01-core)')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'yaml', 'markdown'], default='json',
                       help='Output format')
    args = parser.parse_args()
    try:
        # Classify module
        classification = classify_module(args.module)
        # Generate report
        report = generate_report(classification, args.format)
        # Output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Classification report written to: {args.output}")
        else:
            print(report)
        # Print summary
        print("\n📊 Summary:")
        print(f"   Module: {classification['module_id']}")
        print(f"   Declared: {classification['declared_autonomy_level']}")
        print(f"   Calculated: {classification['calculated_autonomy_level']}")
        print(f"   Score: {classification['total_score']}/100")
        print(f"   Components: {classification['component_count']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    return 0
if __name__ == "__main__":
    exit(main())
