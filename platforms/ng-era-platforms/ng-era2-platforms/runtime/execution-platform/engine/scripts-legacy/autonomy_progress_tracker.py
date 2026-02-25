#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: autonomy-progress-tracker
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# ECO-Layer: GL30-49 (Execution)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomy Progress Tracker for MachineNativeOps
Tracks autonomy level progression over time, stores historical data,
and generates trend analysis reports.
Usage:
    python3 scripts/autonomy-progress-tracker.py --record
    python3 scripts/autonomy-progress-tracker.py --report
    python3 scripts/autonomy-progress-tracker.py --trends
"""
# MNGA-002: Import organization needs review
import yaml
import json
import argparse
from datetime import datetime
from pathlib import Path
class AutonomyProgressTracker:
    """Tracks and analyzes autonomy level progression over time."""
    HISTORY_FILE = "docs/autonomy/progress-history.json"
    REPORT_FILE = "docs/autonomy/AUTONOMY_PROGRESS_REPORT.md"
    AUTONOMY_LEVEL_VALUES = {
        "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, 
        "L1-L2": 1.5, "L2-L3": 2.5, "L3-L4": 3.5, "L4-L5": 4.5,
        "Global Layer": 10
    }
    TARGET_LEVELS = {
        "01-core": {"target": "L2", "target_value": 2},
        "02-intelligence": {"target": "L3", "target_value": 3},
        "03-gl-platform.gl-platform.governance": {"target": "L4", "target_value": 4},
        "04-autonomous": {"target": "L5", "target_value": 5},
        "05-observability": {"target": "L5", "target_value": 5},
        "06-security": {"target": "Global Layer", "target_value": 10}
    }
    def __init__(self):
        self.history = self.load_history()
    def load_history(self):
        """Load historical data from JSON file."""
        history_path = Path(self.HISTORY_FILE)
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"records": [], "metadata": {"created": datetime.utcnow().isoformat()}}
    def save_history(self):
        """Save historical data to JSON file."""
        history_path = Path(self.HISTORY_FILE)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"✅ History saved to: {history_path}")
    def load_current_state(self):
        """Load current module state from registry."""
        registry_path = Path("controlplane/baseline/modules/REGISTRY.yaml")
        if not registry_path.exists():
            print("⚠️ Registry not found, using default values")
            return self._get_default_state()
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = yaml.safe_load(f)
        state = {}
        for module in registry.get('modules', []):
            module_id = module.get('module_id')
            state[module_id] = {
                'autonomy_level': module.get('autonomy_level', 'Unknown'),
                'semantic_health_score': module.get('semantic_health_score', 0),
                'status': module.get('status', 'unknown')
            }
        return state
    def _get_default_state(self):
        """Return default state for modules."""
        return {
            "01-core": {"autonomy_level": "L1-L2", "semantic_health_score": 100, "status": "active"},
            "02-intelligence": {"autonomy_level": "L2-L3", "semantic_health_score": 100, "status": "active"},
            "03-gl-platform.gl-platform.governance": {"autonomy_level": "L3-L4", "semantic_health_score": 100, "status": "active"},
            "04-autonomous": {"autonomy_level": "L4-L5", "semantic_health_score": 85, "status": "in-development"},
            "05-observability": {"autonomy_level": "L4-L5", "semantic_health_score": 100, "status": "active"},
            "06-security": {"autonomy_level": "Global Layer", "semantic_health_score": 100, "status": "active"}
        }
    def get_level_value(self, level):
        """Convert autonomy level to numeric value."""
        return self.AUTONOMY_LEVEL_VALUES.get(level, 0)
    def record_snapshot(self):
        """Record current state as a historical snapshot."""
        current_state = self.load_current_state()
        timestamp = datetime.utcnow().isoformat()
        snapshot = {
            "timestamp": timestamp,
            "modules": {}
        }
        for module_id, data in current_state.items():
            level = data['autonomy_level']
            snapshot["modules"][module_id] = {
                "autonomy_level": level,
                "autonomy_value": self.get_level_value(level),
                "semantic_health_score": data['semantic_health_score'],
                "status": data['status']
            }
        # Calculate aggregate metrics
        values = [m['autonomy_value'] for m in snapshot["modules"].values()]
        health_scores = [m['semantic_health_score'] for m in snapshot["modules"].values()]
        snapshot["aggregate"] = {
            "average_autonomy": sum(values) / len(values) if values else 0,
            "average_health": sum(health_scores) / len(health_scores) if health_scores else 0,
            "total_modules": len(snapshot["modules"]),
            "active_modules": sum(1 for m in snapshot["modules"].values() if m['status'] == 'active')
        }
        self.history["records"].append(snapshot)
        self.save_history()
        print(f"✅ Snapshot recorded at {timestamp}")
        print(f"   Average Autonomy: {snapshot['aggregate']['average_autonomy']:.2f}")
        print(f"   Average Health: {snapshot['aggregate']['average_health']:.1f}%")
        return snapshot
    def analyze_trends(self):
        """Analyze trends from historical data."""
        records = self.history.get("records", [])
        if len(records) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 records for trend analysis",
                "records_count": len(records)
            }
        # Get first and last records
        first = records[0]
        last = records[-1]
        trends = {
            "period": {
                "start": first["timestamp"],
                "end": last["timestamp"],
                "records_count": len(records)
            },
            "modules": {},
            "aggregate": {}
        }
        # Analyze each module
        for module_id in last["modules"].keys():
            if module_id in first["modules"]:
                first_value = first["modules"][module_id]["autonomy_value"]
                last_value = last["modules"][module_id]["autonomy_value"]
                first_health = first["modules"][module_id]["semantic_health_score"]
                last_health = last["modules"][module_id]["semantic_health_score"]
                target = self.TARGET_LEVELS.get(module_id, {})
                target_value = target.get("target_value", 5)
                trends["modules"][module_id] = {
                    "autonomy_change": last_value - first_value,
                    "health_change": last_health - first_health,
                    "current_value": last_value,
                    "target_value": target_value,
                    "progress_to_target": (last_value / target_value * 100) if target_value > 0 else 100,
                    "trend": "improving" if last_value > first_value else "stable" if last_value == first_value else "declining"
                }
        # Aggregate trends
        first_avg = first["aggregate"]["average_autonomy"]
        last_avg = last["aggregate"]["average_autonomy"]
        trends["aggregate"] = {
            "autonomy_change": last_avg - first_avg,
            "health_change": last["aggregate"]["average_health"] - first["aggregate"]["average_health"],
            "overall_trend": "improving" if last_avg > first_avg else "stable" if last_avg == first_avg else "declining"
        }
        return trends
    def generate_report(self, output_path=None):
        """Generate a comprehensive progress report."""
        output_path = output_path or self.REPORT_FILE
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        current_state = self.load_current_state()
        trends = self.analyze_trends()
        records = self.history.get("records", [])
        report = f"""# Autonomy Progress Report
#*Generated**: {timestamp}  
#*Historical Records**: {len(records)}
---
## 📊 Executive Summary
"""
        if trends.get("status") == "insufficient_data":
            report += f"""
> ⚠️ **Note**: {trends['message']}. Record more snapshots to enable trend analysis.
"""
        else:
            overall_trend = trends["aggregate"]["overall_trend"]
            trend_emoji = "📈" if overall_trend == "improving" else "📊" if overall_trend == "stable" else "📉"
            report += f"""
| Metric | Value | Trend |
|--------|-------|-------|
| Overall Trend | {overall_trend.title()} | {trend_emoji} |
| Autonomy Change | {trends['aggregate']['autonomy_change']:+.2f} | - |
| Health Change | {trends['aggregate']['health_change']:+.1f}% | - |
| Analysis Period | {trends['period']['start'][:10]} to {trends['period']['end'][:10]} | - |
"""
        report += """---
## 📋 Current Module Status
| Module | Current Level | Target | Progress | Health | Status |
|--------|---------------|--------|----------|--------|--------|
"""
        for module_id, data in current_state.items():
            target = self.TARGET_LEVELS.get(module_id, {})
            target_level = target.get("target", "N/A")
            target_value = target.get("target_value", 5)
            current_value = self.get_level_value(data['autonomy_level'])
            progress = (current_value / target_value * 100) if target_value > 0 else 100
            progress_bar = self._get_progress_bar(progress)
            status_emoji = "🟢" if data['status'] == 'active' else "🟡"
            report += f"| {module_id} | {data['autonomy_level']} | {target_level} | {progress_bar} {progress:.0f}% | {data['semantic_health_score']}% | {status_emoji} {data['status']} |\n"
        report += """
---
## 🎯 Progress Toward Targets
"""
        for module_id, target in self.TARGET_LEVELS.items():
            data = current_state.get(module_id, {})
            current_value = self.get_level_value(data.get('autonomy_level', 'L1'))
            target_value = target['target_value']
            gap = target_value - current_value
            if gap <= 0:
                status = "✅ Target Achieved"
            elif gap <= 1:
                status = "🔶 Near Target"
            else:
                status = f"⏳ {gap:.1f} levels to go"
            report += f"### {module_id}\n"
            report += f"- **Current**: {data.get('autonomy_level', 'Unknown')}\n"
            report += f"- **Target**: {target['target']}\n"
            report += f"- **Status**: {status}\n\n"
        if len(records) >= 2:
            report += """---
## 📈 Trend Analysis
"""
            for module_id, trend_data in trends.get("modules", {}).items():
                trend_emoji = "📈" if trend_data['trend'] == "improving" else "📊" if trend_data['trend'] == "stable" else "📉"
                report += f"- **{module_id}**: {trend_emoji} {trend_data['trend'].title()} "
                report += f"(Change: {trend_data['autonomy_change']:+.2f})\n"
        report += """
---
## 📅 Historical Snapshots
"""
        if records:
            report += "| Date | Avg Autonomy | Avg Health | Active Modules |\n"
            report += "|------|--------------|------------|----------------|\n"
            for record in records[-10:]:  # Last 10 records
                date = record['timestamp'][:10]
                avg_auto = record['aggregate']['average_autonomy']
                avg_health = record['aggregate']['average_health']
                active = record['aggregate']['active_modules']
                report += f"| {date} | {avg_auto:.2f} | {avg_health:.1f}% | {active} |\n"
        else:
            report += "> No historical snapshots recorded yet. Run `--record` to create snapshots.\n"
        report += """
---
## 🔧 Recommendations
"""
        recommendations = self._generate_recommendations(current_state, trends)
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        report += """
---
## 🔗 Related Documentation
- [Autonomy Classification Framework](AUTONOMY_CLASSIFICATION_FRAMEWORK.md)
- [Autonomy Summary](autonomy/AUTONOMY_SUMMARY.md)
- [Language Governance Dashboard](LANGUAGE_GOVERNANCE_DASHBOARD.md)
---
#This report is automatically generated by the Autonomy Progress Tracker.*
"""
        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Progress report generated: {output_path}")
        return report
    def _get_progress_bar(self, percentage):
        """Generate a text-based progress bar."""
        filled = int(percentage / 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty
    def _generate_recommendations(self, current_state, trends):
        """Generate recommendations based on current state and trends."""
        recommendations = []
        # Check for modules below target
        for module_id, target in self.TARGET_LEVELS.items():
            data = current_state.get(module_id, {})
            current_value = self.get_level_value(data.get('autonomy_level', 'L1'))
            target_value = target['target_value']
            if current_value < target_value:
                gap = target_value - current_value
                recommendations.append(
                    f"**{module_id}**: Increase autonomy level by {gap:.1f} to reach target {target['target']}"
                )
        # Check for low semantic health
        for module_id, data in current_state.items():
            if data.get('semantic_health_score', 100) < 90:
                recommendations.append(
                    f"**{module_id}**: Improve semantic health from {data['semantic_health_score']}% to 90%+"
                )
        # Check for in-development modules
        for module_id, data in current_state.items():
            if data.get('status') == 'in-development':
                recommendations.append(
                    f"**{module_id}**: Complete development to enable full autonomy assessment"
                )
        if not recommendations:
            recommendations.append("✅ All modules are on track! Continue monitoring and incremental improvements.")
        return recommendations[:5]  # Top 5 recommendations
def main():
    parser = argparse.ArgumentParser(
        description="Autonomy Progress Tracker for MachineNativeOps"
    )
    parser.add_argument(
        '--record', '-r',
        action='store_true',
        help='Record current state as a historical snapshot'
    )
    parser.add_argument(
        '--report', '-p',
        action='store_true',
        help='Generate progress report'
    )
    parser.add_argument(
        '--trends', '-t',
        action='store_true',
        help='Analyze and display trends'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path for report'
    )
    args = parser.parse_args()
    tracker = AutonomyProgressTracker()
    # Default to report if no action specified
    if not (args.record or args.report or args.trends):
        args.report = True
    if args.record:
        print("📸 Recording snapshot...")
        tracker.record_snapshot()
    if args.trends:
        print("📈 Analyzing trends...")
        trends = tracker.analyze_trends()
        print(json.dumps(trends, indent=2))
    if args.report:
        print("📊 Generating report...")
        tracker.generate_report(args.output)
if __name__ == "__main__":
    main()