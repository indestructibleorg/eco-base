# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: fhs_automation_master
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
FHS Integration Master Automation
主控自動化腳本：評估所有組件並執行適當的整合
Usage:
    python3 fhs_automation_master.py --assess-all
    python3 fhs_automation_master.py --auto-integrate --threshold 85
"""
# MNGA-002: Import organization needs review
import os
import sys
import json
import argparse
from datetime import datetime
# 添加當前目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maturity_assessor import MaturityAssessor
from fhs_integrator import FHSIntegrator
class FHSAutomationMaster:
    """FHS 整合主控自動化"""
    def __init__(self, repo_root: str = None, threshold: int = 80):
        self.repo_root = repo_root or self._find_repo_root()
        self.threshold = threshold
        self.assessor = MaturityAssessor(repo_root=self.repo_root)
    def _find_repo_root(self) -> str:
        """尋找 git 倉庫根目錄"""
        current = os.path.abspath(os.path.dirname(__file__))
        while os.path.dirname(current) != current:  # Cross-platform check for root
            if os.path.exists(os.path.join(current, '.git')):
                return current
            current = os.path.dirname(current)
        return os.getcwd()
    def run_full_assessment(self) -> dict:
        """執行完整評估"""
        print("=" * 80)
        print("FHS Integration Master Automation")
        print("=" * 80)
        print(f"Repository: {self.repo_root}")
        print(f"Integration Threshold: {self.threshold}")
        print("")
        # 評估所有組件
        print("Assessing all components...")
        assessments = self.assessor.assess_all_components(verbose=False)
        # 分類組件
        categorized = self._categorize_components(assessments)
        # 生成報告
        report = self._generate_full_report(categorized)
        return {
            "timestamp": datetime.now().isoformat(),
            "assessments": assessments,
            "categorized": categorized,
            "report": report
        }
    def _categorize_components(self, assessments: list) -> dict:
        """分類組件"""
        categories = {
            "ready_for_integration": [],    # >= threshold
            "stable": [],                    # 61-79
            "development": [],               # 41-60
            "experimental": []               # < 41
        }
        for assessment in assessments:
            if "error" in assessment:
                continue
            score = assessment.get("total_score", 0)
            if score >= self.threshold:
                categories["ready_for_integration"].append(assessment)
            elif score >= 61:
                categories["stable"].append(assessment)
            elif score >= 41:
                categories["development"].append(assessment)
            else:
                categories["experimental"].append(assessment)
        return categories
    def _generate_full_report(self, categorized: dict) -> str:
        """生成完整報告"""
        lines = []
        lines.append("=" * 80)
        lines.append("FHS INTEGRATION STATUS REPORT")
        lines.append("=" * 80)
        lines.append("")
        # 就緒整合
        if categorized["ready_for_integration"]:
            lines.append(f"✓ READY FOR FHS INTEGRATION ({self.threshold}+ points)")
            lines.append("-" * 80)
            for comp in categorized["ready_for_integration"]:
                lines.append(f"  • {comp['component']:30s} Score: {comp['total_score']:3d}/100 ({comp['percentage']:5.1f}%)")
            lines.append("")
        # 穩定階段
        if categorized["stable"]:
            lines.append("⚡ STABLE - Consider Integration (61-79 points)")
            lines.append("-" * 80)
            for comp in categorized["stable"]:
                lines.append(f"  • {comp['component']:30s} Score: {comp['total_score']:3d}/100 ({comp['percentage']:5.1f}%)")
            lines.append("")
        # 開發階段
        if categorized["development"]:
            lines.append("🔧 IN DEVELOPMENT (41-60 points)")
            lines.append("-" * 80)
            for comp in categorized["development"]:
                lines.append(f"  • {comp['component']:30s} Score: {comp['total_score']:3d}/100 ({comp['percentage']:5.1f}%)")
            lines.append("")
        # 實驗階段
        if categorized["experimental"]:
            lines.append("🧪 EXPERIMENTAL (< 41 points)")
            lines.append("-" * 80)
            for comp in categorized["experimental"]:
                lines.append(f"  • {comp['component']:30s} Score: {comp['total_score']:3d}/100 ({comp['percentage']:5.1f}%)")
            lines.append("")
        # 統計
        lines.append("=" * 80)
        lines.append("SUMMARY")
        lines.append("=" * 80)
        total = sum(len(v) for v in categorized.values())
        lines.append(f"Total Components: {total}")
        lines.append(f"  Ready for Integration: {len(categorized['ready_for_integration'])}")
        lines.append(f"  Stable: {len(categorized['stable'])}")
        lines.append(f"  In Development: {len(categorized['development'])}")
        lines.append(f"  Experimental: {len(categorized['experimental'])}")
        lines.append("")
        return "\n".join(lines)
    def auto_integrate(self, dry_run: bool = True) -> dict:
        """自動整合達到閾值的組件"""
        result = self.run_full_assessment()
        ready_components = result["categorized"]["ready_for_integration"]
        if not ready_components:
            print("No components ready for integration.")
            return {
                "integrated": [],
                "message": "No components meet the threshold"
            }
        print(f"\nFound {len(ready_components)} component(s) ready for integration:")
        for comp in ready_components:
            print(f"  • {comp['component']} ({comp['total_score']}/100)")
        print("")
        integrated = []
        for comp in ready_components:
            component_name = comp['component']
            score = comp['total_score']
            print(f"\nIntegrating {component_name}...")
            print("-" * 80)
            integrator = FHSIntegrator(repo_root=self.repo_root, dry_run=dry_run)
            integration_result = integrator.integrate_component(component_name, score)
            integrated.append({
                "component": component_name,
                "score": score,
                "result": integration_result
            })
        return {
            "integrated": integrated,
            "total": len(integrated),
            "dry_run": dry_run
        }
    def save_assessment_report(self, output_path: str = None):
        """保存評估報告"""
        if output_path is None:
            output_path = os.path.join(
                self.repo_root,
                "docs",
                "fhs-integration",
                f"assessment-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
        result = self.run_full_assessment()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Assessment report saved: {output_path}")
        # 也保存一個可讀的文本報告
        text_output = output_path.replace('.json', '.txt')
        with open(text_output, 'w') as f:
            f.write(result["report"])
        print(f"✓ Text report saved: {text_output}")
        return output_path
def main():
    parser = argparse.ArgumentParser(description="FHS Integration Master Automation")
    parser.add_argument("--assess-all", action="store_true", 
                       help="Assess all components and generate report")
    parser.add_argument("--auto-integrate", action="store_true",
                       help="Automatically integrate components that meet threshold")
    parser.add_argument("--threshold", type=int, default=80,
                       help="Minimum score for integration (default: 80)")
    parser.add_argument("--execute", action="store_true",
                       help="Execute integration (default is dry-run)")
    parser.add_argument("--save-report", action="store_true",
                       help="Save assessment report to file")
    parser.add_argument("--repo-root", help="Repository root path")
    args = parser.parse_args()
    if not any([args.assess_all, args.auto_integrate, args.save_report]):
        parser.print_help()
        return
    master = FHSAutomationMaster(
        repo_root=args.repo_root,
        threshold=args.threshold
    )
    if args.assess_all:
        result = master.run_full_assessment()
        print(result["report"])
    if args.auto_integrate:
        dry_run = not args.execute
        result = master.auto_integrate(dry_run=dry_run)
        print("\n" + "=" * 80)
        print("AUTO-INTEGRATION SUMMARY")
        print("=" * 80)
        print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
        print(f"Components processed: {result['total']}")
        if args.execute:
            print("\n✓ Integration completed!")
            print("\nNext steps:")
            print("  1. Review the migrated files")
            print("  2. Test the new FHS commands")
            print("  3. Update documentation")
            print("  4. Commit changes to git")
        else:
            print("\nTo execute integration, add --execute flag")
    if args.save_report:
        master.save_assessment_report()
if __name__ == "__main__":
    main()
