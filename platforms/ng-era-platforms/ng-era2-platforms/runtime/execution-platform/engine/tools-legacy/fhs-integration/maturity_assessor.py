# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: maturity_assessor
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
FHS Integration Maturity Assessor
自動評估 workspace 組件的成熟度，判斷是否可以整合到 FHS 目錄
Usage:
    python3 maturity_assessor.py [component_name]
    python3 maturity_assessor.py --all
    python3 maturity_assessor.py repository-understanding --verbose
"""
# MNGA-002: Import organization needs review
import os
import sys
import json
import yaml
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
class MaturityAssessor:
    """評估組件成熟度並生成整合建議"""
    def __init__(self, repo_root: str = None):
        self.repo_root = repo_root or os.getcwd()
        self.workspace_tools = os.path.join(self.repo_root, "workspace", "tools")
        self.criteria_file = os.path.join(
            self.workspace_tools, "fhs-integration", "maturity-criteria.yaml"
        )
        self.criteria = self._load_criteria()
    def _load_criteria(self) -> dict:
        """加載成熟度標準"""
        if os.path.exists(self.criteria_file):
            with open(self.criteria_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    def assess_component(self, component_name: str, verbose: bool = False) -> Dict:
        """評估單個組件的成熟度"""
        component_path = os.path.join(self.workspace_tools, component_name)
        if not os.path.exists(component_path):
            return {
                "error": f"Component {component_name} not found",
                "path": component_path
            }
        scores = {}
        total_score = 0
        max_score = 100
        # 1. 代碼品質評估 (30分)
        code_quality = self._assess_code_quality(component_path, verbose)
        scores["code_quality"] = code_quality
        total_score += code_quality["total"]
        # 2. 穩定性評估 (25分)
        stability = self._assess_stability(component_path, component_name, verbose)
        scores["stability"] = stability
        total_score += stability["total"]
        # 3. 使用情況評估 (20分)
        usage = self._assess_usage(component_path, verbose)
        scores["usage"] = usage
        total_score += usage["total"]
        # 4. 依賴管理評估 (15分)
        dependencies = self._assess_dependencies(component_path, verbose)
        scores["dependencies"] = dependencies
        total_score += dependencies["total"]
        # 5. 維護狀態評估 (10分)
        maintenance = self._assess_maintenance(component_path, component_name, verbose)
        scores["maintenance"] = maintenance
        total_score += maintenance["total"]
        # 計算成熟度等級
        maturity_level = self._determine_maturity_level(total_score)
        return {
            "component": component_name,
            "path": component_path,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(total_score / max_score * 100, 2),
            "maturity_level": maturity_level,
            "scores": scores,
            "recommendation": self._generate_recommendation(total_score, maturity_level),
            "assessed_at": datetime.now().isoformat()
        }
    def _assess_code_quality(self, path: str, verbose: bool) -> Dict:
        """評估代碼品質 (30分)"""
        scores = {}
        # 測試覆蓋率 (10分)
        test_score = self._check_test_coverage(path, verbose)
        scores["test_coverage"] = test_score
        # 文檔完整性 (10分)
        doc_score = self._check_documentation(path, verbose)
        scores["documentation"] = doc_score
        # 代碼風格 (10分)
        style_score = self._check_code_style(path, verbose)
        scores["code_style"] = style_score
        total = test_score + doc_score + style_score
        return {
            "total": total,
            "max": 30,
            "details": scores
        }
    def _check_test_coverage(self, path: str, verbose: bool) -> int:
        """檢查測試覆蓋率"""
        # 檢查是否有測試文件
        test_files = list(Path(path).rglob("test_*.py")) + \
                     list(Path(path).rglob("*_test.py"))
        # 檢查是否有 tests 目錄
        tests_dir = os.path.join(path, "tests")
        has_tests_dir = os.path.exists(tests_dir)
        if len(test_files) == 0 and not has_tests_dir:
            return 0  # 無測試
        elif len(test_files) < 3 and not has_tests_dir:
            return 4  # 少量測試
        elif len(test_files) < 5 or has_tests_dir:
            return 6  # 中等測試覆蓋
        else:
            return 10  # 良好測試覆蓋
    def _check_documentation(self, path: str, verbose: bool) -> int:
        """檢查文檔完整性"""
        score = 0
        # 檢查 README
        readme = os.path.join(path, "README.md")
        if os.path.exists(readme):
            size = os.path.getsize(readme)
            if size > 1000:
                score += 5  # 詳細 README
            else:
                score += 3  # 基本 README
        # 檢查其他文檔
        doc_files = list(Path(path).glob("*.md"))
        if len(doc_files) > 3:
            score += 3
        elif len(doc_files) > 1:
            score += 2
        # 檢查 API 文檔或範例
        examples_dir = os.path.join(path, "examples")
        docs_dir = os.path.join(path, "docs")
        if os.path.exists(examples_dir) or os.path.exists(docs_dir):
            score += 2
        return min(score, 10)
    def _check_code_style(self, path: str, verbose: bool) -> int:
        """檢查代碼風格"""
        # 檢查是否有 .pylintrc, .flake8, pyproject.toml 等
        style_files = [
            ".pylintrc", ".flake8", "pyproject.toml",
            ".black.toml", "setup.cfg"
        ]
        for style_file in style_files:
            if os.path.exists(os.path.join(path, style_file)):
                return 8  # 有代碼風格配置
        # 檢查代碼是否符合基本 PEP8
        py_files = list(Path(path).glob("*.py"))
        if len(py_files) > 0:
            return 5  # 有 Python 代碼
        return 3  # 基本分數
    def _assess_stability(self, path: str, component_name: str, verbose: bool) -> Dict:
        """評估穩定性 (25分)"""
        scores = {}
        # API 穩定性 (10分)
        api_score = self._check_api_stability(path, component_name, verbose)
        scores["api_stability"] = api_score
        # Bug 率 (8分)
        bug_score = self._check_bug_rate(component_name, verbose)
        scores["bug_rate"] = bug_score
        # 發布頻率 (7分)
        release_score = self._check_release_frequency(path, verbose)
        scores["release_frequency"] = release_score
        total = api_score + bug_score + release_score
        return {
            "total": total,
            "max": 25,
            "details": scores
        }
    def _check_api_stability(self, path: str, component_name: str, verbose: bool) -> int:
        """檢查 API 穩定性（通過 git 歷史）"""
        try:
            # 檢查最近 6 個月的主要變更
            six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            cmd = f"git log --since='{six_months_ago}' --oneline -- {path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.repo_root)
            output = result.stdout.strip()
            if not output:
                commit_count = 0
            else:
                commits = output.split('\n')
                commit_count = len([c for c in commits if c])
            # 較少的提交可能意味著更穩定
            if commit_count < 10:
                return 10  # 非常穩定
            elif commit_count < 30:
                return 7   # 穩定
            elif commit_count < 60:
                return 5   # 中等
            else:
                return 3   # 頻繁變更
        except Exception as e:
            logger.error(f"Error: {e}")
            return 5  # 默認中等
    def _check_bug_rate(self, component_name: str, verbose: bool) -> int:
        """檢查 Bug 率（這裡用簡化的啟發式方法）"""
        # 實際應用中可以整合 issue tracker
        # 這裡用 git log 中的 "fix", "bug" 關鍵字
        try:
            cmd = f"git log --since='1 month ago' --oneline --grep='fix' --grep='bug' --grep='patch' -- workspace/tools/{component_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.repo_root)
            output = result.stdout.strip()
            if not output:
                fix_commits = 0
            else:
                fix_commits = len([line for line in output.split('\n') if line])
            if fix_commits <= 1:
                return 8  # 低 bug 率
            elif fix_commits <= 3:
                return 6  # 中等
            elif fix_commits <= 5:
                return 4  # 較高
            else:
                return 2  # 高 bug 率
        except:
            return 6  # 默認中等
    def _check_release_frequency(self, path: str, verbose: bool) -> int:
        """檢查發布頻率"""
        # 檢查是否有版本標記或 CHANGELOG
        changelog = os.path.join(path, "CHANGELOG.md")
        version_file = os.path.join(path, "VERSION")
        if os.path.exists(changelog):
            return 7
        elif os.path.exists(version_file):
            return 5
        else:
            return 3
    def _assess_usage(self, path: str, verbose: bool) -> Dict:
        """評估使用情況 (20分)"""
        scores = {}
        # 生產環境使用 (10分)
        prod_score = self._check_production_usage(path, verbose)
        scores["production_usage"] = prod_score
        # 用戶採用率 (10分)
        adoption_score = self._check_user_adoption(path, verbose)
        scores["user_adoption"] = adoption_score
        total = prod_score + adoption_score
        return {
            "total": total,
            "max": 20,
            "details": scores
        }
    def _check_production_usage(self, path: str, verbose: bool) -> int:
        """檢查生產環境使用"""
        # 檢查是否有 systemd service 文件
        service_files = list(Path(path).glob("*.service"))
        if service_files:
            return 10
        # 檢查是否在 etc/systemd 中引用
        etc_systemd = os.path.join(self.repo_root, "etc", "systemd")
        if os.path.exists(etc_systemd):
            component_name = os.path.basename(path)
            for service_file in Path(etc_systemd).glob("*.service"):
                content = service_file.read_text()
                if component_name in content:
                    return 10
        return 5  # 默認中等使用
    def _check_user_adoption(self, path: str, verbose: bool) -> int:
        """檢查用戶採用率"""
        # 這裡用啟發式方法：檢查腳本引用
        scripts_dir = os.path.join(self.repo_root, "scripts")
        component_name = os.path.basename(path)
        references = 0
        if os.path.exists(scripts_dir):
            for script_file in Path(scripts_dir).rglob("*.sh"):
                try:
                    content = script_file.read_text()
                    if component_name in content:
                        references += 1
                except:
                    pass
        if references >= 5:
            return 10
        elif references >= 3:
            return 7
        elif references >= 1:
            return 5
        else:
            return 3
    def _assess_dependencies(self, path: str, verbose: bool) -> Dict:
        """評估依賴管理 (15分)"""
        scores = {}
        # 依賴穩定性 (8分)
        stability_score = self._check_dependency_stability(path, verbose)
        scores["dependency_stability"] = stability_score
        # 依賴數量 (7分)
        count_score = self._check_dependency_count(path, verbose)
        scores["dependency_count"] = count_score
        total = stability_score + count_score
        return {
            "total": total,
            "max": 15,
            "details": scores
        }
    def _check_dependency_stability(self, path: str, verbose: bool) -> int:
        """檢查依賴穩定性"""
        requirements = os.path.join(path, "requirements.txt")
        if not os.path.exists(requirements):
            return 5  # 無依賴或未明確聲明
        with open(requirements, 'r') as f:
            deps = f.readlines()
        # 檢查是否有版本固定
        pinned = sum(1 for dep in deps if '==' in dep)
        total = len([d for d in deps if d.strip() and not d.startswith('#')])
        if total == 0:
            return 8
        ratio = pinned / total
        if ratio > 0.8:
            return 8  # 大部分依賴固定版本
        elif ratio > 0.5:
            return 6
        else:
            return 4
    def _check_dependency_count(self, path: str, verbose: bool) -> int:
        """檢查依賴數量"""
        requirements = os.path.join(path, "requirements.txt")
        if not os.path.exists(requirements):
            return 7  # 無外部依賴
        with open(requirements, 'r') as f:
            deps = [d for d in f.readlines() if d.strip() and not d.startswith('#')]
        count = len(deps)
        if count < 5:
            return 7
        elif count < 10:
            return 5
        elif count < 20:
            return 3
        else:
            return 2
    def _assess_maintenance(self, path: str, component_name: str, verbose: bool) -> Dict:
        """評估維護狀態 (10分)"""
        scores = {}
        # 積極維護 (10分)
        maintenance_score = self._check_active_maintenance(component_name, verbose)
        scores["active_maintenance"] = maintenance_score
        return {
            "total": maintenance_score,
            "max": 10,
            "details": scores
        }
    def _check_active_maintenance(self, component_name: str, verbose: bool) -> int:
        """檢查積極維護狀態"""
        try:
            # 檢查最近一個月的提交
            cmd = f"git log --since='1 month ago' --oneline -- workspace/tools/{component_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.repo_root)
            output = result.stdout.strip()
            if not output:
                recent_commits = 0
            else:
                recent_commits = len([line for line in output.split('\n') if line])
            if recent_commits >= 4:
                return 10  # 每週都有提交
            elif recent_commits >= 1:
                return 7   # 每月有提交
            else:
                # 檢查最近 3 個月
                cmd_3m = f"git log --since='3 months ago' --oneline -- workspace/tools/{component_name}"
                result_3m = subprocess.run(cmd_3m, shell=True, capture_output=True, text=True, cwd=self.repo_root)
                output_3m = result_3m.stdout.strip()
                if not output_3m:
                    commits_3m = 0
                else:
                    commits_3m = len([line for line in output_3m.split('\n') if line])
                if commits_3m >= 1:
                    return 5  # 每季度有提交
                else:
                    return 3  # 長期無更新
        except:
            return 5  # 默認中等
    def _determine_maturity_level(self, score: int) -> str:
        """根據分數確定成熟度等級"""
        if score >= 81:
            return "production"
        elif score >= 61:
            return "stable"
        elif score >= 41:
            return "development"
        else:
            return "experimental"
    def _generate_recommendation(self, score: int, level: str) -> Dict:
        """生成整合建議"""
        recommendations = {
            "production": {
                "action": "應該整合到 FHS 目錄",
                "priority": "high",
                "steps": [
                    "創建 bin/ 或 sbin/ 命令包裝器",
                    "遷移 Python 模組到 lib/",
                    "配置文件移至 etc/",
                    "更新文檔",
                    "創建遷移 PR"
                ]
            },
            "stable": {
                "action": "可選擇性整合到 FHS 目錄",
                "priority": "medium",
                "steps": [
                    "評估是否適合生產部署",
                    "改進測試覆蓋率",
                    "完善文檔",
                    "考慮創建 FHS 整合計劃"
                ]
            },
            "development": {
                "action": "繼續在 workspace/ 開發",
                "priority": "low",
                "steps": [
                    "提高代碼品質",
                    "增加測試覆蓋率",
                    "完善文檔",
                    "穩定 API"
                ]
            },
            "experimental": {
                "action": "保持在 workspace/ 實驗開發",
                "priority": "none",
                "steps": [
                    "專注於功能開發",
                    "建立基本文檔",
                    "考慮添加測試",
                    "評估是否繼續投入"
                ]
            }
        }
        return recommendations.get(level, recommendations["experimental"])
    def assess_all_components(self, verbose: bool = False) -> List[Dict]:
        """評估所有組件"""
        components = []
        if not os.path.exists(self.workspace_tools):
            return []
        for item in os.listdir(self.workspace_tools):
            item_path = os.path.join(self.workspace_tools, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                assessment = self.assess_component(item, verbose)
                components.append(assessment)
        # 按分數排序
        components.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        return components
    def generate_report(self, assessments: List[Dict]) -> str:
        """生成評估報告"""
        report = []
        report.append("=" * 80)
        report.append("FHS Integration Maturity Assessment Report")
        report.append("=" * 80)
        report.append("")
        for assessment in assessments:
            if "error" in assessment:
                continue
            report.append(f"Component: {assessment['component']}")
            report.append(f"Score: {assessment['total_score']}/{assessment['max_score']} ({assessment['percentage']}%)")
            report.append(f"Maturity Level: {assessment['maturity_level'].upper()}")
            report.append(f"Recommendation: {assessment['recommendation']['action']}")
            report.append("-" * 80)
        report.append("")
        report.append("Summary:")
        # 統計各成熟度等級
        levels = {}
        for assessment in assessments:
            if "error" not in assessment:
                level = assessment['maturity_level']
                levels[level] = levels.get(level, 0) + 1
        for level, count in sorted(levels.items()):
            report.append(f"  {level}: {count} component(s)")
        return "\n".join(report)
def main():
    """主函數"""
    import argparse
    parser = argparse.ArgumentParser(description="FHS Integration Maturity Assessor")
    parser.add_argument("component", nargs="?", help="Component name to assess")
    parser.add_argument("--all", action="store_true", help="Assess all components")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--repo-root", help="Repository root path")
    args = parser.parse_args()
    assessor = MaturityAssessor(repo_root=args.repo_root)
    if args.all:
        assessments = assessor.assess_all_components(verbose=args.verbose)
        if args.json:
            print(json.dumps(assessments, indent=2))
        else:
            print(assessor.generate_report(assessments))
    elif args.app.kubernetes.io/component: assessment = assessor.assess_component(args.component, verbose=args.verbose)
        if args.json:
            print(json.dumps(assessment, indent=2))
        else:
            if "error" in assessment:
                print(f"Error: {assessment['error']}")
                sys.exit(1)
            print("=" * 80)
            print(f"Component: {assessment['component']}")
            print("=" * 80)
            print(f"Total Score: {assessment['total_score']}/{assessment['max_score']} ({assessment['percentage']}%)")
            print(f"Maturity Level: {assessment['maturity_level'].upper()}")
            print("")
            print("Detailed Scores:")
            for category, data in assessment['scores'].items():
                print(f"  {category}: {data['total']}/{data['max']}")
            print("")
            print(f"Recommendation: {assessment['recommendation']['action']}")
            print(f"Priority: {assessment['recommendation']['priority']}")
            print("")
            print("Next Steps:")
            for i, step in enumerate(assessment['recommendation']['steps'], 1):
                print(f"  {i}. {step}")
    else:
        parser.print_help()
if __name__ == "__main__":
    main()
