#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: check_ci_comments
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
CI 評論質量檢查工具
確保所有 CI 評論具有實際功能和高度輔助價值
"""
import re
import sys
from typing import Dict
class CommentQualityChecker:
    def __init__(self):
        self.quality_score = 0
        self.issues = []
        self.recommendations = []
    def check_comment(self, comment: str) -> Dict:
        """檢查評論質量並返回詳細報告"""
        self.quality_score = 0
        self.issues = []
        self.recommendations = []
        # 檢查具體性
        specificity_score = self._check_specificity(comment)
        self.quality_score += specificity_score
        if specificity_score < 25:
            self.issues.append("評論缺乏具體性（缺少文件和行號引用）")
            self.recommendations.append("在評論中引用具體的文件和行號，例如：src/auth.js:45")
        # 檢查可操作性
        actionable_score = self._check_actionable(comment)
        self.quality_score += actionable_score
        if actionable_score < 25:
            self.issues.append("評論缺乏可操作性（缺少修復建議）")
            self.recommendations.append("提供具體的修復建議或代碼示例")
        # 檢查上下文
        context_score = self._check_context(comment)
        self.quality_score += context_score
        if context_score < 25:
            self.issues.append("評論缺乏上下文說明（缺少影響和原因）")
            self.recommendations.append("說明問題的影響範圍和為什麼這是問題")
        # 檢查建設性
        constructive_score = self._check_constructive(comment)
        self.quality_score += constructive_score
        if constructive_score < 25:
            self.issues.append("評論語氣不夠建設性（可能過於批評）")
            self.recommendations.append("使用專業和尊重的語氣，專注於改進代碼質量")
        return {
            'score': self.quality_score,
            'passing': self.quality_score >= 75,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'analysis': {
                'specificity': specificity_score,
                'actionable': actionable_score,
                'context': context_score,
                'constructive': constructive_score
            }
        }
    def _check_specificity(self, comment: str) -> int:
        """檢查是否包含具體的文件和行號引用"""
        score = 0
        # 檢查文件引用
        if re.search(r'[\w/]+\.\w+', comment):
            score += 10
        # 檢查行號引用
        if re.search(r':\d+', comment):
            score += 10
        # 檢查函數或變量名引用
        if re.search(r'function\s+\w+|const\s+\w+|let\s+\w+|class\s+\w+', comment):
            score += 5
        return score
    def _check_actionable(self, comment: str) -> int:
        """檢查是否提供修復建議"""
        score = 0
        # 檢查建議性關鍵詞
        actionable_keywords = [
            '建議', '應該', '可以', '改進', '修改', '調整',
            '重構', '優化', '添加', '刪除', '替換',
            'suggestion', 'should', 'could', 'improve', 'refactor'
        ]
        for keyword in actionable_keywords:
            if keyword in comment.lower():
                score += 10
                break
        # 檢查是否有代碼示例
        if re.search(r'```[\s\S]*?```', comment):
            score += 15
        return score
    def _check_context(self, comment: str) -> int:
        """檢查是否提供上下文說明"""
        score = 0
        # 檢查上下文關鍵詞
        context_keywords = [
            '影響', '因為', '由於', '導致', '背景', '原因',
            '影響範圍', '風險', '問題', '安全',
            'impact', 'because', 'due to', 'risk', 'security'
        ]
        for keyword in context_keywords:
            if keyword in comment.lower():
                score += 10
                break
        # 檢查是否有原因說明
        if re.search(r'(導致|因為|由於|will cause|because|due to)', comment, re.IGNORECASE):
            score += 15
        return score
    def _check_constructive(self, comment: str) -> int:
        """檢查是否使用建設性語言"""
        score = 25  # 默認給滿分，除非發現問題
        # 檢查消極語言模式
        negative_patterns = [
            r'你的.*很差',
            r'這是錯的',
            r'寫得很差',
            r'不應該這樣寫',
            r'糟糕的代碼',
            r'廢話'
        ]
        for pattern in negative_patterns:
            if re.search(pattern, comment, re.IGNORECASE):
                score = 0
                break
        # 檢查是否有建設性批評
        constructive_patterns = [
            r'建議',
            r'可以改進',
            r'也許可以',
            r'考慮',
            r'suggest', 'could improve', 'consider'
        ]
        if score == 0:
            for pattern in constructive_patterns:
                if re.search(pattern, comment, re.IGNORECASE):
                    score = 10
                    break
        return score
def print_report(result: Dict):
    """打印評論質量報告"""
    print("=" * 60)
    print("CI 評論質量檢查報告")
    print("=" * 60)
    print(f"\n總體分數: {result['score']}/100")
    if result['passing']:
        print("狀態: ✅ 通過（質量良好）")
    else:
        print("狀態: ❌ 未通過（需要改進）")
    # 詳細分析
    print("\n--- 詳細分析 ---")
    analysis = result['analysis']
    print(f"具體性: {analysis['specificity']}/25")
    print(f"可操作性: {analysis['actionable']}/25")
    print(f"上下文: {analysis['context']}/25")
    print(f"建設性: {analysis['constructive']}/25")
    # 問題列表
    if result['issues']:
        print("\n--- 發現的問題 ---")
        for i, issue in enumerate(result['issues'], 1):
            print(f"{i}. {issue}")
    # 建議
    if result['recommendations']:
        print("\n--- 改進建議 ---")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
    print("\n" + "=" * 60)
def main():
    """主函數"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("使用方法:")
        print("  python check_ci_comments.py <評論內容>")
        print("  echo '評論內容' | python check_ci_comments.py")
        print("\n評論質量標準:")
        print("  - 總分 ≥ 75 分為通過")
        print("  - 具體性: 引用文件和行號")
        print("  - 可操作性: 提供修復建議")
        print("  - 上下文: 說明影響和原因")
        print("  - 建設性: 使用專業語氣")
        return
    # 讀取評論內容
    if not sys.stdin.isatty():
        comment = sys.stdin.read()
    elif len(sys.argv) > 1:
        comment = ' '.join(sys.argv[1:])
    else:
        print("錯誤: 請提供評論內容")
        print("使用 --help 查看使用說明")
        sys.exit(1)
    # 檢查評論
    checker = CommentQualityChecker()
    result = checker.check_comment(comment)
    # 打印報告
    print_report(result)
    # 根據結果返回退出碼
    sys.exit(0 if result['passing'] else 1)
if __name__ == "__main__":
    main()