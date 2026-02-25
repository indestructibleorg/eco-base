# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: phase4_learning_system
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
第四階段：持續學習機制
"""
# MNGA-002: Import organization needs review
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Optional
import hashlib
class ContinuousLearningSystem:
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        self.learning_log = []
        self.operation_feedback = []
        self.knowledge_updates = []
        self.best_practices = []
        self.error_patterns = defaultdict(int)
    def load_knowledge_base(self):
        """載入知識庫"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 無法載入知識庫: {e}")
            return None
    def record_operation_feedback(self, operation: Dict, success: bool, feedback: str = ""):
        """記錄操作回饋"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'success': success,
            'feedback': feedback,
            'learned': False
        }
        self.operation_feedback.append(feedback_entry)
        # 分析回饋並學習
        if not success:
            self.analyze_failure_pattern(operation, feedback)
        print(f"📝 操作回饋已記錄: {'成功' if success else '失敗'}")
        return feedback_entry
    def analyze_failure_pattern(self, operation: Dict, feedback: str):
        """分析失敗模式"""
        operation_type = operation.get('operation', '')
        target = operation.get('target', '')
        # 記錄失敗模式
        pattern_key = f"{operation_type}:{Path(target).suffix if '.' in target else 'dir'}"
        self.error_patterns[pattern_key] += 1
        # 檢查是否是重複錯誤
        if self.error_patterns[pattern_key] >= 3:
            self.generate_best_practice(operation, feedback)
    def generate_best_practice(self, operation: Dict, error_feedback: str):
        """生成最佳實踐"""
        operation_type = operation.get('operation', '')
        target = operation.get('target', '')
        best_practice = {
            'id': hashlib.md5(f"{operation_type}:{target}:{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'target_pattern': self.extract_pattern(target),
            'error': error_feedback,
            'recommendation': self.generate_recommendation(operation, error_feedback),
            'usage_count': 0
        }
        self.best_practices.append(best_practice)
        print(f"💡 生成最佳實踐: {best_practice['id']}")
    def extract_pattern(self, target: str) -> str:
        """提取目標模式"""
        path = Path(target)
        # 提取檔案類型
        if '.' in path.name:
            return f"*{path.suffix}"
        # 提取目錄模式
        parts = path.parts
        if len(parts) >= 2:
            return f"*/{parts[-1]}"
        return str(path)
    def generate_recommendation(self, operation: Dict, error_feedback: str) -> str:
        """生成建議"""
        operation_type = operation.get('operation', '')
        operation.get('target', '')
        recommendations = {
            'delete': [
                "⚠️  刪除前務必確認檔案不再被其他檔案引用",
                "🔍 使用查詢系統檢查依賴關係",
                "💾 確認有適當的備份",
                "📋 在測試環境中先驗證影響"
            ],
            'modify': [
                "🔍 先查詢檔案上下文了解用途",
                "⚠️  檢查是否為關鍵檔案",
                "📝 記錄修改原因和影響",
                "✅ 修改後執行相關測試"
            ],
            'create': [
                "📁 確認檔案放置在正確的目錄",
                "🏷️  使用清晰的命名規範",
                "📝 添加適當的註釋和文件",
                "🔗 更新相關的依賴關係"
            ]
        }
        base_recommendations = recommendations.get(operation_type, ["🔍 詳細了解操作影響範圍"])
        # 根據錯誤訊息添加特定建議
        if "權限" in error_feedback or "permission" in error_feedback.lower():
            base_recommendations.append("🔐 檢查檔案權限設定")
        elif "不存在" in error_feedback or "not found" in error_feedback.lower():
            base_recommendations.append("📂 確認目標路徑正確性")
        elif "關鍵" in error_feedback or "critical" in error_feedback.lower():
            base_recommendations.append("⚠️  對關鍵檔案的操作需要額外審查")
        return "；".join(base_recommendations)
    def get_best_practice(self, operation_type: str, target: str) -> Optional[Dict]:
        """獲取最佳實踐建議"""
        target_pattern = self.extract_pattern(target)
        # 查找匹配的最佳實踐
        for practice in self.best_practices:
            if (practice['operation_type'] == operation_type and 
                self.pattern_matches(target_pattern, practice['target_pattern'])):
                practice['usage_count'] += 1
                return practice
        return None
    def pattern_matches(self, current_pattern: str, stored_pattern: str) -> bool:
        """檢查模式是否匹配"""
        # 簡單實現：精確匹配或萬用字元匹配
        if current_pattern == stored_pattern:
            return True
        if stored_pattern.startswith('*') and current_pattern.endswith(stored_pattern[1:]):
            return True
        if current_pattern.startswith('*') and stored_pattern.endswith(current_pattern[1:]):
            return True
        return False
    def update_knowledge_base(self, updates: Dict):
        """更新知識庫"""
        update_entry = {
            'timestamp': datetime.now().isoformat(),
            'updates': updates,
            'applied': False
        }
        try:
            # 應用更新
            for key, value in updates.items():
                if key in self.knowledge_base:
                    if isinstance(self.knowledge_base[key], dict) and isinstance(value, dict):
                        self.knowledge_base[key].update(value)
                    else:
                        self.knowledge_base[key] = value
                else:
                    self.knowledge_base[key] = value
            update_entry['applied'] = True
            self.knowledge_updates.append(update_entry)
            # 保存更新後的知識庫
            self.save_knowledge_base()
            print(f"✅ 知識庫已更新: {len(updates)} 項更新")
        except Exception as e:
            print(f"❌ 知識庫更新失敗: {e}")
            update_entry['error'] = str(e)
        return update_entry
    def save_knowledge_base(self, backup=True):
        """保存知識庫"""
        if backup:
            # 創建備份
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"knowledge_base_backup_{timestamp}.json"
            shutil.copy2(self.knowledge_base_path, backup_path)
            print(f"💾 知識庫備份已創建: {backup_path}")
        # 保存當前知識庫
        try:
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            print("✅ 知識庫已保存")
        except Exception as e:
            print(f"❌ 知識庫保存失敗: {e}")
    def scan_for_changes(self):
        """掃描檔案系統變化"""
        print("\n🔍 掃描檔案系統變化...")
        changes_detected = []
        current_files = set()
        current_dirs = set()
        # 掃描當前狀態
        for root, dirs, files in os.walk('.'):
            root_path = Path(root)
            for file in files:
                if not file.startswith('.'):
                    file_path = root_path / file
                    current_files.add(str(file_path))
            for dir_name in dirs:
                if not dir_name.startswith('.') and dir_name not in ['__pycache__', 'node_modules']:
                    dir_path = root_path / dir_name
                    current_dirs.add(str(dir_path))
        # 比較變化
        known_files = set(self.knowledge_base.get('files', {}).keys())
        known_dirs = set(self.knowledge_base.get('directories', {}).keys())
        new_files = current_files - known_files
        deleted_files = known_files - current_files
        new_dirs = current_dirs - known_dirs
        deleted_dirs = known_dirs - current_dirs
        if new_files:
            changes_detected.append({
                'type': 'new_files',
                'count': len(new_files),
                'items': list(new_files)[:10]
            })
        if deleted_files:
            changes_detected.append({
                'type': 'deleted_files',
                'count': len(deleted_files),
                'items': list(deleted_files)[:10]
            })
        if new_dirs:
            changes_detected.append({
                'type': 'new_directories',
                'count': len(new_dirs),
                'items': list(new_dirs)[:10]
            })
        if deleted_dirs:
            changes_detected.append({
                'type': 'deleted_directories',
                'count': len(deleted_dirs),
                'items': list(deleted_dirs)[:10]
            })
        if changes_detected:
            print(f"📊 檢測到 {len(changes_detected)} 種變化")
            for change in changes_detected:
                print(f"   - {change['type']}: {change['count']} 項")
            # 記錄學習日誌
            learning_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'change_detection',
                'changes': changes_detected,
                'action_required': True
            }
            self.learning_log.append(learning_entry)
        else:
            print("✅ 未檢測到變化")
        return changes_detected
    def generate_learning_report(self, filename='phase4_report.md'):
        """生成學習報告"""
        print(f"\n📝 生成學習報告到 {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 第四階段：持續學習機制報告\n\n")
            f.write(f"**生成日期**: {datetime.now().isoformat()}\n")
            f.write("**階段**: Phase 4 - Continuous Learning System\n\n")
            f.write("## 📊 學習統計\n\n")
            f.write(f"- **操作回饋次數**: {len(self.operation_feedback)}\n")
            f.write(f"- **成功操作**: {sum(1 for f in self.operation_feedback if f['success'])}\n")
            f.write(f"- **失敗操作**: {sum(1 for f in self.operation_feedback if not f['success'])}\n")
            f.write(f"- **知識庫更新次數**: {len(self.knowledge_updates)}\n")
            f.write(f"- **最佳實踐數量**: {len(self.best_practices)}\n")
            f.write(f"- **學習日誌條目**: {len(self.learning_log)}\n\n")
            f.write("## 🎯 最佳實踐\n\n")
            if self.best_practices:
                for practice in self.best_practices:
                    f.write(f"### 實踐 #{practice['id']}\n\n")
                    f.write(f"- **操作類型**: {practice['operation_type']}\n")
                    f.write(f"- **目標模式**: {practice['target_pattern']}\n")
                    f.write(f"- **錯誤**: {practice['error']}\n")
                    f.write(f"- **建議**: {practice['recommendation']}\n")
                    f.write(f"- **使用次數**: {practice['usage_count']}\n\n")
            else:
                f.write("暫無最佳實踐記錄\n\n")
            f.write("## 📈 失敗模式分析\n\n")
            if self.error_patterns:
                f.write("| 模式 | 次數 |\n")
                f.write("|------|------|\n")
                for pattern, count in sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"| {pattern} | {count} |\n")
            else:
                f.write("暫無失敗模式記錄\n\n")
            f.write("## 🔄 學習機制\n\n")
            f.write("### 1. 操作回饋循環\n\n")
            f.write("系統會記錄每次操作的結果和回饋：\n")
            f.write("- ✅ 成功操作：記錄成功模式和最佳實踐\n")
            f.write("- ❌ 失敗操作：分析失敗原因，生成預防措施\n\n")
            f.write("### 2. 知識庫自動更新\n\n")
            f.write("- 📊 定期掃描檔案系統變化\n")
            f.write("- 🔍 自動檢測新增/刪除的檔案和目錄\n")
            f.write("- 💾 自動創建備份並更新知識庫\n\n")
            f.write("### 3. 智能建議系統\n\n")
            f.write("- 💡 基於歷史操作生成最佳實踐\n")
            f.write("- ⚠️  識別重複錯誤模式\n")
            f.write("- 🎯 提供個性化的操作建議\n\n")
        print("✅ 學習報告已生成")
        return filename
    def get_system_health(self) -> Dict:
        """獲取系統健康狀態"""
        return {
            'knowledge_base_size': len(self.knowledge_base),
            'learning_active': len(self.learning_log) > 0,
            'best_practices_available': len(self.best_practices) > 0,
            'error_patterns_detected': len(self.error_patterns) > 0,
            'last_scan': self.learning_log[-1]['timestamp'] if self.learning_log else None,
            'total_learning_events': len(self.learning_log) + len(self.operation_feedback)
        }
def main():
    """主程式 - 演示持續學習機制"""
    print("="*60)
    print("🚀 第四階段：建立持續學習機制")
    print("="*60 + "\n")
    # 創建學習系統
    learning_system = ContinuousLearningSystem()
    if not learning_system.knowledge_base:
        print("❌ 無法載入知識庫，請先執行第一階段")
        return False
    print("✅ 學習系統初始化成功")
    # 演示學習功能
    print("\n" + "="*60)
    print("📊 演示學習功能")
    print("="*60)
    # 1. 記錄一些操作回饋
    print("\n📝 記錄操作回饋...")
    test_operations = [
        {'operation': 'read', 'target': 'test_file.txt'},
        {'operation': 'modify', 'target': 'config.yaml'},
        {'operation': 'delete', 'target': 'important_file.py'}
    ]
    for i, operation in enumerate(test_operations):
        success = i % 2 == 0  # 模擬一些失敗
        feedback = "操作成功完成" if success else "檔案不存在或權限不足"
        learning_system.record_operation_feedback(operation, success, feedback)
    # 2. 掃描變化
    print("\n🔍 掃描檔案系統變化...")
    learning_system.scan_for_changes()
    # 3. 生成報告
    report_file = learning_system.generate_learning_report('phase4_report.md')
    # 4. 顯示系統健康狀態
    health = learning_system.get_system_health()
    print("\n📊 系統健康狀態:")
    print(f"   知識庫大小: {health['knowledge_base_size']} 項")
    print(f"   學習活躍: {'是' if health['learning_active'] else '否'}")
    print(f"   最佳實踐: {len(learning_system.best_practices)} 項")
    print(f"   總學習事件: {health['total_learning_events']} 次")
    print("\n" + "="*60)
    print("✅ 第四階段完成！")
    print("="*60)
    print(f"📄 報告: {report_file}")
    print("="*60)
    return True
if __name__ == '__main__':
    main()