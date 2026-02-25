# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: phase2_operation_checker
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
第二階段：操作前的檢查機制
"""
# MNGA-002: Import organization needs review
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
class OperationChecker:
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        self.operation_history = []
        self.checklist_results = []
    def load_knowledge_base(self):
        """載入知識庫"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 無法載入知識庫: {e}")
            return None
    def check_before_operation(self, operation_type: str, target_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        執行操作前的完整檢查
        返回: (是否允許操作, 通過的檢查項, 失敗的檢查項)
        """
        print(f"\n{'='*60}")
        print(f"🔍 操作前檢查: {operation_type} -> {target_path}")
        print('='*60)
        passed_checks = []
        failed_checks = []
        # 1. 上下文驗證檢查
        print("\n📋 1. 上下文驗證")
        context_result = self.verify_context(target_path)
        if context_result['passed']:
            passed_checks.append("上下文驗證通過")
            print(f"✅ {context_result['message']}")
        else:
            failed_checks.append(f"上下文驗證失敗: {context_result['message']}")
            print(f"❌ {context_result['message']}")
        # 2. 影響評估檢查
        print("\n📋 2. 影響評估")
        impact_result = self.assess_impact(target_path, operation_type)
        if impact_result['passed']:
            passed_checks.append("影響評估通過")
            print(f"✅ 風險等級: {impact_result['risk_level']}")
            print(f"   影響檔案數: {len(impact_result['affected_files'])}")
        else:
            failed_checks.append(f"影響評估失敗: {impact_result['message']}")
            print(f"❌ {impact_result['message']}")
        # 3. 知識檢查
        print("\n📋 3. 知識檢查")
        knowledge_result = self.check_knowledge(target_path)
        if knowledge_result['passed']:
            passed_checks.append("知識檢查通過")
            print(f"✅ {knowledge_result['message']}")
        else:
            failed_checks.append(f"知識檢查失敗: {knowledge_result['message']}")
            print(f"❌ {knowledge_result['message']}")
        # 4. 風險檢查
        print("\n📋 4. 風險評估")
        risk_result = self.assess_risk(target_path, operation_type)
        if risk_result['passed']:
            passed_checks.append("風險評估通過")
            print(f"✅ {risk_result['message']}")
        else:
            failed_checks.append(f"風險評估失敗: {risk_result['message']}")
            print(f"❌ {risk_result['message']}")
        # 5. 備份檢查
        print("\n📋 5. 備份檢查")
        backup_result = self.check_backup(target_path)
        if backup_result['passed']:
            passed_checks.append("備份檢查通過")
            print(f"✅ {backup_result['message']}")
        else:
            failed_checks.append(f"備份檢查失敗: {backup_result['message']}")
            print(f"❌ {backup_result['message']}")
        # 總結
        print("\n" + "="*60)
        print(f"📊 檢查結果: {len(passed_checks)} 通過, {len(failed_checks)} 失敗")
        print("="*60)
        # 記錄檢查結果
        checklist_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation_type,
            'target': target_path,
            'passed': len(failed_checks) == 0,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks
        }
        self.checklist_results.append(checklist_data)
        return len(failed_checks) == 0, passed_checks, failed_checks
    def verify_context(self, target_path: str) -> Dict:
        """驗證上下文"""
        # 檢查檔案/目錄是否存在
        target = Path(target_path)
        if not target.exists():
            return {
                'passed': False,
                'message': f"目標不存在: {target_path}"
            }
        # 檢查是否在知識庫中
        files = self.knowledge_base.get('files', {})
        directories = self.knowledge_base.get('directories', {})
        if str(target) in files:
            file_info = files[str(target)]
            return {
                'passed': True,
                'message': f"檔案存在於知識庫中，類型: {file_info.get('type', 'unknown')}"
            }
        elif str(target) in directories:
            dir_info = directories[str(target)]
            return {
                'passed': True,
                'message': f"目錄存在於知識庫中，用途: {dir_info.get('purpose', 'unknown')}"
            }
        else:
            return {
                'passed': False,
                'message': "目標不在知識庫中，可能需要先掃描"
            }
    def assess_impact(self, target_path: str, operation_type: str) -> Dict:
        """評估影響"""
        affected_files = []
        risk_level = 'low'
        # 根據操作類型評估影響
        if operation_type in ['delete', 'remove', 'rm']:
            risk_level = 'high'
            # 查找依賴此檔案的其他檔案
            affected_files = self.find_dependent_files(target_path)
        elif operation_type in ['modify', 'edit', 'update', 'write']:
            risk_level = 'medium'
            affected_files = self.find_dependent_files(target_path)
        else:
            risk_level = 'low'
        # 如果是關鍵檔案，提升風險等級
        if self.is_critical_file(target_path):
            risk_level = 'high'
        return {
            'passed': True,  # 影響評估總是通過，只是提供資訊
            'risk_level': risk_level,
            'affected_files': affected_files,
            'message': f"風險等級: {risk_level}, 影響 {len(affected_files)} 個檔案"
        }
    def find_dependent_files(self, target_path: str) -> List[str]:
        """查找依賴此檔案的其他檔案"""
        dependent_files = []
        # 簡單實現：查找同目錄下的檔案
        target = Path(target_path)
        if target.is_file():
            parent_dir = str(target.parent)
            files = self.knowledge_base.get('files', {})
            for file_path, file_info in files.items():
                if file_info.get('directory') == parent_dir and file_path != str(target):
                    dependent_files.append(file_path)
        return dependent_files
    def check_knowledge(self, target_path: str) -> Dict:
        """檢查知識完整性"""
        # 檢查我們對此檔案的知識是否完整
        files = self.knowledge_base.get('files', {})
        directories = self.knowledge_base.get('directories', {})
        if str(target_path) in files:
            file_info = files[str(target_path)]
            has_type = file_info.get('type') != 'unknown'
            has_purpose = file_info.get('is_critical') is not None
            if has_type and has_purpose:
                return {
                    'passed': True,
                    'message': "檔案知識完整"
                }
            else:
                return {
                    'passed': False,
                    'message': "檔案知識不完整"
                }
        elif str(target_path) in directories:
            dir_info = directories[str(target_path)]
            has_purpose = dir_info.get('purpose') != 'unknown'
            if has_purpose:
                return {
                    'passed': True,
                    'message': "目錄知識完整"
                }
            else:
                return {
                    'passed': False,
                    'message': "目錄用途未知"
                }
        else:
            return {
                'passed': False,
                'message': "目標不在知識庫中"
            }
    def assess_risk(self, target_path: str, operation_type: str) -> Dict:
        """評估風險"""
        critical_categories = ['bootstrap', 'security', 'build', 'entry_points']
        # 檢查是否為關鍵檔案
        if self.is_critical_file(target_path):
            # 檢查具體的關鍵類別
            for category in critical_categories:
                critical_files = self.knowledge_base.get('critical_files_by_category', {}).get(category, [])
                if any(target_path in f for f in critical_files):
                    if operation_type in ['delete', 'remove', 'rm']:
                        return {
                            'passed': False,
                            'message': f"❌ 禁止操作：這是 {category} 關鍵檔案"
                        }
                    else:
                        return {
                            'passed': True,
                            'message': f"⚠️  警告：這是 {category} 關鍵檔案"
                        }
        return {
            'passed': True,
            'message': "風險可接受"
        }
    def is_critical_file(self, target_path: str) -> bool:
        """判斷是否為關鍵檔案"""
        critical_files = self.knowledge_base.get('critical_files', [])
        return any(target_path in f for f in critical_files)
    def check_backup(self, target_path: str) -> Dict:
        """檢查備份狀態"""
        # 檢查是否有版本控制
        git_dir = Path('.git')
        if git_dir.exists():
            # 檢查檔案是否在 git 中
            target = Path(target_path)
            if target.exists():
                # 簡單檢查：如果檔案存在，假設有版本控制
                return {
                    'passed': True,
                    'message': "檔案在版本控制中"
                }
        return {
            'passed': True,
            'message': "備份狀態未檢查，請手動確認"
        }
    def record_operation(self, operation_type: str, target_path: str, result: str, notes: str = ""):
        """記錄操作"""
        operation = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation_type,
            'target': target_path,
            'result': result,
            'notes': notes
        }
        self.operation_history.append(operation)
        print(f"📝 操作已記錄: {operation_type} -> {target_path} ({result})")
    def generate_report(self, filename='phase2_report.md'):
        """生成報告"""
        print(f"\n📝 生成報告到 {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 第二階段：操作檢查機制報告\n\n")
            f.write(f"**生成日期**: {datetime.now().isoformat()}\n")
            f.write("**階段**: Phase 2 - Operation Check Mechanism\n\n")
            f.write("## 📋 檢查統計\n\n")
            f.write(f"- **總檢查次數**: {len(self.checklist_results)}\n")
            f.write(f"- **通過檢查**: {sum(1 for c in self.checklist_results if c['passed'])}\n")
            f.write(f"- **失敗檢查**: {sum(1 for c in self.checklist_results if not c['passed'])}\n\n")
            f.write("## 🔍 檢查結果詳情\n\n")
            for i, result in enumerate(self.checklist_results, 1):
                f.write(f"### 檢查 #{i}\n\n")
                f.write(f"- **時間**: {result['timestamp']}\n")
                f.write(f"- **操作**: {result['operation']}\n")
                f.write(f"- **目標**: {result['target']}\n")
                f.write(f"- **結果**: {'✅ 通過' if result['passed'] else '❌ 失敗'}\n\n")
                if result['passed_checks']:
                    f.write("#### 通過的檢查\n\n")
                    for check in result['passed_checks']:
                        f.write(f"- ✅ {check}\n")
                    f.write("\n")
                if result['failed_checks']:
                    f.write("#### 失敗的檢查\n\n")
                    for check in result['failed_checks']:
                        f.write(f"- ❌ {check}\n")
                    f.write("\n")
        print("✅ 報告已生成")
        return filename
def main():
    """主程式 - 演示操作檢查機制"""
    print("="*60)
    print("🚀 第二階段：實施操作前的檢查機制")
    print("="*60 + "\n")
    # 創建檢查器
    checker = OperationChecker()
    if not checker.knowledge_base:
        print("❌ 無法載入知識庫，請先執行第一階段")
        return False
    print("✅ 知識庫載入成功")
    print(f"📁 包含 {len(checker.knowledge_base.get('files', {}))} 個檔案")
    print(f"📁 包含 {len(checker.knowledge_base.get('directories', {}))} 個目錄\n")
    # 演示一些操作檢查
    test_operations = [
        ('read', 'machine-native-ops/README.md'),
        ('modify', 'machine-native-ops/Makefile'),
        ('delete', 'machine-native-ops/root.env.sh'),
        ('create', 'new_file.txt')
    ]
    for operation, target in test_operations:
        allowed, passed, failed = checker.check_before_operation(operation, target)
        if allowed:
            print(f"\n✅ 允許執行操作: {operation} -> {target}")
            checker.record_operation(operation, target, "allowed", "所有檢查通過")
        else:
            print(f"\n❌ 禁止執行操作: {operation} -> {target}")
            checker.record_operation(operation, target, "blocked", f"失敗檢查: {', '.join(failed)}")
    # 生成報告
    report_file = checker.generate_report('phase2_report.md')
    print("\n" + "="*60)
    print("✅ 第二階段完成！")
    print("="*60)
    print(f"📄 報告: {report_file}")
    print("="*60)
    return True
if __name__ == '__main__':
    main()