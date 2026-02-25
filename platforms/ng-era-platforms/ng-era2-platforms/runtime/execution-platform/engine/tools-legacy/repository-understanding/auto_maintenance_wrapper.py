# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: auto_maintenance_wrapper
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
輕量級自動維護包裝器
在工作進程中定期執行維護任務
"""
# MNGA-002: Import organization needs review
import os
import json
import time
import subprocess
from datetime import datetime, timedelta
class LightweightAutoMaintenance:
    def __init__(self, check_interval=300):
        """
        初始化輕量級自動維護系統
        Args:
            check_interval: 檢查間隔（秒），默認5分鐘
        """
        self.check_interval = check_interval
        self.knowledge_base_path = 'knowledge_base.json'
        self.last_maintenance_time = None
        self.maintenance_log = []
    def check_if_maintenance_needed(self):
        """檢查是否需要執行維護"""
        # 檢查1: 知識庫檔案是否存在
        if not os.path.exists(self.knowledge_base_path):
            print("🔍 知識庫不存在，需要執行維護")
            return True
        # 檢查2: 最後維護時間
        if self.last_maintenance_time:
            time_since_maintenance = datetime.now() - self.last_maintenance_time
            if time_since_maintenance > timedelta(hours=1):  # 超過1小時
                print(f"⏰ 距離上次維護已過 {time_since_maintenance.seconds//60} 分鐘")
                return True
        # 檢查3: 檔案系統變化
        if self.detect_filesystem_changes():
            print("📂 檢測到檔案系統變化")
            return True
        return False
    def detect_filesystem_changes(self):
        """檢測檔案系統變化"""
        try:
            # 比較當前檔案數量與知識庫記錄
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            recorded_files = len(kb.get('files', {}))
            # 簡單檢查：掃描當前目錄的檔案數
            current_file_count = 0
            for root, dirs, files in os.walk('.'):
                # 忽略隱藏目錄和系統目錄
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                           d not in ['__pycache__', 'node_modules', '.git']]
                for file in files:
                    if not file.startswith('.'):
                        current_file_count += 1
            # 如果差異超過10%，認為有變化
            if abs(current_file_count - recorded_files) > recorded_files * 0.1:
                print(f"📊 檔案數量變化: {recorded_files} -> {current_file_count}")
                return True
            return False
        except Exception as e:
            print(f"⚠️  變化檢測失敗: {e}")
            return False
    def perform_maintenance(self):
        """執行維護任務"""
        print("\n" + "="*60)
        print(f"🤖 執行自動維護 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        maintenance_results = {
            'timestamp': datetime.now().isoformat(),
            'phases': {},
            'success': False
        }
        try:
            # 第一階段：掃描
            print("\n📊 第一階段：掃描儲存庫...")
            result1 = subprocess.run(['python3', 'phase1_scanner.py'], 
                                    capture_output=True, text=True, timeout=120)
            maintenance_results['phases']['phase1'] = {
                'success': result1.returncode == 0,
                'output': result1.stdout[:200] if result1.stdout else ''
            }
            # 第二階段：檢查機制
            print("🛡️  第二階段：驗證檢查機制...")
            result2 = subprocess.run(['python3', 'phase2_operation_checker.py'],
                                    capture_output=True, text=True, timeout=60)
            maintenance_results['phases']['phase2'] = {
                'success': result2.returncode == 0,
                'output': result2.stdout[:200] if result2.stdout else ''
            }
            # 第三階段：查詢系統
            print("🔍 第三階段：更新查詢系統...")
            result3 = subprocess.run(['python3', 'phase3_visualizer.py'],
                                    capture_output=True, text=True, timeout=60)
            maintenance_results['phases']['phase3'] = {
                'success': result3.returncode == 0,
                'output': result3.stdout[:200] if result3.stdout else ''
            }
            # 第四階段：學習系統
            print("🧠 第四階段：執行學習系統...")
            result4 = subprocess.run(['python3', 'phase4_learning_system.py'],
                                    capture_output=True, text=True, timeout=60)
            maintenance_results['phases']['phase4'] = {
                'success': result4.returncode == 0,
                'output': result4.stdout[:200] if result4.stdout else ''
            }
            # 檢查所有階段是否成功
            all_success = all(phase['success'] for phase in maintenance_results['phases'].values())
            maintenance_results['success'] = all_success
            if all_success:
                self.last_maintenance_time = datetime.now()
                print("\n✅ 自動維護成功完成")
            else:
                print("\n⚠️  部分階段完成，但可能有錯誤")
        except subprocess.TimeoutExpired:
            print("❌ 維護超時")
            maintenance_results['error'] = 'timeout'
        except Exception as e:
            print(f"❌ 維護失敗: {e}")
            maintenance_results['error'] = str(e)
        # 記錄維護日誌
        self.maintenance_log.append(maintenance_results)
        return maintenance_results
    def run_automated_maintenance(self, max_iterations=None):
        """
        運行自動化維護循環
        Args:
            max_iterations: 最大迭代次數，None表示無限循環
        """
        print("🚀 啟動輕量級自動維護系統")
        print(f"⏰ 檢查間隔: {self.check_interval} 秒")
        if max_iterations:
            print(f"🔄 最大迭代: {max_iterations} 次")
        else:
            print("🔄 模式: 無限循環")
        iteration = 0
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    print(f"\n✅ 達到最大迭代次數 {max_iterations}，停止運行")
                    break
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                # 檢查是否需要維護
                if self.check_if_maintenance_needed():
                    result = self.perform_maintenance()
                    if result.get('success'):
                        print("✅ 維護完成，等待下次檢查...")
                    else:
                        print("⚠️  維護完成但可能存在問題")
                else:
                    print(f"\r[{current_time}] ✅ 系統正常，等待下次檢查...", end='', flush=True)
                # 等待下次檢查
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\n\n🛑 收到停止信號")
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
        print(f"📊 總共執行了 {iteration} 次檢查")
        print(f"📋 維護日誌: {len(self.maintenance_log)} 條記錄")
def integrate_into_workflow():
    """
    集成到工作流程中
    在主要工作開始前和結束後自動執行維護
    """
    print("="*60)
    print("🔄 工作流程集成模式")
    print("="*60)
    maintenance = LightweightAutoMaintenance()
    print("\n📋 工作開始前的檢查...")
    if maintenance.check_if_maintenance_needed():
        print("🔧 執行預維護...")
        maintenance.perform_maintenance()
    else:
        print("✅ 無需維護，繼續工作")
    print("\n" + "="*60)
    print("🚀 主要工作可以在這裡執行...")
    print("在此處執行你的主要工作任務")
    print("="*60)
    # 模擬工作完成
    print("\n📋 工作結束後的檢查...")
    if maintenance.check_if_maintenance_needed():
        print("🔧 執行後維護...")
        maintenance.perform_maintenance()
    else:
        print("✅ 無需維護，工作完成")
def main():
    """主程式"""
    print("""
╔═════════════════════════════════════════════════════════╗
║     自動化維護系統選擇器                               ║
╚═════════════════════════════════════════════════════════╝
請選擇運行模式：
1. 🤖 守護進程模式 - 持續監控和自動維護
2. 🔄 工作流程集成 - 在工作前後執行維護
3. 🚨 立即執行 - 單次執行維護
4. 📊 狀態檢查 - 檢查系統狀態
""")
    choice = input("請選擇 (1-4): ").strip()
    maintenance = LightweightAutoMaintenance()
    if choice == '1':
        print("\n🤖 啟動守護進程模式...")
        print("提示：按 Ctrl+C 可隨時停止")
        maintenance.run_automated_maintenance(max_iterations=None)
    elif choice == '2':
        print("\n🔄 啟動工作流程集成模式...")
        integrate_into_workflow()
    elif choice == '3':
        print("\n🚨 立即執行維護...")
        result = maintenance.perform_maintenance()
        print(f"\n結果: {'成功' if result.get('success') else '部分失敗'}")
    elif choice == '4':
        print("\n📊 系統狀態檢查...")
        needs_maintenance = maintenance.check_if_maintenance_needed()
        print(f"需要維護: {'是' if needs_maintenance else '否'}")
        if maintenance.last_maintenance_time:
            print(f"最後維護: {maintenance.last_maintenance_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("最後維護: 尚未執行")
    else:
        print("❌ 無效選擇")
if __name__ == '__main__':
    main()