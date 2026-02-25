# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: event_driven_system
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
完全事件驅動的自動化系統
所有操作都由事件自動觸發，無需人工干預
"""
# MNGA-002: Import organization needs review
import os
import json
import time
import subprocess
import hashlib
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Callable
import queue
class Event:
    """事件基類"""
    def __init__(self, event_type: str, data: Dict = None, priority: int = 5):
        self.event_type = event_type
        self.data = data or {}
        self.priority = priority
        self.timestamp = datetime.now()
        self.event_id = hashlib.md5(f"{event_type}:{self.timestamp.isoformat()}".encode()).hexdigest()[:8]
    def __lt__(self, other):
        """支持事件比较，用于优先级队列"""
        if not isinstance(other, Event):
            return NotImplemented
        return self.priority < other.priority
class EventDrivenAutomationSystem:
    """完全事件驅動的自動化系統"""
    def __init__(self):
        self.event_queue = queue.PriorityQueue()
        self.event_handlers = defaultdict(list)
        self.event_history = deque(maxlen=1000)
        self.running = False
        self.worker_threads = []
        self.monitoring_active = False
        # 系統狀態
        self.last_system_check = None
        self.last_maintenance = None
        self.file_hashes = {}
        self.system_metrics = {
            'events_processed': 0,
            'events_generated': 0,
            'maintenance_runs': 0,
            'errors_detected': 0
        }
    def register_handler(self, event_type: str, handler: Callable, priority: int = 5):
        """註冊事件處理器"""
        self.event_handlers[event_type].append({
            'handler': handler,
            'priority': priority
        })
        print(f"📝 已註冊處理器: {event_type} (優先級: {priority})")
    def emit_event(self, event: Event):
        """發出事件"""
        self.event_queue.put((event.priority, event))
        self.event_history.append(event)
        self.system_metrics['events_generated'] += 1
        print(f"📨 事件發出: {event.event_type} (ID: {event.event_id})")
    def process_event(self, event: Event):
        """處理事件"""
        handlers = self.event_handlers.get(event.event_type, [])
        if not handlers:
            print(f"⚠️  無處理器: {event.event_type}")
            return
        # 按優先級排序處理器
        sorted_handlers = sorted(handlers, key=lambda x: x['priority'])
        for handler_info in sorted_handlers:
            try:
                handler_info['handler'](event.data)
                self.system_metrics['events_processed'] += 1
                print(f"✅ 事件處理完成: {event.event_type}")
            except Exception as e:
                print(f"❌ 處理器錯誤: {e}")
                self.emit_event(Event('error', {
                    'source_event': event.event_type,
                    'error_message': str(e)
                }))
    def start_event_loop(self, num_workers: int = 3):
        """啟動事件循環"""
        self.running = True
        print(f"🚀 啟動事件驅動系統 ({num_workers} 個工作線程)")
        # 創建工作線程
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
            print(f"👷 工作線程 {i+1} 已啟動")
        # 啟動事件生成器
        self.monitoring_active = True
        generator_thread = threading.Thread(target=self._event_generator_loop)
        generator_thread.daemon = True
        generator_thread.start()
        print("👁️  事件生成器已啟動")
        print("\n✅ 事件驅動系統正在運行...")
        print("   所有操作將由事件自動觸發")
        print("   無需人工干預，系統將自主運作")
    def stop_event_loop(self):
        """停止事件循環"""
        print("\n🛑 停止事件驅動系統...")
        self.running = False
        self.monitoring_active = False
        print("✅ 系統已停止")
    def _worker_loop(self, worker_id: int):
        """工作線程循環"""
        while self.running:
            try:
                # 從隊列中獲取事件
                priority, event = self.event_queue.get(timeout=1)
                print(f"\n👷 [線程 {worker_id}] 處理事件: {event.event_type}")
                self.process_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 工作線程錯誤: {e}")
    def _event_generator_loop(self):
        """事件生成器循環"""
        while self.monitoring_active:
            try:
                # 定期生成系統檢查事件
                self._generate_system_check_events()
                # 監控檔案變化
                self._monitor_file_changes()
                # 監控系統健康
                self._monitor_system_health()
                # 等待下一次檢查
                time.sleep(10)  # 每10秒檢查一次
            except Exception as e:
                print(f"❌ 事件生成器錯誤: {e}")
                time.sleep(5)
    def _generate_system_check_events(self):
        """生成系統檢查事件"""
        now = datetime.now()
        # 每5分鐘生成一次系統檢查事件
        if not self.last_system_check or (now - self.last_system_check) > timedelta(minutes=5):
            self.emit_event(Event('system_check', {
                'check_time': now.isoformat(),
                'trigger': 'scheduled'
            }))
            self.last_system_check = now
    def _monitor_file_changes(self):
        """監控檔案變化"""
        try:
            # 檢查關鍵檔案的變化
            key_files = [
                'knowledge_base.json',
                'phase1_report.md',
                'phase2_report.md',
                'phase3_report.md',
                'phase4_report.md'
            ]
            for file_path in key_files:
                if os.path.exists(file_path):
                    current_hash = self._calculate_file_hash(file_path)
                    if file_path in self.file_hashes:
                        if self.file_hashes[file_path] != current_hash:
                            self.emit_event(Event('file_changed', {
                                'file_path': file_path,
                                'old_hash': self.file_hashes[file_path],
                                'new_hash': current_hash,
                                'change_type': 'modified'
                            }))
                    else:
                        self.emit_event(Event('file_detected', {
                            'file_path': file_path,
                            'file_hash': current_hash
                        }))
                    self.file_hashes[file_path] = current_hash
        except Exception as e:
            print(f"❌ 檔案監控錯誤: {e}")
    def _monitor_system_health(self):
        """監控系統健康"""
        try:
            # 檢查知識庫是否存在和有效
            if os.path.exists('knowledge_base.json'):
                try:
                    with open('knowledge_base.json', 'r') as f:
                        kb = json.load(f)
                    # 檢查知識庫是否需要更新
                    if self._knowledge_base_needs_update(kb):
                        self.emit_event(Event('knowledge_base_outdated', {
                            'reason': 'file_system_changes_detected',
                            'timestamp': datetime.now().isoformat()
                        }))
                except Exception as e:
                    self.emit_event(Event('knowledge_base_error', {
                        'error': str(e)
                    }))
            else:
                self.emit_event(Event('knowledge_base_missing', {}))
        except Exception as e:
            print(f"❌ 系統健康檢查錯誤: {e}")
    def _calculate_file_hash(self, file_path: str) -> str:
        """計算檔案哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:16]
        except Exception:
            return ""
    def _knowledge_base_needs_update(self, kb: Dict) -> bool:
        """檢查知識庫是否需要更新"""
        # 簡單檢查：比較檔案數量
        recorded_files = len(kb.get('files', {}))
        # 掃描當前檔案數量
        current_file_count = 0
        try:
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                           d not in ['__pycache__', 'node_modules', '.git']]
                current_file_count += len([f for f in files if not f.startswith('.')])
        except Exception:
            pass
        # 如果變化超過5%，認為需要更新
        return abs(current_file_count - recorded_files) > recorded_files * 0.05
    def get_system_status(self) -> Dict:
        """獲取系統狀態"""
        return {
            'running': self.running,
            'monitoring_active': self.monitoring_active,
            'queue_size': self.event_queue.qsize(),
            'events_processed': self.system_metrics['events_processed'],
            'events_generated': self.system_metrics['events_generated'],
            'maintenance_runs': self.system_metrics['maintenance_runs'],
            'errors_detected': self.system_metrics['errors_detected'],
            'active_handlers': len(self.event_handlers),
            'history_size': len(self.event_history)
        }
# ============ 事件處理器 ============
def handle_system_check(data: Dict):
    """處理系統檢查事件"""
    print("🔍 執行系統檢查...")
    # 檢查是否需要執行維護
    if needs_maintenance():
        emit_maintenance_event('system_check_triggered')
    else:
        print("✅ 系統正常，無需維護")
def handle_file_changed(data: Dict):
    """處理檔案變化事件"""
    file_path = data.get('file_path')
    change_type = data.get('change_type', 'unknown')
    print(f"📄 檔案變化: {file_path} ({change_type})")
    # 如果是關鍵檔案變化，觸發維護
    if file_path in ['knowledge_base.json', 'phase4_report.md']:
        emit_maintenance_event('file_change_triggered')
def handle_knowledge_base_outdated(data: Dict):
    """處理知識庫過期事件"""
    reason = data.get('reason', 'unknown')
    print(f"⚠️  知識庫過期: {reason}")
    emit_maintenance_event('knowledge_outdated')
def handle_knowledge_base_missing(data: Dict):
    """處理知識庫缺失事件"""
    print("❌ 知識庫缺失，立即創建...")
    emit_maintenance_event('knowledge_missing', priority=1)  # 高優先級
def handle_error(data: Dict):
    """處理錯誤事件"""
    source_event = data.get('source_event', 'unknown')
    error_message = data.get('error_message', 'unknown')
    print(f"❌ 錯誤: {source_event} - {error_message}")
    # 如果是嚴重錯誤，可能需要立即處理
    if 'critical' in error_message.lower():
        emit_maintenance_event('critical_error', priority=1)
def handle_maintenance(data: Dict):
    """處理維護事件"""
    trigger = data.get('trigger', 'unknown')
    data.get('priority', 5)
    print(f"\n🔧 執行維護任務 (觸發: {trigger})")
    try:
        # 執行四個階段
        run_phase1()
        run_phase2()
        run_phase3()
        run_phase4()
        print("✅ 維護完成")
    except Exception as e:
        print(f"❌ 維護失敗: {e}")
        emit_error_event('maintenance_failed', str(e))
# ============ 輔助函數 ============
# 創建全局系統實例
system = EventDrivenAutomationSystem()
def emit_maintenance_event(trigger: str, priority: int = 5):
    """發出維護事件"""
    system.emit_event(Event('maintenance_needed', {
        'trigger': trigger,
        'priority': priority
    }, priority=priority))
def emit_error_event(source: str, error: str):
    """發出錯誤事件"""
    system.emit_event(Event('error', {
        'source_event': source,
        'error_message': error
    }, priority=1))
def needs_maintenance() -> bool:
    """檢查是否需要維護"""
    # 檢查知識庫
    if not os.path.exists('knowledge_base.json'):
        return True
    # 檢查最近維護時間
    if not system.last_maintenance:
        return True
    # 檢查時間間隔
    time_since_maintenance = datetime.now() - system.last_maintenance
    if time_since_maintenance > timedelta(hours=2):
        return True
    return False
def run_phase1():
    """執行第一階段"""
    result = subprocess.run(['python3', 'phase1_scanner.py'], 
                          capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise Exception(f"Phase 1 failed: {result.stderr}")
def run_phase2():
    """執行第二階段"""
    result = subprocess.run(['python3', 'phase2_operation_checker.py'],
                          capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise Exception(f"Phase 2 failed: {result.stderr}")
def run_phase3():
    """執行第三階段"""
    result = subprocess.run(['python3', 'phase3_visualizer.py'],
                          capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise Exception(f"Phase 3 failed: {result.stderr}")
def run_phase4():
    """執行第四階段"""
    result = subprocess.run(['python3', 'phase4_learning_system.py'],
                          capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise Exception(f"Phase 4 failed: {result.stderr}")
    system.last_maintenance = datetime.now()
    system.system_metrics['maintenance_runs'] += 1
# ============ 系統初始化 ============
def initialize_event_driven_system():
    """初始化事件驅動系統"""
    print("🔧 初始化事件驅動系統...")
    # 註冊事件處理器
    system.register_handler('system_check', handle_system_check, priority=5)
    system.register_handler('file_changed', handle_file_changed, priority=4)
    system.register_handler('file_detected', handle_file_changed, priority=4)
    system.register_handler('knowledge_base_outdated', handle_knowledge_base_outdated, priority=3)
    system.register_handler('knowledge_base_missing', handle_knowledge_base_missing, priority=1)
    system.register_handler('error', handle_error, priority=1)
    system.register_handler('maintenance_needed', handle_maintenance, priority=2)
    print("✅ 事件驅動系統初始化完成")
    print(f"📋 已註冊 {len(system.event_handlers)} 種事件處理器")
def start_event_driven_system(num_workers: int = 3):
    """啟動事件驅動系統"""
    print("\n" + "="*60)
    print("🤖 完全事件驅動的自動化系統")
    print("="*60)
    print("\n✨ 系統特性：")
    print("   🔄 所有操作由事件自動觸發")
    print("   👁️  持續監控系統狀態")
    print("   🤖 智能決定執行時機")
    print("   🛡️  自動錯誤處理和恢復")
    print("   📊 實時系統狀態監控")
    print("\n" + "="*60)
    initialize_event_driven_system()
    system.start_event_loop(num_workers)
    print("\n💡 系統現已自主運作，無需人工干預")
    print("   所有維護任務將由事件自動觸發執行")
    print("="*60 + "\n")
def show_system_status():
    """顯示系統狀態"""
    status = system.get_system_status()
    print("\n" + "="*60)
    print("📊 系統狀態")
    print("="*60)
    print(f"運行狀態: {'🟢 運行中' if status['running'] else '🔴 已停止'}")
    print(f"監控狀態: {'🟢 活躍' if status['monitoring_active'] else '🔴 非活躍'}")
    print(f"待處理事件: {status['queue_size']}")
    print(f"已處理事件: {status['events_processed']}")
    print(f"生成事件: {status['events_generated']}")
    print(f"維護執行: {status['maintenance_runs']}")
    print(f"錯誤檢測: {status['errors_detected']}")
    print(f"活躍處理器: {status['active_handlers']}")
    print(f"歷史記錄: {status['history_size']}")
    print("="*60 + "\n")
def main():
    """主程式"""
    try:
        # 啟動事件驅動系統
        start_event_driven_system(3)
        # 主循環
        last_status_update = datetime.now()
        while True:
            time.sleep(10)
            # 定期顯示狀態
            if datetime.now() - last_status_update > timedelta(minutes=1):
                show_system_status()
                last_status_update = datetime.now()
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信號...")
        system.stop_event_loop()
        show_system_status()
        print("✅ 系統已正常停止")
if __name__ == '__main__':
    main()