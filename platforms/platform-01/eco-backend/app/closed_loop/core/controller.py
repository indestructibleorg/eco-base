# =============================================================================
# Closed Loop Controller
# =============================================================================
# 閉環主控制器 - Phase 1 基礎閉環核心組件
# 整合異常檢測、規則引擎、自動修復和指標收集
# =============================================================================

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto

from app.core.logging import get_logger
from app.core.events import event_bus, EventType, on_event, emit_event

from app.closed_loop.detector.anomaly_detector import (
    AnomalyDetector, DetectionRule, AnomalyType, Severity, AnomalyEvent
)
from app.closed_loop.remediator.remediator import (
    Remediator, RemediationAction, RemediationType, RemediationResult
)
from app.closed_loop.rules.rule_engine import (
    RuleEngine, Rule, Action, Condition, ConditionOperator
)
from app.closed_loop.metrics.closed_loop_metrics import (
    ClosedLoopMetrics, closed_loop_metrics
)

logger = get_logger("closed_loop_controller")


class ClosedLoopMode(Enum):
    """閉環運行模式"""
    MANUAL = "manual"           # 手動模式：只檢測不執行
    SEMI_AUTO = "semi_auto"     # 半自動：需要確認
    FULL_AUTO = "full_auto"     # 全自動


class ClosedLoopState(Enum):
    """閉環狀態"""
    IDLE = "idle"
    DETECTING = "detecting"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    COOLDOWN = "cooldown"
    ERROR = "error"


@dataclass
class ClosedLoopConfig:
    """閉環配置"""
    mode: ClosedLoopMode = ClosedLoopMode.SEMI_AUTO
    detection_interval_seconds: int = 30
    max_concurrent_remediations: int = 3
    global_cooldown_minutes: int = 5
    enable_metrics: bool = True
    enable_events: bool = True


class ClosedLoopController:
    """
    閉環主控制器
    
    實現 MAPE-K 循環：
    - Monitor (監控): 收集指標和事件
    - Analyze (分析): 檢測異常
    - Plan (規劃): 評估規則和決策
    - Execute (執行): 執行修復動作
    - Knowledge (知識): 存儲歷史和經驗
    """
    
    def __init__(self, config: Optional[ClosedLoopConfig] = None):
        self.config = config or ClosedLoopConfig()
        
        # 核心組件
        self.detector = AnomalyDetector()
        self.remediator = Remediator()
        self.rule_engine = RuleEngine(self.remediator)
        self.metrics = closed_loop_metrics if self.config.enable_metrics else None
        
        # 狀態管理
        self.state = ClosedLoopState.IDLE
        self._running = False
        self._detection_task: Optional[asyncio.Task] = None
        
        # 歷史記錄
        self._anomaly_history: List[AnomalyEvent] = []
        self._remediation_history: List[RemediationResult] = []
        
        # 冷卻管理
        self._last_execution: Optional[datetime] = None
        self._active_remediations: int = 0
        
        # 回調
        self._on_anomaly_callbacks: List[Callable[[AnomalyEvent], None]] = []
        self._on_remediation_callbacks: List[Callable[[RemediationResult], None]] = []
        
        logger.info("closed_loop_controller_initialized", mode=self.config.mode.value)
    
    # =====================================================================
    # 生命周期管理
    # =====================================================================
    
    async def start(self) -> None:
        """啟動閉環系統"""
        if self._running:
            logger.warning("closed_loop_already_running")
            return
        
        self._running = True
        self.state = ClosedLoopState.IDLE
        
        if self.metrics:
            self.metrics.set_active(True)
        
        # 註冊事件監聽
        self._register_event_handlers()
        
        # 啟動檢測循環
        self._detection_task = asyncio.create_task(self._detection_loop())
        
        logger.info("closed_loop_started", mode=self.config.mode.value)
        
        await emit_event(
            EventType.USER_LOGIN,  # 復用現有事件
            {"event": "closed_loop_started", "mode": self.config.mode.value},
            source="closed_loop_controller"
        )
    
    async def stop(self) -> None:
        """停止閉環系統"""
        if not self._running:
            return
        
        self._running = False
        
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        
        if self.metrics:
            self.metrics.set_active(False)
        
        self.state = ClosedLoopState.IDLE
        
        logger.info("closed_loop_stopped")
    
    def _register_event_handlers(self) -> None:
        """註冊事件處理器"""
        # 監聽異常事件
        @on_event(EventType.PROVIDER_ERROR)
        async def handle_anomaly_event(event):
            await self._handle_external_anomaly(event.payload)
    
    # =====================================================================
    # 檢測循環 (Monitor + Analyze)
    # =====================================================================
    
    async def _detection_loop(self) -> None:
        """檢測循環"""
        while self._running:
            try:
                self.state = ClosedLoopState.DETECTING
                
                # 執行所有檢測規則
                await self._run_detections()
                
                # 評估所有規則
                self.state = ClosedLoopState.ANALYZING
                triggered_rules = self.rule_engine.evaluate_all()
                
                # 處理觸發的規則
                if triggered_rules:
                    await self._process_triggered_rules(triggered_rules)
                
                self.state = ClosedLoopState.IDLE
                
                # 等待下一次檢測
                await asyncio.sleep(self.config.detection_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("detection_loop_error", error=str(e))
                self.state = ClosedLoopState.ERROR
                await asyncio.sleep(self.config.detection_interval_seconds)
    
    async def _run_detections(self) -> None:
        """運行所有檢測"""
        for rule_name, rule in self.detector.rules.items():
            try:
                start_time = datetime.utcnow()
                
                # 這裡需要從某處獲取指標數據
                # 實際實現中會調用 Prometheus API
                
                latency = (datetime.utcnow() - start_time).total_seconds()
                
                if self.metrics:
                    self.metrics.record_anomaly_detection(
                        metric_name=rule.metric_name,
                        anomaly_type=rule.anomaly_type.name,
                        severity=rule.severity.value,
                        latency_seconds=latency
                    )
                    
            except Exception as e:
                logger.error("detection_failed", rule_name=rule_name, error=str(e))
    
    async def _handle_external_anomaly(self, payload: Dict[str, Any]) -> None:
        """處理外部異常事件"""
        # 從事件中提取異常信息
        anomaly_type = payload.get("anomaly_type", "UNKNOWN")
        severity = payload.get("severity", "warning")
        
        logger.info(
            "external_anomaly_received",
            anomaly_type=anomaly_type,
            severity=severity
        )
        
        # 評估規則
        facts = {"anomaly": payload}
        triggered = self.rule_engine.evaluate_all(facts)
        
        if triggered:
            await self._process_triggered_rules(triggered)
    
    # =====================================================================
    # 決策與執行 (Plan + Execute)
    # =====================================================================
    
    async def _process_triggered_rules(self, rule_names: List[str]) -> None:
        """處理觸發的規則"""
        for rule_name in rule_names:
            try:
                await self._execute_rule(rule_name)
            except Exception as e:
                logger.error("rule_execution_failed", rule_name=rule_name, error=str(e))
    
    async def _execute_rule(self, rule_name: str) -> List[RemediationResult]:
        """執行規則"""
        # 檢查冷卻
        if not self._can_execute():
            logger.info("execution_cooldown_active", rule_name=rule_name)
            return []
        
        # 檢查並發限制
        if self._active_remediations >= self.config.max_concurrent_remediations:
            logger.warning(
                "max_concurrent_remediations_reached",
                current=self._active_remediations
            )
            return []
        
        # 根據模式決定執行方式
        if self.config.mode == ClosedLoopMode.MANUAL:
            logger.info("manual_mode_skip_execution", rule_name=rule_name)
            return []
        
        if self.config.mode == ClosedLoopMode.SEMI_AUTO:
            # 半自動模式：發送通知等待確認
            logger.info("semi_auto_mode_waiting_approval", rule_name=rule_name)
            await self._request_approval(rule_name)
            return []
        
        # 全自動模式：直接執行
        self.state = ClosedLoopState.EXECUTING
        self._active_remediations += 1
        self._last_execution = datetime.utcnow()
        
        try:
            results = await self.rule_engine.execute_rule(rule_name)
            
            # 記錄結果
            for result in results:
                self._remediation_history.append(result)
                
                if self.metrics:
                    self.metrics.record_remediation(
                        action_type=result.action_type.name,
                        status=result.status.value,
                        latency_seconds=result.duration_seconds
                    )
                
                # 觸發回調
                for callback in self._on_remediation_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error("remediation_callback_failed", error=str(e))
            
            return results
            
        finally:
            self._active_remediations -= 1
            self.state = ClosedLoopState.COOLDOWN
    
    def _can_execute(self) -> bool:
        """檢查是否可以執行"""
        if not self._last_execution:
            return True
        
        cooldown = timedelta(minutes=self.config.global_cooldown_minutes)
        return datetime.utcnow() - self._last_execution >= cooldown
    
    async def _request_approval(self, rule_name: str) -> None:
        """請求人工審批"""
        await emit_event(
            EventType.USER_LOGIN,  # 復用現有事件
            {
                "event": "approval_required",
                "rule_name": rule_name,
                "timestamp": datetime.utcnow().isoformat()
            },
            source="closed_loop_controller"
        )
    
    # =====================================================================
    # 公共 API
    # =====================================================================
    
    def add_detection_rule(self, rule: DetectionRule) -> None:
        """添加檢測規則"""
        self.detector.register_rule(rule)
    
    def add_remediation_rule(self, rule: Rule) -> None:
        """添加修復規則"""
        self.rule_engine.add_rule(rule)
    
    def load_rules_from_yaml(self, yaml_content: str) -> int:
        """從 YAML 加載規則"""
        return self.rule_engine.load_rules_from_yaml(yaml_content)
    
    async def manual_trigger(self, rule_name: str) -> List[RemediationResult]:
        """手動觸發規則"""
        return await self.rule_engine.execute_rule(rule_name)
    
    async def approve_execution(self, rule_name: str) -> List[RemediationResult]:
        """批准執行（半自動模式）"""
        if self.config.mode != ClosedLoopMode.SEMI_AUTO:
            logger.warning("approve_called_in_non_semi_auto_mode")
        
        return await self._execute_rule(rule_name)
    
    def get_status(self) -> Dict[str, Any]:
        """獲取閉環狀態"""
        return {
            "state": self.state.value,
            "mode": self.config.mode.value,
            "running": self._running,
            "active_remediations": self._active_remediations,
            "detection_rules": len(self.detector.rules),
            "remediation_rules": len(self.rule_engine.rules),
            "anomaly_history_count": len(self._anomaly_history),
            "remediation_history_count": len(self._remediation_history),
            "metrics": self.metrics.get_summary() if self.metrics else None
        }
    
    def get_metrics(self) -> Optional[str]:
        """獲取 Prometheus 格式指標"""
        if self.metrics:
            return self.metrics.export_prometheus_format()
        return None
    
    def on_anomaly(self, callback: Callable[[AnomalyEvent], None]) -> None:
        """註冊異常回調"""
        self._on_anomaly_callbacks.append(callback)
    
    def on_remediation(self, callback: Callable[[RemediationResult], None]) -> None:
        """註冊修復回調"""
        self._on_remediation_callbacks.append(callback)
    
    # =====================================================================
    # 預置規則
    # =====================================================================
    
    def setup_default_rules(self) -> None:
        """設置默認規則"""
        # 高延遲自動擴容規則
        self.add_detection_rule(DetectionRule(
            name="high_latency_detection",
            metric_name="http_request_duration_seconds",
            anomaly_type=AnomalyType.THRESHOLD,
            severity=Severity.WARNING,
            threshold_max=2.0
        ))
        
        # 錯誤率過高檢測
        self.add_detection_rule(DetectionRule(
            name="high_error_rate_detection",
            metric_name="http_requests_failed_total",
            anomaly_type=AnomalyType.STATISTICAL,
            severity=Severity.CRITICAL,
            sigma_multiplier=2.0
        ))
        
        # 高延遲自動擴容規則
        scale_up_rule = Rule(
            name="auto_scale_on_high_latency",
            description="Auto scale up when latency is high",
            priority=100,
            condition=Condition(
                field="anomaly.rule_name",
                operator=ConditionOperator.EQ,
                value="high_latency_detection"
            ),
            actions=[
                Action(
                    action_type="scale_up",
                    parameters={
                        "namespace": "default",
                        "deployment": "eco-backend",
                        "replicas_delta": 2,
                        "max_replicas": 10
                    },
                    delay_seconds=0,
                    require_approval=self.config.mode == ClosedLoopMode.SEMI_AUTO
                )
            ],
            cooldown_minutes=10
        )
        self.add_remediation_rule(scale_up_rule)
        
        # Pod 重啟規則
        restart_rule = Rule(
            name="restart_pod_on_crash",
            description="Restart pod when it crashes",
            priority=50,
            condition=Condition(
                field="anomaly.anomaly_type",
                operator=ConditionOperator.EQ,
                value="POD_CRASH"
            ),
            actions=[
                Action(
                    action_type="restart_pod",
                    parameters={
                        "namespace": "default",
                        "label_selector": "app=eco-backend"
                    },
                    delay_seconds=30
                )
            ],
            cooldown_minutes=5
        )
        self.add_remediation_rule(restart_rule)
        
        logger.info("default_rules_setup_complete")


# 全局閉環控制器實例
_closed_loop_controller: Optional[ClosedLoopController] = None


def get_closed_loop_controller() -> ClosedLoopController:
    """獲取閉環控制器實例"""
    global _closed_loop_controller
    if _closed_loop_controller is None:
        _closed_loop_controller = ClosedLoopController()
    return _closed_loop_controller


def init_closed_loop(config: Optional[ClosedLoopConfig] = None) -> ClosedLoopController:
    """初始化閉環系統"""
    global _closed_loop_controller
    _closed_loop_controller = ClosedLoopController(config)
    _closed_loop_controller.setup_default_rules()
    return _closed_loop_controller
