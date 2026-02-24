# =============================================================================
# Closed Loop System - Phase 1
# =============================================================================
# 閉環自動化系統 - 基礎版本
# 
# 實現 MAPE-K 循環：
# - Monitor (監控): 收集指標和事件
# - Analyze (分析): 檢測異常
# - Plan (規劃): 評估規則和決策
# - Execute (執行): 執行修復動作
# - Knowledge (知識): 存儲歷史和經驗
# =============================================================================

from app.closed_loop.detector.anomaly_detector import (
    AnomalyDetector,
    DetectionRule,
    AnomalyEvent,
    AnomalyType,
    Severity,
    StatisticalDetector,
    TrendDetector,
    anomaly_detector,
)

from app.closed_loop.remediator.remediator import (
    Remediator,
    RemediationAction,
    RemediationResult,
    RemediationType,
    RemediationStatus,
    RemediationActionHandler,
    RestartPodHandler,
    ClearCacheHandler,
    SwitchProviderHandler,
    ScaleHandler,
    remediator,
)

from app.closed_loop.rules.rule_engine import (
    RuleEngine,
    Rule,
    Action,
    Condition,
    CompoundCondition,
    ConditionOperator,
    LogicalOperator,
    RuleParser,
    rule_engine,
)

from app.closed_loop.metrics.closed_loop_metrics import (
    ClosedLoopMetrics,
    Counter,
    Gauge,
    Histogram,
    closed_loop_metrics,
)

from app.closed_loop.core.controller import (
    ClosedLoopController,
    ClosedLoopConfig,
    ClosedLoopMode,
    ClosedLoopState,
    get_closed_loop_controller,
    init_closed_loop,
)

__version__ = "1.0.0"
__all__ = [
    # 異常檢測
    "AnomalyDetector",
    "DetectionRule",
    "AnomalyEvent",
    "AnomalyType",
    "Severity",
    "StatisticalDetector",
    "TrendDetector",
    "anomaly_detector",
    
    # 自動修復
    "Remediator",
    "RemediationAction",
    "RemediationResult",
    "RemediationType",
    "RemediationStatus",
    "RemediationActionHandler",
    "RestartPodHandler",
    "ClearCacheHandler",
    "SwitchProviderHandler",
    "ScaleHandler",
    "remediator",
    
    # 規則引擎
    "RuleEngine",
    "Rule",
    "Action",
    "Condition",
    "CompoundCondition",
    "ConditionOperator",
    "LogicalOperator",
    "RuleParser",
    "rule_engine",
    
    # 指標
    "ClosedLoopMetrics",
    "Counter",
    "Gauge",
    "Histogram",
    "closed_loop_metrics",
    
    # 控制器
    "ClosedLoopController",
    "ClosedLoopConfig",
    "ClosedLoopMode",
    "ClosedLoopState",
    "get_closed_loop_controller",
    "init_closed_loop",
]
