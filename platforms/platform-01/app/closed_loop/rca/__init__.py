"""
根因分析引擎 (RCA Engine)

提供事件收集、关联分析、根因识别和报告生成功能
"""

from .event_collector import (
    EventCollector,
    Event,
    EventType,
    EventSeverity,
    EventBatch,
)

from .correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationResult,
    EventCluster,
)

from .root_cause_identifier import (
    RootCauseIdentifier,
    RootCauseAnalysis,
    RootCause,
    RootCauseCategory,
)

from .report_generator import (
    ReportGenerator,
    RCAReport,
    ReportFormat,
)

__all__ = [
    # 事件收集器
    'EventCollector',
    'Event',
    'EventType',
    'EventSeverity',
    'EventBatch',
    
    # 关联分析器
    'CorrelationAnalyzer',
    'CorrelationResult',
    'EventCluster',
    
    # 根因识别器
    'RootCauseIdentifier',
    'RootCauseAnalysis',
    'RootCause',
    'RootCauseCategory',
    
    # 报告生成器
    'ReportGenerator',
    'RCAReport',
    'ReportFormat',
]
