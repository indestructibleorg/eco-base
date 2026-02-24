# =============================================================================
# RCA Engine - Root Cause Analysis
# =============================================================================
# 根因分析引擎 - Phase 2 核心组件
# 
# 功能：
# - 事件收集和标准化
# - 事件关联分析
# - 根本原因识别
# - RCA 报告生成
# =============================================================================

from app.closed_loop.rca.event_collector import (
    Event,
    EventType,
    EventSeverity,
    EventGroup,
    EventCollector,
    event_collector,
)

from app.closed_loop.rca.correlation_analyzer import (
    CorrelationResult,
    EventCluster,
    TemporalCorrelationAnalyzer,
    AttributeCorrelationAnalyzer,
    CausalInferenceAnalyzer,
    CorrelationAnalyzer,
)

from app.closed_loop.rca.root_cause_identifier import (
    RootCause,
    RCAResult,
    CausalGraph,
    BayesianNetworkAnalyzer,
    DecisionTreeAnalyzer,
    RootCauseIdentifier,
)

from app.closed_loop.rca.report_generator import (
    ReportGenerator,
    report_generator,
)


class RCAEngine:
    """
    根因分析引擎
    
    整合事件收集、关联分析、根因识别和报告生成
    """
    
    def __init__(
        self,
        event_collector: EventCollector = None,
        correlation_analyzer: CorrelationAnalyzer = None,
        root_cause_identifier: RootCauseIdentifier = None,
        report_generator: ReportGenerator = None,
    ):
        self.event_collector = event_collector or EventCollector()
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        self.root_cause_identifier = root_cause_identifier or RootCauseIdentifier()
        self.report_generator = report_generator or ReportGenerator()
    
    def collect_event(self, event: Event) -> bool:
        """收集事件"""
        return self.event_collector.collect(event)
    
    def analyze(
        self,
        event_ids: list = None,
        time_window_minutes: int = 30,
    ) -> RCAResult:
        """
        执行根因分析
        
        Args:
            event_ids: 要分析的事件 ID 列表，如果为 None 则分析最近的事件
            time_window_minutes: 分析时间窗口
        
        Returns:
            RCA 分析结果
        """
        from datetime import datetime, timedelta
        
        # 获取要分析的事件
        if event_ids:
            events = [
                self.event_collector.get_event(eid)
                for eid in event_ids
            ]
            events = [e for e in events if e]
        else:
            # 获取最近时间窗口内的事件
            start_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            events = self.event_collector.get_events(start=start_time)
        
        if not events:
            # 返回空结果
            return RCAResult(
                id=f"rca-{self._generate_id()}",
                timestamp=datetime.utcnow(),
                root_causes=[],
                contributing_factors=[],
                affected_services=[],
                timeline=[],
                confidence=0.0,
                recommended_actions=["No events found for analysis"],
                analysis_method="none",
            )
        
        # 执行根因识别
        result = self.root_cause_identifier.identify_root_cause(events)
        
        return result
    
    def analyze_event(self, event: Event) -> RCAResult:
        """分析单个事件及其相关事件"""
        # 收集事件
        self.collect_event(event)
        
        # 查找相关事件
        all_events = self.event_collector.get_events(limit=1000)
        correlation = self.correlation_analyzer.analyze_correlation(
            event, all_events
        )
        
        # 获取相关事件详情
        related_event_ids = [eid for eid, _ in correlation.correlated_events]
        related_events = [
            self.event_collector.get_event(eid)
            for eid in related_event_ids
        ]
        related_events = [e for e in related_events if e]
        
        # 合并事件列表
        all_related = [event] + related_events
        
        # 执行根因识别
        result = self.root_cause_identifier.identify_root_cause(all_related)
        
        return result
    
    def generate_report(
        self,
        result: RCAResult,
        format: str = "dict",
    ) -> str or dict:
        """
        生成 RCA 报告
        
        Args:
            result: RCA 分析结果
            format: 报告格式 (dict, json, markdown, html)
        
        Returns:
            报告内容
        """
        if format == "dict":
            return self.report_generator.generate_report(result)
        elif format == "json":
            return self.report_generator.export_to_json(result)
        elif format == "markdown":
            return self.report_generator.export_to_markdown(result)
        elif format == "html":
            return self.report_generator.export_to_html(result)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_event_stats(self) -> dict:
        """获取事件统计信息"""
        return self.event_collector.get_stats()
    
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        import uuid
        return uuid.uuid4().hex[:8]


# 全局 RCA 引擎实例
rca_engine = RCAEngine()


__all__ = [
    # 事件收集
    "Event",
    "EventType",
    "EventSeverity",
    "EventGroup",
    "EventCollector",
    "event_collector",
    
    # 关联分析
    "CorrelationResult",
    "EventCluster",
    "TemporalCorrelationAnalyzer",
    "AttributeCorrelationAnalyzer",
    "CausalInferenceAnalyzer",
    "CorrelationAnalyzer",
    
    # 根因识别
    "RootCause",
    "RCAResult",
    "CausalGraph",
    "BayesianNetworkAnalyzer",
    "DecisionTreeAnalyzer",
    "RootCauseIdentifier",
    
    # 报告生成
    "ReportGenerator",
    "report_generator",
    
    # RCA 引擎
    "RCAEngine",
    "rca_engine",
]
