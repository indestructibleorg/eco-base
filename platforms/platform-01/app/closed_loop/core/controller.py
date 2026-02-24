"""
闭环主控制器
整合所有闭环组件，实现MAPE-K循环
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto

from ..detector import AnomalyDetector, Anomaly
from ..remediator import AutoRemediator, RemediationAction, RemediationType
from ..rules import RuleEngine, RuleAction
from ..metrics import ClosedLoopMetrics, record_full_loop_latency

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """闭环状态"""
    IDLE = auto()
    MONITORING = auto()
    ANALYZING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    LEARNING = auto()


@dataclass
class LoopContext:
    """闭环上下文"""
    metric_name: str
    metric_value: float
    timestamp: datetime
    anomaly: Optional[Anomaly] = None
    triggered_rules: List = field(default_factory=list)
    remediation_results: List = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClosedLoopController:
    """闭环主控制器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 组件
        self.detector: Optional[AnomalyDetector] = None
        self.remediator: Optional[AutoRemediator] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.metrics: Optional[ClosedLoopMetrics] = None
        
        # 状态
        self.state = LoopState.IDLE
        self.running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        
        # 回调
        self.anomaly_callbacks: List[Callable[[Anomaly], None]] = []
        self.remediation_callbacks: List[Callable[[Any], None]] = []
        
        logger.info("闭环主控制器初始化完成")
    
    def set_detector(self, detector: AnomalyDetector):
        """设置检测器"""
        self.detector = detector
        # 注册异常处理器
        self.detector.register_anomaly_handler(self._on_anomaly_detected)
        logger.info("检测器已设置")
    
    def set_remediator(self, remediator: AutoRemediator):
        """设置修复器"""
        self.remediator = remediator
        logger.info("修复器已设置")
    
    def set_rule_engine(self, rule_engine: RuleEngine):
        """设置规则引擎"""
        self.rule_engine = rule_engine
        # 注册规则动作处理器
        self.rule_engine.register_action_handler(
            RuleAction.REMEDIATE, 
            self._on_remediate_action
        )
        logger.info("规则引擎已设置")
    
    def set_metrics(self, metrics: ClosedLoopMetrics):
        """设置指标收集器"""
        self.metrics = metrics
        logger.info("指标收集器已设置")
    
    def register_anomaly_callback(self, callback: Callable[[Anomaly], None]):
        """注册异常回调"""
        self.anomaly_callbacks.append(callback)
    
    def register_remediation_callback(self, callback: Callable[[Any], None]):
        """注册修复回调"""
        self.remediation_callbacks.append(callback)
    
    async def start(self):
        """启动闭环系统"""
        self.running = True
        self._main_loop_task = asyncio.create_task(self._main_loop())
        logger.info("闭环系统已启动")
    
    async def stop(self):
        """停止闭环系统"""
        self.running = False
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("闭环系统已停止")
    
    async def _main_loop(self):
        """主循环"""
        while self.running:
            try:
                # 清理过期指标
                if self.metrics:
                    self.metrics.cleanup_old_metrics()
                
                await asyncio.sleep(60)  # 每分钟执行一次维护
            except Exception as e:
                logger.exception(f"主循环异常: {e}")
    
    async def process_metric(self, metric_name: str, 
                             value: float,
                             timestamp: datetime = None) -> Optional[LoopContext]:
        """处理指标数据 (MAPE-K循环入口)"""
        import time
        loop_start = time.time()
        
        timestamp = timestamp or datetime.now()
        context = LoopContext(
            metric_name=metric_name,
            metric_value=value,
            timestamp=timestamp
        )
        
        # Monitor: 数据已接收
        self.state = LoopState.MONITORING
        
        # Analyze: 异常检测
        self.state = LoopState.ANALYZING
        anomaly = await self._analyze(metric_name, value, timestamp)
        
        if not anomaly:
            # 无异常，记录正常指标
            if self.metrics:
                self.metrics.set_gauge(f"metric.{metric_name}", value)
            return None
        
        context.anomaly = anomaly
        
        # Plan: 规则评估
        self.state = LoopState.PLANNING
        triggered_rules = await self._plan(anomaly, context)
        context.triggered_rules = triggered_rules
        
        if not triggered_rules:
            logger.info(f"无匹配规则: {metric_name}")
            return context
        
        # Execute: 执行修复
        self.state = LoopState.EXECUTING
        remediation_results = await self._execute(triggered_rules, context)
        context.remediation_results = remediation_results
        
        # Knowledge: 记录结果
        self.state = LoopState.LEARNING
        await self._learn(context)
        
        # 记录闭环延迟
        loop_duration = (time.time() - loop_start) * 1000
        if self.metrics:
            record_full_loop_latency(self.metrics, loop_duration)
        
        self.state = LoopState.IDLE
        return context
    
    async def _analyze(self, metric_name: str, 
                       value: float,
                       timestamp: datetime) -> Optional[Anomaly]:
        """分析阶段 - 异常检测"""
        if not self.detector:
            return None
        
        import time
        start = time.time()
        
        anomaly = self.detector.add_data_point(metric_name, timestamp, value)
        
        # 记录检测延迟
        if self.metrics:
            duration = (time.time() - start) * 1000
            from ..metrics import record_detection_latency
            record_detection_latency(self.metrics, duration)
        
        return anomaly
    
    async def _plan(self, anomaly: Anomaly, 
                    context: LoopContext) -> List:
        """规划阶段 - 规则评估"""
        if not self.rule_engine:
            return []
        
        # 构建评估上下文
        eval_context = {
            'metrics': {context.metric_name: context.metric_value},
            'anomalies': [{
                'type': anomaly.anomaly_type.value,
                'severity': anomaly.severity,
                'metric': anomaly.metric_name
            }]
        }
        
        triggered = self.rule_engine.evaluate_all(eval_context)
        return triggered
    
    async def _execute(self, rules: List, 
                       context: LoopContext) -> List:
        """执行阶段 - 执行修复"""
        if not self.remediator:
            return []
        
        import time
        start = time.time()
        
        eval_context = {
            'metric_name': context.metric_name,
            'metric_value': context.metric_value,
            'anomaly': context.anomaly
        }
        
        results = await self.rule_engine.execute_actions(rules, eval_context)
        
        # 记录修复延迟
        if self.metrics:
            duration = (time.time() - start) * 1000
            from ..metrics import record_remediation_latency
            record_remediation_latency(self.metrics, duration)
        
        return results
    
    async def _learn(self, context: LoopContext):
        """学习阶段 - 记录知识"""
        # 记录事件
        if self.metrics:
            self.metrics.increment_counter('closed_loop_executions')
            
            if context.anomaly:
                self.metrics.increment_counter(
                    'anomalies_detected',
                    labels={'type': context.anomaly.anomaly_type.value}
                )
            
            for result in context.remediation_results:
                if result.get('error'):
                    self.metrics.increment_counter('remediation_failures')
                else:
                    self.metrics.increment_counter('remediation_success')
        
        logger.info(f"闭环执行完成: {context.metric_name}")
    
    def _on_anomaly_detected(self, anomaly: Anomaly):
        """异常检测回调"""
        for callback in self.anomaly_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(anomaly))
                else:
                    callback(anomaly)
            except Exception as e:
                logger.warning(f"异常回调失败: {e}")
    
    async def _on_remediate_action(self, rule, context):
        """修复动作回调"""
        if not self.remediator:
            return None
        
        # 创建修复动作
        action = RemediationAction(
            action_id=f"rem_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            action_type=RemediationType.CUSTOM,
            target=context.get('metric_name', 'unknown'),
            params=rule.action_config
        )
        
        result = await self.remediator.execute(action)
        
        for callback in self.remediation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.warning(f"修复回调失败: {e}")
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'status': self.state.name.lower(),
            'state': self.state.name,
            'running': self.running,
            'components': {
                'detector': self.detector is not None,
                'remediator': self.remediator is not None,
                'rule_engine': self.rule_engine is not None,
                'metrics': self.metrics is not None
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if self.metrics:
            return self.metrics.get_metrics_summary()
        return {}


class Phase2Controller:
    """Phase 2 智能闭环控制器"""
    
    def __init__(self, 
                 base_controller: ClosedLoopController,
                 rca_engine=None,
                 alert_router=None,
                 capacity_manager=None,
                 workflow_engine=None):
        self.base = base_controller
        self.rca_engine = rca_engine
        self.alert_router = alert_router
        self.capacity_manager = capacity_manager
        self.workflow_engine = workflow_engine
        
        logger.info("Phase 2 控制器初始化完成")
    
    async def process_with_rca(self, event_id: str) -> Dict[str, Any]:
        """使用RCA处理问题"""
        if not self.rca_engine:
            return {'error': 'RCA引擎未配置'}
        
        # 生成RCA报告
        from ..rca import ReportFormat
        report = await self.rca_engine.generate(event_id)
        
        return {
            'report_id': report.report_id,
            'root_causes': report.root_causes,
            'recommendations': report.recommendations
        }
    
    async def route_alert(self, alert_data: Dict) -> Dict[str, Any]:
        """路由告警"""
        if not self.alert_router:
            return {'error': '告警路由器未配置'}
        
        from ..alert import Alert, AlertSeverity, AlertStatus
        
        alert = Alert(
            alert_id=alert_data.get('id', f"alert_{datetime.now().timestamp()}"),
            title=alert_data.get('title', 'Unknown'),
            description=alert_data.get('description', ''),
            source=alert_data.get('source', 'unknown'),
            severity=AlertSeverity(alert_data.get('severity', 3)),
            status=AlertStatus.NEW,
            created_at=datetime.now()
        )
        
        result = await self.alert_router.route_alert(alert)
        return result
    
    async def auto_scale(self, service: str) -> Dict[str, Any]:
        """自动扩缩容"""
        if not self.capacity_manager:
            return {'error': '容量管理器未配置'}
        
        # 分析并制定计划
        plans = await self.capacity_manager.analyze_and_plan(service)
        
        return {
            'service': service,
            'plans': [p.to_dict() for p in plans]
        }
    
    def get_integrated_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        return {
            'base': self.base.get_status(),
            'phase2': {
                'rca_engine': self.rca_engine is not None,
                'alert_router': self.alert_router is not None,
                'capacity_manager': self.capacity_manager is not None,
                'workflow_engine': self.workflow_engine is not None
            }
        }
