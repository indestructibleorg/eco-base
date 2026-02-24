"""
智能告警路由
实现告警的智能分发、聚合和升级
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from collections import defaultdict
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重级别"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


class AlertStatus(Enum):
    """告警状态"""
    NEW = auto()
    ACKNOWLEDGED = auto()
    SUPPRESSED = auto()
    RESOLVED = auto()
    ESCALATED = auto()


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """告警"""
    alert_id: str
    title: str
    description: str
    source: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notification_history: List[Dict] = field(default_factory=list)


@dataclass
class RoutingRule:
    """路由规则"""
    rule_id: str
    name: str
    condition: Callable[[Alert], bool]
    channels: List[NotificationChannel]
    recipients: List[str]
    priority: int = 5
    suppress_duplicates: bool = True
    suppress_window_minutes: int = 30
    escalate_after_minutes: Optional[int] = None
    escalate_to: Optional[str] = None
    enabled: bool = True


@dataclass
class OnCallSchedule:
    """值班安排"""
    schedule_id: str
    name: str
    rotations: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_current_oncall(self, timestamp: datetime = None) -> Optional[str]:
        """获取当前值班人员"""
        timestamp = timestamp or datetime.now()
        
        for rotation in self.rotations:
            start = rotation.get('start')
            end = rotation.get('end')
            if start and end:
                if start <= timestamp <= end:
                    return rotation.get('person')
        
        return None


class AlertAggregator:
    """告警聚合器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.aggregation_window = self.config.get('aggregation_window_minutes', 5)
        self.max_group_size = self.config.get('max_group_size', 50)
        
        # 聚合组
        self.alert_groups: Dict[str, List[Alert]] = defaultdict(list)
        self.group_metadata: Dict[str, Dict] = {}
        
        logger.info("告警聚合器初始化完成")
    
    def get_group_key(self, alert: Alert) -> str:
        """获取告警分组键"""
        # 基于来源和严重级别分组
        return f"{alert.source}:{alert.severity.name}"
    
    async def add_alert(self, alert: Alert) -> Optional[Alert]:
        """添加告警，可能返回聚合后的告警"""
        group_key = self.get_group_key(alert)
        
        # 检查组是否存在且未过期
        if group_key in self.group_metadata:
            last_update = self.group_metadata[group_key]['last_update']
            if datetime.now() - last_update > timedelta(minutes=self.aggregation_window):
                # 窗口过期，发送聚合告警
                aggregated = self._create_aggregated_alert(group_key)
                self._reset_group(group_key)
                self.alert_groups[group_key].append(alert)
                self._update_group_metadata(group_key, alert)
                return aggregated
        
        # 添加到组
        self.alert_groups[group_key].append(alert)
        self._update_group_metadata(group_key, alert)
        
        # 检查组大小
        if len(self.alert_groups[group_key]) >= self.max_group_size:
            aggregated = self._create_aggregated_alert(group_key)
            self._reset_group(group_key)
            return aggregated
        
        return None
    
    def _update_group_metadata(self, group_key: str, alert: Alert):
        """更新组元数据"""
        if group_key not in self.group_metadata:
            self.group_metadata[group_key] = {
                'first_alert': alert.created_at,
                'last_update': alert.created_at,
                'count': 0,
                'sources': set(),
                'max_severity': alert.severity
            }
        
        meta = self.group_metadata[group_key]
        meta['last_update'] = alert.created_at
        meta['count'] += 1
        meta['sources'].add(alert.source)
        if alert.severity.value > meta['max_severity'].value:
            meta['max_severity'] = alert.severity
    
    def _reset_group(self, group_key: str):
        """重置组"""
        self.alert_groups[group_key] = []
        if group_key in self.group_metadata:
            del self.group_metadata[group_key]
    
    def _create_aggregated_alert(self, group_key: str) -> Alert:
        """创建聚合告警"""
        alerts = self.alert_groups[group_key]
        meta = self.group_metadata[group_key]
        
        # 提取关键信息
        sources = list(meta['sources'])
        max_severity = meta['max_severity']
        count = meta['count']
        
        # 创建聚合告警
        aggregated = Alert(
            alert_id=f"agg_{uuid.uuid4().hex[:8]}",
            title=f"聚合告警: {group_key} ({count} 个相关告警)",
            description=f"检测到 {count} 个来自 {', '.join(sources[:3])} 的告警",
            source=group_key.split(':')[0],
            severity=max_severity,
            status=AlertStatus.NEW,
            created_at=datetime.now(),
            tags=['aggregated', 'auto_generated'],
            metadata={
                'aggregated_count': count,
                'aggregated_sources': sources,
                'aggregation_window_start': meta['first_alert'].isoformat(),
                'original_alerts': [a.alert_id for a in alerts[:10]]
            }
        )
        
        logger.info(f"创建聚合告警: {aggregated.alert_id} (包含 {count} 个告警)")
        return aggregated
    
    def flush_all(self) -> List[Alert]:
        """刷新所有聚合组"""
        aggregated = []
        for group_key in list(self.alert_groups.keys()):
            if self.alert_groups[group_key]:
                aggregated.append(self._create_aggregated_alert(group_key))
                self._reset_group(group_key)
        return aggregated


class SmartAlertRouter:
    """智能告警路由"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 组件
        self.aggregator = AlertAggregator(self.config.get('aggregation', {}))
        
        # 数据存储
        self.alerts: Dict[str, Alert] = {}
        self.routing_rules: List[RoutingRule] = []
        self.schedules: Dict[str, OnCallSchedule] = {}
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # 抑制记录
        self.suppression_log: Dict[str, datetime] = {}
        
        # 升级跟踪
        self.escalation_timers: Dict[str, asyncio.Task] = {}
        
        logger.info("智能告警路由初始化完成")
    
    def register_notification_handler(self, channel: NotificationChannel, 
                                      handler: Callable[[Alert, List[str]], None]):
        """注册通知处理器"""
        self.notification_handlers[channel] = handler
        logger.info(f"通知处理器注册: {channel.value}")
    
    def add_routing_rule(self, rule: RoutingRule):
        """添加路由规则"""
        self.routing_rules.append(rule)
        # 按优先级排序
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"路由规则添加: {rule.name}")
    
    def add_schedule(self, schedule: OnCallSchedule):
        """添加值班安排"""
        self.schedules[schedule.schedule_id] = schedule
        logger.info(f"值班安排添加: {schedule.name}")
    
    async def route_alert(self, alert: Alert) -> Dict[str, Any]:
        """路由告警"""
        # 如果传入的是字典，转换为 Alert 对象
        if isinstance(alert, dict):
            # 处理 severity 大小写
            severity_str = alert.get('severity', 'warning').upper()
            try:
                severity = AlertSeverity[severity_str]
            except KeyError:
                severity = AlertSeverity.MEDIUM
            
            # 处理 status 大小写
            status_str = alert.get('status', 'new').upper()
            try:
                status = AlertStatus[status_str]
            except KeyError:
                status = AlertStatus.NEW
            
            alert = Alert(
                alert_id=alert.get('alert_id', str(uuid.uuid4())),
                title=alert.get('title', alert.get('name', 'Unknown')),
                description=alert.get('description', alert.get('message', '')),
                source=alert.get('source', 'unknown'),
                severity=severity,
                status=status,
                created_at=alert.get('created_at', datetime.now()),
                tags=alert.get('tags', []),
                metadata=alert.get('metadata', {})
            )
        
        # 存储告警
        self.alerts[alert.alert_id] = alert
        
        # 聚合检查
        aggregated = await self.aggregator.add_alert(alert)
        if aggregated:
            # 发送聚合告警
            return await self._process_alert(aggregated)
        
        # 检查是否需要立即发送（基于配置）
        if self.config.get('immediate_delivery', True):
            return await self._process_alert(alert)
        
        return {'alert_id': alert.alert_id, 'status': 'queued'}
    
    async def _process_alert(self, alert: Alert) -> Dict[str, Any]:
        """处理告警"""
        results = {
            'alert_id': alert.alert_id,
            'routed_to': [],
            'suppressed': False,
            'escalation_scheduled': False
        }
        
        # 匹配路由规则
        matched_rules = self._match_rules(alert)
        
        if not matched_rules:
            # 使用默认路由
            await self._send_to_default(alert)
            results['routed_to'].append('default')
            return results
        
        for rule in matched_rules:
            # 检查抑制
            if rule.suppress_duplicates:
                if self._is_suppressed(alert, rule):
                    results['suppressed'] = True
                    alert.status = AlertStatus.SUPPRESSED
                    logger.info(f"告警被抑制: {alert.alert_id} (规则: {rule.name})")
                    continue
            
            # 发送通知
            for channel in rule.channels:
                await self._send_notification(alert, channel, rule.recipients)
                results['routed_to'].append(f"{channel.value}:{','.join(rule.recipients)}")
            
            # 记录抑制
            if rule.suppress_duplicates:
                self._record_suppression(alert, rule)
            
            # 设置升级定时器
            if rule.escalate_after_minutes and rule.escalate_to:
                await self._schedule_escalation(alert, rule)
                results['escalation_scheduled'] = True
        
        return results
    
    def _match_rules(self, alert: Alert) -> List[RoutingRule]:
        """匹配路由规则"""
        matched = []
        for rule in self.routing_rules:
            if not rule.enabled:
                continue
            try:
                if rule.condition(alert):
                    matched.append(rule)
            except Exception as e:
                logger.warning(f"规则匹配失败 {rule.name}: {e}")
        return matched
    
    def _is_suppressed(self, alert: Alert, rule: RoutingRule) -> bool:
        """检查是否被抑制"""
        suppression_key = f"{alert.source}:{rule.rule_id}"
        last_suppressed = self.suppression_log.get(suppression_key)
        
        if last_suppressed:
            elapsed = datetime.now() - last_suppressed
            if elapsed < timedelta(minutes=rule.suppress_window_minutes):
                return True
        
        return False
    
    def _record_suppression(self, alert: Alert, rule: RoutingRule):
        """记录抑制"""
        suppression_key = f"{alert.source}:{rule.rule_id}"
        self.suppression_log[suppression_key] = datetime.now()
    
    async def _send_notification(self, alert: Alert, 
                                  channel: NotificationChannel,
                                  recipients: List[str]):
        """发送通知"""
        handler = self.notification_handlers.get(channel)
        if not handler:
            logger.warning(f"未找到通知处理器: {channel.value}")
            return
        
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(alert, recipients)
            else:
                handler(alert, recipients)
            
            # 记录通知历史
            alert.notification_history.append({
                'channel': channel.value,
                'recipients': recipients,
                'sent_at': datetime.now().isoformat(),
                'status': 'sent'
            })
            
            logger.info(f"通知发送: {alert.alert_id} -> {channel.value}")
        
        except Exception as e:
            logger.error(f"通知发送失败 {channel.value}: {e}")
            alert.notification_history.append({
                'channel': channel.value,
                'recipients': recipients,
                'sent_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
    
    async def _send_to_default(self, alert: Alert):
        """发送到默认路由"""
        # 获取当前值班人员
        for schedule in self.schedules.values():
            oncall = schedule.get_current_oncall()
            if oncall:
                await self._send_notification(
                    alert, 
                    NotificationChannel.EMAIL, 
                    [oncall]
                )
                alert.assigned_to = oncall
                break
        
        # 记录日志
        if NotificationChannel.LOG in self.notification_handlers:
            await self._send_notification(
                alert,
                NotificationChannel.LOG,
                []
            )
    
    async def _schedule_escalation(self, alert: Alert, rule: RoutingRule):
        """设置升级定时器"""
        async def escalate():
            await asyncio.sleep(rule.escalate_after_minutes * 60)
            
            # 检查告警状态
            if alert.status in [AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED]:
                return
            
            # 执行升级
            alert.escalation_level += 1
            alert.status = AlertStatus.ESCALATED
            
            # 通知升级
            await self._send_notification(
                alert,
                NotificationChannel.PAGERDUTY,
                [rule.escalate_to]
            )
            
            logger.info(f"告警升级: {alert.alert_id} -> {rule.escalate_to}")
        
        # 取消之前的定时器
        if alert.alert_id in self.escalation_timers:
            self.escalation_timers[alert.alert_id].cancel()
        
        # 创建新定时器
        self.escalation_timers[alert.alert_id] = asyncio.create_task(escalate())
    
    async def acknowledge_alert(self, alert_id: str, 
                                user: str) -> bool:
        """确认告警"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.now()
        
        # 取消升级定时器
        if alert_id in self.escalation_timers:
            self.escalation_timers[alert_id].cancel()
            del self.escalation_timers[alert_id]
        
        logger.info(f"告警已确认: {alert_id} by {user}")
        return True
    
    async def resolve_alert(self, alert_id: str, 
                           resolution: str = "") -> bool:
        """解决告警"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.metadata['resolution'] = resolution
        
        # 取消升级定时器
        if alert_id in self.escalation_timers:
            self.escalation_timers[alert_id].cancel()
            del self.escalation_timers[alert_id]
        
        logger.info(f"告警已解决: {alert_id}")
        return True
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        return self.alerts.get(alert_id)
    
    def query_alerts(self, 
                     status: AlertStatus = None,
                     severity: AlertSeverity = None,
                     source: str = None,
                     assigned_to: str = None,
                     limit: int = 100) -> List[Alert]:
        """查询告警"""
        results = list(self.alerts.values())
        
        if status:
            results = [a for a in results if a.status == status]
        if severity:
            results = [a for a in results if a.severity == severity]
        if source:
            results = [a for a in results if a.source == source]
        if assigned_to:
            results = [a for a in results if a.assigned_to == assigned_to]
        
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:limit]
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活动告警"""
        return self.query_alerts(
            status=AlertStatus.NEW,
            limit=1000
        ) + self.query_alerts(
            status=AlertStatus.ACKNOWLEDGED,
            limit=1000
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.alerts)
        
        by_status = defaultdict(int)
        by_severity = defaultdict(int)
        by_source = defaultdict(int)
        
        for alert in self.alerts.values():
            by_status[alert.status.name] += 1
            by_severity[alert.severity.name] += 1
            by_source[alert.source] += 1
        
        # 未确认告警
        unacknowledged = sum(1 for a in self.alerts.values() 
                           if a.status == AlertStatus.NEW)
        
        return {
            'total_alerts': total,
            'by_status': dict(by_status),
            'by_severity': dict(by_severity),
            'by_source': dict(by_source),
            'unacknowledged': unacknowledged,
            'routing_rules': len(self.routing_rules),
            'schedules': len(self.schedules)
        }


def create_default_router(config: Optional[Dict] = None) -> SmartAlertRouter:
    """创建默认路由器"""
    router = SmartAlertRouter(config)
    
    # 添加默认规则：严重告警
    router.add_routing_rule(RoutingRule(
        rule_id="critical_alerts",
        name="严重告警",
        condition=lambda a: a.severity == AlertSeverity.CRITICAL,
        channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK],
        recipients=["oncall", "team-leads"],
        priority=10,
        suppress_duplicates=False,
        escalate_after_minutes=15,
        escalate_to="manager"
    ))
    
    # 添加默认规则：高优先级告警
    router.add_routing_rule(RoutingRule(
        rule_id="high_alerts",
        name="高优先级告警",
        condition=lambda a: a.severity == AlertSeverity.HIGH,
        channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
        recipients=["oncall"],
        priority=8,
        suppress_duplicates=True,
        suppress_window_minutes=10,
        escalate_after_minutes=30,
        escalate_to="team-lead"
    ))
    
    # 添加默认规则：中等告警
    router.add_routing_rule(RoutingRule(
        rule_id="medium_alerts",
        name="中等告警",
        condition=lambda a: a.severity == AlertSeverity.MEDIUM,
        channels=[NotificationChannel.EMAIL],
        recipients=["oncall"],
        priority=5,
        suppress_duplicates=True,
        suppress_window_minutes=30
    ))
    
    # 添加默认规则：低优先级告警
    router.add_routing_rule(RoutingRule(
        rule_id="low_alerts",
        name="低优先级告警",
        condition=lambda a: a.severity in [AlertSeverity.LOW, AlertSeverity.INFO],
        channels=[NotificationChannel.LOG],
        recipients=[],
        priority=1,
        suppress_duplicates=True,
        suppress_window_minutes=60
    ))
    
    return router
