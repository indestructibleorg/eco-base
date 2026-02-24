"""
事件收集器
收集和标准化各类系统事件用于根因分析
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from collections import defaultdict
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型"""
    ALERT = "alert"                    # 告警事件
    METRIC_ANOMALY = "metric_anomaly"  # 指标异常
    LOG_ERROR = "log_error"            # 日志错误
    DEPLOYMENT = "deployment"          # 部署事件
    CONFIG_CHANGE = "config_change"    # 配置变更
    INFRASTRUCTURE = "infrastructure" # 基础设施事件
    NETWORK = "network"                # 网络事件
    SECURITY = "security"              # 安全事件
    CUSTOM = "custom"                  # 自定义事件


class EventSeverity(Enum):
    """事件严重级别"""
    CRITICAL = 5    # 严重
    HIGH = 4        # 高
    MEDIUM = 3      # 中
    LOW = 2         # 低
    INFO = 1        # 信息


@dataclass
class Event:
    """事件数据类"""
    event_id: str
    event_type: EventType
    source: str                      # 事件来源 (服务/组件名)
    timestamp: datetime
    severity: EventSeverity
    title: str
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None  # 关联ID
    parent_event_id: Optional[str] = None  # 父事件ID
    related_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'raw_data': self.raw_data,
            'tags': self.tags,
            'correlation_id': self.correlation_id,
            'parent_event_id': self.parent_event_id,
            'related_events': self.related_events,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            severity=EventSeverity(data['severity']),
            title=data['title'],
            description=data['description'],
            raw_data=data.get('raw_data', {}),
            tags=data.get('tags', []),
            correlation_id=data.get('correlation_id'),
            parent_event_id=data.get('parent_event_id'),
            related_events=data.get('related_events', []),
            metadata=data.get('metadata', {})
        )


@dataclass
class EventBatch:
    """事件批次"""
    batch_id: str
    events: List[Event]
    start_time: datetime
    end_time: datetime
    source_filter: Optional[str] = None
    type_filter: Optional[EventType] = None


class EventCollector:
    """事件收集器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.events: Dict[str, Event] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: List[Callable[[Event], None]] = []
        self.source_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # 配置
        self.retention_hours = self.config.get('retention_hours', 24)
        self.max_events = self.config.get('max_events', 100000)
        self.batch_size = self.config.get('batch_size', 100)
        
        logger.info("事件收集器初始化完成")
    
    def register_handler(self, handler: Callable[[Event], None], 
                        source: str = None):
        """注册事件处理器"""
        if source:
            self.source_handlers[source].append(handler)
        else:
            self.event_handlers.append(handler)
        logger.info(f"事件处理器注册: {source or 'global'}")
    
    async def collect(self, event: Event) -> str:
        """收集事件"""
        # 存储事件
        self.events[event.event_id] = event
        
        # 加入队列
        await self.event_queue.put(event)
        
        # 触发处理器
        await self._trigger_handlers(event)
        
        logger.debug(f"事件收集: {event.event_id} ({event.event_type.value})")
        return event.event_id
    
    def collect_event(self, event_data: Dict[str, Any]) -> str:
        """同步收集事件 (简化接口)"""
        import asyncio
        
        event = Event(
            event_id=event_data.get('id', str(uuid.uuid4())),
            event_type=EventType(event_data.get('type', 'custom')),
            source=event_data.get('source', 'unknown'),
            timestamp=event_data.get('timestamp', datetime.now()),
            severity=EventSeverity(event_data.get('severity', 3)),
            title=event_data.get('title', 'Unknown Event'),
            description=event_data.get('description', ''),
            raw_data=event_data
        )
        
        # 存储事件
        self.events[event.event_id] = event
        
        # 尝试异步收集
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                asyncio.create_task(self.event_queue.put(event))
        except:
            pass
        
        return event.event_id
    
    async def collect_from_alert(self, alert_data: Dict[str, Any]) -> str:
        """从告警数据收集事件"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ALERT,
            source=alert_data.get('service', 'unknown'),
            timestamp=datetime.now(),
            severity=self._map_severity(alert_data.get('severity', 'medium')),
            title=alert_data.get('title', 'Unknown Alert'),
            description=alert_data.get('description', ''),
            raw_data=alert_data,
            tags=alert_data.get('tags', []),
            metadata={
                'alert_id': alert_data.get('alert_id'),
                'rule_name': alert_data.get('rule_name'),
                'threshold': alert_data.get('threshold'),
                'current_value': alert_data.get('current_value')
            }
        )
        return await self.collect(event)
    
    async def collect_from_metric(self, metric_data: Dict[str, Any]) -> str:
        """从指标数据收集事件"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.METRIC_ANOMALY,
            source=metric_data.get('service', 'unknown'),
            timestamp=datetime.now(),
            severity=self._map_severity(metric_data.get('severity', 'medium')),
            title=f"指标异常: {metric_data.get('metric_name', 'Unknown')}",
            description=metric_data.get('description', ''),
            raw_data=metric_data,
            tags=['metric', 'anomaly'],
            metadata={
                'metric_name': metric_data.get('metric_name'),
                'metric_value': metric_data.get('value'),
                'expected_range': metric_data.get('expected_range'),
                'anomaly_score': metric_data.get('anomaly_score')
            }
        )
        return await self.collect(event)
    
    async def collect_from_log(self, log_data: Dict[str, Any]) -> str:
        """从日志数据收集事件"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.LOG_ERROR,
            source=log_data.get('service', 'unknown'),
            timestamp=datetime.fromisoformat(log_data.get('timestamp', datetime.now().isoformat())),
            severity=self._map_log_level(log_data.get('level', 'error')),
            title=f"日志错误: {log_data.get('error_type', 'Unknown')}",
            description=log_data.get('message', ''),
            raw_data=log_data,
            tags=['log', 'error'],
            metadata={
                'log_level': log_data.get('level'),
                'error_type': log_data.get('error_type'),
                'stack_trace': log_data.get('stack_trace'),
                'request_id': log_data.get('request_id')
            }
        )
        return await self.collect(event)
    
    async def collect_from_deployment(self, deploy_data: Dict[str, Any]) -> str:
        """从部署数据收集事件"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.DEPLOYMENT,
            source=deploy_data.get('service', 'unknown'),
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            title=f"部署: {deploy_data.get('service', 'Unknown')}",
            description=f"版本 {deploy_data.get('version', 'unknown')} 部署{deploy_data.get('status', '')}",
            raw_data=deploy_data,
            tags=['deployment', deploy_data.get('status', 'unknown')],
            metadata={
                'version': deploy_data.get('version'),
                'status': deploy_data.get('status'),
                'deployed_by': deploy_data.get('deployed_by'),
                'rollback_version': deploy_data.get('rollback_version')
            }
        )
        return await self.collect(event)
    
    async def collect_from_config_change(self, config_data: Dict[str, Any]) -> str:
        """从配置变更收集事件"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CONFIG_CHANGE,
            source=config_data.get('service', 'unknown'),
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            title=f"配置变更: {config_data.get('config_key', 'Unknown')}",
            description=f"配置项 {config_data.get('config_key')} 已变更",
            raw_data=config_data,
            tags=['config', 'change'],
            metadata={
                'config_key': config_data.get('config_key'),
                'old_value': config_data.get('old_value'),
                'new_value': config_data.get('new_value'),
                'changed_by': config_data.get('changed_by')
            }
        )
        return await self.collect(event)
    
    async def _trigger_handlers(self, event: Event):
        """触发事件处理器"""
        # 全局处理器
        for handler in self.event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.warning(f"事件处理器失败: {e}")
        
        # 来源特定处理器
        for handler in self.source_handlers.get(event.source, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.warning(f"来源处理器失败: {e}")
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        return self.events.get(event_id)
    
    def query_events(self, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     event_type: Optional[EventType] = None,
                     source: Optional[str] = None,
                     severity: Optional[EventSeverity] = None,
                     tags: List[str] = None,
                     limit: int = 100) -> List[Event]:
        """查询事件"""
        results = list(self.events.values())
        
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if source:
            results = [e for e in results if e.source == source]
        if severity:
            results = [e for e in results if e.severity == severity]
        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]
        
        # 按时间排序
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]
    
    def get_events_batch(self, 
                         start_time: datetime,
                         end_time: datetime,
                         source: Optional[str] = None,
                         event_type: Optional[EventType] = None) -> EventBatch:
        """获取事件批次"""
        events = self.query_events(
            start_time=start_time,
            end_time=end_time,
            source=source,
            event_type=event_type,
            limit=self.batch_size
        )
        
        return EventBatch(
            batch_id=str(uuid.uuid4()),
            events=events,
            start_time=start_time,
            end_time=end_time,
            source_filter=source,
            type_filter=event_type
        )
    
    def get_recent_events(self, minutes: int = 30, 
                          source: Optional[str] = None) -> List[Event]:
        """获取最近事件"""
        start_time = datetime.now() - timedelta(minutes=minutes)
        return self.query_events(start_time=start_time, source=source, limit=1000)
    
    def correlate_events(self, event_id: str, 
                         time_window_minutes: int = 10) -> List[Event]:
        """关联事件"""
        event = self.events.get(event_id)
        if not event:
            return []
        
        start_time = event.timestamp - timedelta(minutes=time_window_minutes)
        end_time = event.timestamp + timedelta(minutes=time_window_minutes)
        
        # 查询时间窗口内的事件
        related = self.query_events(
            start_time=start_time,
            end_time=end_time,
            source=event.source
        )
        
        # 排除自己
        related = [e for e in related if e.event_id != event_id]
        
        return related
    
    def _map_severity(self, severity: str) -> EventSeverity:
        """映射严重级别"""
        mapping = {
            'critical': EventSeverity.CRITICAL,
            'high': EventSeverity.HIGH,
            'warning': EventSeverity.MEDIUM,
            'medium': EventSeverity.MEDIUM,
            'low': EventSeverity.LOW,
            'info': EventSeverity.INFO
        }
        return mapping.get(severity.lower(), EventSeverity.MEDIUM)
    
    def _map_log_level(self, level: str) -> EventSeverity:
        """映射日志级别"""
        mapping = {
            'fatal': EventSeverity.CRITICAL,
            'error': EventSeverity.HIGH,
            'warn': EventSeverity.MEDIUM,
            'warning': EventSeverity.MEDIUM,
            'info': EventSeverity.INFO,
            'debug': EventSeverity.INFO
        }
        return mapping.get(level.lower(), EventSeverity.MEDIUM)
    
    async def start(self):
        """启动收集器"""
        self.running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("事件收集器已启动")
    
    async def stop(self):
        """停止收集器"""
        self.running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("事件收集器已停止")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                await self._cleanup_old_events()
                await asyncio.sleep(3600)  # 每小时清理一次
            except Exception as e:
                logger.exception(f"清理失败: {e}")
    
    async def _cleanup_old_events(self):
        """清理过期事件"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        expired_ids = [
            event_id for event_id, event in self.events.items()
            if event.timestamp < cutoff_time
        ]
        
        for event_id in expired_ids:
            del self.events[event_id]
        
        if expired_ids:
            logger.info(f"清理过期事件: {len(expired_ids)} 个")
        
        # 检查事件数量限制
        if len(self.events) > self.max_events:
            # 删除最旧的事件
            sorted_events = sorted(self.events.items(), key=lambda x: x[1].timestamp)
            to_remove = len(self.events) - self.max_events
            for event_id, _ in sorted_events[:to_remove]:
                del self.events[event_id]
            logger.info(f"清理超额事件: {to_remove} 个")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.events)
        
        # 按类型统计
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_source = defaultdict(int)
        
        for event in self.events.values():
            by_type[event.event_type.value] += 1
            by_severity[event.severity.name] += 1
            by_source[event.source] += 1
        
        # 最近24小时
        last_24h = datetime.now() - timedelta(hours=24)
        recent_count = sum(1 for e in self.events.values() if e.timestamp > last_24h)
        
        return {
            'total_events': total,
            'recent_24h': recent_count,
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_source': dict(by_source),
            'queue_size': self.event_queue.qsize()
        }
