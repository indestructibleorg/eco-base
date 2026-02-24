# =============================================================================
# RCA Event Collector
# =============================================================================
# 根因分析 - 事件收集器
# 收集、标准化、去重和存储异常事件
# =============================================================================

import hashlib
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum, auto
import uuid


class EventSeverity(Enum):
    """事件严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """事件类型"""
    METRIC_ANOMALY = "metric_anomaly"
    LOG_ERROR = "log_error"
    TRACE_SPAN_ERROR = "trace_span_error"
    POD_CRASH = "pod_crash"
    DEPLOYMENT_FAILURE = "deployment_failure"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    PROVIDER_ERROR = "provider_error"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    HIGH_LATENCY = "high_latency"
    HIGH_ERROR_RATE = "high_error_rate"


@dataclass
class Event:
    """
    标准化事件
    
    Attributes:
        id: 唯一标识
        timestamp: 发生时间
        source: 事件来源 (组件名)
        event_type: 事件类型
        severity: 严重级别
        message: 事件消息
        attributes: 附加属性
        correlation_id: 关联ID (用于分组相关事件)
        fingerprint: 事件指纹 (用于去重)
    """
    id: str
    timestamp: datetime
    source: str
    event_type: EventType
    severity: EventSeverity
    message: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """生成事件指纹"""
        # 基于事件类型、来源和关键属性生成指纹
        key_parts = [
            self.event_type.value,
            self.source,
            self.attributes.get("metric_name", ""),
            self.attributes.get("error_type", ""),
            self.attributes.get("service", ""),
        ]
        key = "|".join(filter(None, key_parts))
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "attributes": self.attributes,
            "correlation_id": self.correlation_id,
            "fingerprint": self.fingerprint,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """从字典创建"""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            event_type=EventType(data["event_type"]),
            severity=EventSeverity(data["severity"]),
            message=data["message"],
            attributes=data.get("attributes", {}),
            correlation_id=data.get("correlation_id"),
            fingerprint=data.get("fingerprint"),
        )


@dataclass
class EventGroup:
    """事件组 (相关事件的集合)"""
    id: str
    events: List[Event]
    pattern: str  # 聚合模式描述
    first_seen: datetime
    last_seen: datetime
    
    @property
    def count(self) -> int:
        return len(self.events)
    
    @property
    def severity(self) -> EventSeverity:
        """返回组内最高严重级别"""
        severity_order = [
            EventSeverity.INFO,
            EventSeverity.WARNING,
            EventSeverity.ERROR,
            EventSeverity.CRITICAL,
        ]
        max_index = max(
            severity_order.index(e.severity) for e in self.events
        )
        return severity_order[max_index]
    
    def add_event(self, event: Event) -> None:
        """添加事件到组"""
        self.events.append(event)
        self.last_seen = event.timestamp


class EventCollector:
    """
    事件收集器
    
    负责收集、标准化、去重和存储异常事件
    """
    
    def __init__(self, max_history: int = 10000):
        self._events: List[Event] = []
        self._event_groups: Dict[str, EventGroup] = {}
        self._fingerprint_index: Dict[str, List[Event]] = defaultdict(list)
        self._max_history = max_history
        self._event_handlers: List[Callable[[Event], None]] = []
    
    def register_handler(self, handler: Callable[[Event], None]) -> None:
        """注册事件处理器"""
        self._event_handlers.append(handler)
    
    def collect(self, event: Event) -> bool:
        """
        收集事件
        
        Returns:
            True 如果是新事件，False 如果是重复事件
        """
        # 检查重复
        if self._is_duplicate(event):
            return False
        
        # 存储事件
        self._events.append(event)
        self._fingerprint_index[event.fingerprint].append(event)
        
        # 限制历史大小
        if len(self._events) > self._max_history:
            removed = self._events.pop(0)
            self._fingerprint_index[removed.fingerprint].remove(removed)
        
        # 触发处理器
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Event handler failed: {e}")
        
        return True
    
    def collect_from_anomaly(
        self,
        anomaly_id: str,
        metric_name: str,
        severity: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """从异常检测结果创建事件"""
        event = Event(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            timestamp=timestamp or datetime.utcnow(),
            source="anomaly_detector",
            event_type=EventType.METRIC_ANOMALY,
            severity=EventSeverity(severity),
            message=f"Anomaly detected in {metric_name}: {value}",
            attributes={
                "anomaly_id": anomaly_id,
                "metric_name": metric_name,
                "value": value,
                **(metadata or {}),
            },
        )
        self.collect(event)
        return event
    
    def collect_from_log(
        self,
        source: str,
        message: str,
        severity: str,
        error_type: Optional[str] = None,
        stack_trace: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Event:
        """从日志创建事件"""
        event = Event(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            timestamp=timestamp or datetime.utcnow(),
            source=source,
            event_type=EventType.LOG_ERROR,
            severity=EventSeverity(severity),
            message=message,
            attributes={
                "error_type": error_type,
                "stack_trace": stack_trace,
            },
        )
        self.collect(event)
        return event
    
    def collect_from_trace(
        self,
        trace_id: str,
        span_id: str,
        service: str,
        operation: str,
        error_message: str,
        duration_ms: float,
        timestamp: Optional[datetime] = None,
    ) -> Event:
        """从追踪数据创建事件"""
        event = Event(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            timestamp=timestamp or datetime.utcnow(),
            source=service,
            event_type=EventType.TRACE_SPAN_ERROR,
            severity=EventSeverity.ERROR,
            message=f"Error in {operation}: {error_message}",
            attributes={
                "trace_id": trace_id,
                "span_id": span_id,
                "operation": operation,
                "duration_ms": duration_ms,
            },
        )
        self.collect(event)
        return event
    
    def _is_duplicate(self, event: Event, window_seconds: int = 300) -> bool:
        """检查事件是否重复"""
        similar_events = self._fingerprint_index.get(event.fingerprint, [])
        
        for existing in similar_events:
            time_diff = abs((event.timestamp - existing.timestamp).total_seconds())
            if time_diff < window_seconds:
                return True
        
        return False
    
    def find_duplicates(
        self,
        event: Event,
        window_seconds: int = 300
    ) -> List[Event]:
        """查找相似事件"""
        similar_events = self._fingerprint_index.get(event.fingerprint, [])
        
        return [
            e for e in similar_events
            if abs((event.timestamp - e.timestamp).total_seconds()) < window_seconds
        ]
    
    def get_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
        severity: Optional[EventSeverity] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """查询事件"""
        events = self._events
        
        if start:
            events = [e for e in events if e.timestamp >= start]
        if end:
            events = [e for e in events if e.timestamp <= end]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]
        if source:
            events = [e for e in events if e.source == source]
        
        return events[-limit:]
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """获取单个事件"""
        for event in self._events:
            if event.id == event_id:
                return event
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_events": len(self._events),
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int),
            "by_source": defaultdict(int),
            "time_range": None,
        }
        
        for event in self._events:
            stats["by_type"][event.event_type.value] += 1
            stats["by_severity"][event.severity.value] += 1
            stats["by_source"][event.source] += 1
        
        if self._events:
            timestamps = [e.timestamp for e in self._events]
            stats["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
            }
        
        return dict(stats)
    
    def clear(self) -> None:
        """清空所有事件"""
        self._events.clear()
        self._event_groups.clear()
        self._fingerprint_index.clear()
    
    def export_to_json(self) -> str:
        """导出为 JSON"""
        events_data = [e.to_dict() for e in self._events]
        return json.dumps(events_data, indent=2)
    
    def import_from_json(self, json_str: str) -> int:
        """从 JSON 导入"""
        events_data = json.loads(json_str)
        count = 0
        for data in events_data:
            event = Event.from_dict(data)
            if self.collect(event):
                count += 1
        return count


# 全局事件收集器实例
event_collector = EventCollector()
