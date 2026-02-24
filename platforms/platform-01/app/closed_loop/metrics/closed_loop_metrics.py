"""
闭环指标
收集和计算闭环系统性能指标
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class ClosedLoopMetrics:
    """闭环指标收集器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 指标存储
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        
        # 配置
        self.retention_hours = self.config.get('retention_hours', 24)
        
        logger.info("闭环指标收集器初始化完成")
    
    def increment_counter(self, name: str, value: float = 1, 
                          labels: Dict[str, str] = None):
        """增加计数器"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        self._record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, 
                  labels: Dict[str, str] = None):
        """设置仪表盘值"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        self._record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_timer(self, name: str, duration_ms: float,
                     labels: Dict[str, str] = None):
        """记录计时器"""
        self._record_metric(name, duration_ms, MetricType.TIMER, labels)
    
    def record_histogram(self, name: str, value: float,
                         labels: Dict[str, str] = None):
        """记录直方图"""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def _record_metric(self, name: str, value: float, 
                       metric_type: MetricType,
                       labels: Dict[str, str] = None):
        """记录指标"""
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        
        # 限制存储大小
        if len(self.metrics[name]) > 10000:
            self.metrics[name] = self.metrics[name][-5000:]
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成键"""
        if not labels:
            return name
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取计数器值"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0)
    
    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取仪表盘值"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0)
    
    def get_histogram_stats(self, name: str, 
                            minutes: int = 60) -> Dict[str, float]:
        """获取直方图统计"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        values = [
            m.value for m in self.metrics.get(name, [])
            if m.timestamp > cutoff
        ]
        
        if not values:
            return {}
        
        import numpy as np
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def get_timer_stats(self, name: str, 
                        minutes: int = 60) -> Dict[str, float]:
        """获取计时器统计"""
        return self.get_histogram_stats(name, minutes)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'metric_names': list(self.metrics.keys())
        }
    
    def cleanup_old_metrics(self):
        """清理过期指标"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        for name in list(self.metrics.keys()):
            self.metrics[name] = [
                m for m in self.metrics[name]
                if m.timestamp > cutoff
            ]


class MTTRTracker:
    """MTTR (Mean Time To Recovery) 追踪器"""
    
    def __init__(self):
        self.incidents: List[Dict] = []
    
    def record_incident(self, incident_id: str, 
                        start_time: datetime,
                        end_time: datetime = None):
        """记录事件"""
        self.incidents.append({
            'incident_id': incident_id,
            'start_time': start_time,
            'end_time': end_time,
            'resolved': end_time is not None
        })
    
    def resolve_incident(self, incident_id: str, 
                         end_time: datetime):
        """解决事件"""
        for incident in self.incidents:
            if incident['incident_id'] == incident_id:
                incident['end_time'] = end_time
                incident['resolved'] = True
                break
    
    def calculate_mttr(self, hours: int = 24) -> Optional[float]:
        """计算MTTR (分钟)"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        resolved = [
            i for i in self.incidents
            if i['resolved'] and i['start_time'] > cutoff
        ]
        
        if not resolved:
            return None
        
        total_minutes = sum(
            (i['end_time'] - i['start_time']).total_seconds() / 60
            for i in resolved
        )
        
        return total_minutes / len(resolved)


class EffectivenessMetrics:
    """有效性指标"""
    
    def __init__(self):
        self.anomaly_count = 0
        self.auto_fixed_count = 0
        self.escalated_count = 0
        self.false_positive_count = 0
    
    def record_detection(self, is_anomaly: bool = True):
        """记录检测"""
        if is_anomaly:
            self.anomaly_count += 1
    
    def record_auto_fix(self, success: bool = True):
        """记录自动修复"""
        if success:
            self.auto_fixed_count += 1
    
    def record_escalation(self):
        """记录升级"""
        self.escalated_count += 1
    
    def record_false_positive(self):
        """记录误报"""
        self.false_positive_count += 1
    
    def get_effectiveness(self) -> Dict[str, float]:
        """获取有效性指标"""
        if self.anomaly_count == 0:
            return {
                'auto_fix_rate': 0.0,
                'escalation_rate': 0.0,
                'false_positive_rate': 0.0
            }
        
        return {
            'auto_fix_rate': self.auto_fixed_count / self.anomaly_count,
            'escalation_rate': self.escalated_count / self.anomaly_count,
            'false_positive_rate': self.false_positive_count / self.anomaly_count
        }


# 便捷函数
def record_detection_latency(metrics: ClosedLoopMetrics, 
                              duration_ms: float):
    """记录检测延迟"""
    metrics.record_timer('detection_latency_ms', duration_ms)


def record_remediation_latency(metrics: ClosedLoopMetrics,
                                duration_ms: float):
    """记录修复延迟"""
    metrics.record_timer('remediation_latency_ms', duration_ms)


def record_full_loop_latency(metrics: ClosedLoopMetrics,
                              duration_ms: float):
    """记录完整闭环延迟"""
    metrics.record_timer('full_loop_latency_ms', duration_ms)
