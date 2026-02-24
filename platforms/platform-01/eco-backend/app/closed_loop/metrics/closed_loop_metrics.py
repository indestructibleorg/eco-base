# =============================================================================
# Closed Loop Metrics
# =============================================================================
# 閉環系統指標 - Phase 1 基礎閉環核心組件
# 監控閉環系統自身的效能指標
# =============================================================================

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import time

from app.core.logging import get_logger

logger = get_logger("closed_loop_metrics")


# =============================================================================
# Prometheus 風格指標
# =============================================================================

class Counter:
    """計數器指標"""
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, int] = {}
    
    def inc(self, amount: int = 1, **label_values) -> None:
        """增加計數"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount
    
    def get(self, **label_values) -> int:
        """獲取當前值"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)
    
    def reset(self) -> None:
        """重置計數器"""
        self._values = {}


class Gauge:
    """儀表指標"""
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}
    
    def set(self, value: float, **label_values) -> None:
        """設置值"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = value
    
    def inc(self, amount: float = 1, **label_values) -> None:
        """增加值"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount
    
    def dec(self, amount: float = 1, **label_values) -> None:
        """減少值"""
        self.inc(-amount, **label_values)
    
    def get(self, **label_values) -> float:
        """獲取當前值"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)


class Histogram:
    """直方圖指標"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self._values: Dict[tuple, List[float]] = {}
    
    def observe(self, value: float, **label_values) -> None:
        """記錄觀察值"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)
    
    def get_stats(self, **label_values) -> Dict[str, float]:
        """獲取統計信息"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        values = self._values.get(key, [])
        
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }
    
    def get_bucket_counts(self, **label_values) -> Dict[str, int]:
        """獲取桶計數"""
        key = tuple(label_values.get(l, "") for l in self.labels)
        values = self._values.get(key, [])
        
        counts = {f"le_{b}": 0 for b in self.buckets}
        counts["le_inf"] = 0
        
        for value in values:
            counts["le_inf"] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    counts[f"le_{bucket}"] += 1
        
        return counts


# =============================================================================
# 閉環系統指標
# =============================================================================

class ClosedLoopMetrics:
    """
    閉環系統指標收集器
    
    追蹤閉環系統的效能指標
    """
    
    def __init__(self):
        # 異常檢測指標
        self.anomaly_detections_total = Counter(
            "closed_loop_anomaly_detections_total",
            "Total number of anomaly detections",
            ["metric_name", "anomaly_type", "severity"]
        )
        
        self.anomaly_detection_latency_seconds = Histogram(
            "closed_loop_anomaly_detection_latency_seconds",
            "Latency of anomaly detection",
            ["metric_name"]
        )
        
        # 自動修復指標
        self.remediations_total = Counter(
            "closed_loop_remediations_total",
            "Total number of remediation actions",
            ["action_type", "status"]
        )
        
        self.remediation_latency_seconds = Histogram(
            "closed_loop_remediation_latency_seconds",
            "Latency of remediation actions",
            ["action_type"]
        )
        
        self.remediation_success_rate = Gauge(
            "closed_loop_remediation_success_rate",
            "Success rate of remediation actions",
            ["action_type"]
        )
        
        # 規則引擎指標
        self.rule_evaluations_total = Counter(
            "closed_loop_rule_evaluations_total",
            "Total number of rule evaluations",
            ["rule_name", "result"]
        )
        
        self.rule_triggers_total = Counter(
            "closed_loop_rule_triggers_total",
            "Total number of rule triggers",
            ["rule_name"]
        )
        
        self.rule_evaluation_latency_seconds = Histogram(
            "closed_loop_rule_evaluation_latency_seconds",
            "Latency of rule evaluations",
            ["rule_name"]
        )
        
        # 決策指標
        self.decisions_total = Counter(
            "closed_loop_decisions_total",
            "Total number of decisions made",
            ["decision_type"]
        )
        
        self.decision_latency_seconds = Histogram(
            "closed_loop_decision_latency_seconds",
            "Latency of decision making"
        )
        
        # 整體效能指標
        self.mttd_seconds = Gauge(
            "closed_loop_mttd_seconds",
            "Mean Time To Detection"
        )
        
        self.mttr_seconds = Gauge(
            "closed_loop_mttr_seconds",
            "Mean Time To Remediation"
        )
        
        self.closed_loop_active = Gauge(
            "closed_loop_active",
            "Whether closed loop is active (1) or not (0)"
        )
        
        # 歷史數據
        self._detection_history: deque = deque(maxlen=1000)
        self._remediation_history: deque = deque(maxlen=1000)
        self._start_time: Optional[datetime] = None
    
    def record_anomaly_detection(
        self,
        metric_name: str,
        anomaly_type: str,
        severity: str,
        latency_seconds: float
    ) -> None:
        """記錄異常檢測"""
        self.anomaly_detections_total.inc(
            metric_name=metric_name,
            anomaly_type=anomaly_type,
            severity=severity
        )
        
        self.anomaly_detection_latency_seconds.observe(
            latency_seconds,
            metric_name=metric_name
        )
        
        self._detection_history.append({
            "timestamp": datetime.utcnow(),
            "metric_name": metric_name,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "latency_seconds": latency_seconds
        })
    
    def record_remediation(
        self,
        action_type: str,
        status: str,
        latency_seconds: float
    ) -> None:
        """記錄自動修復"""
        self.remediations_total.inc(
            action_type=action_type,
            status=status
        )
        
        self.remediation_latency_seconds.observe(
            latency_seconds,
            action_type=action_type
        )
        
        self._remediation_history.append({
            "timestamp": datetime.utcnow(),
            "action_type": action_type,
            "status": status,
            "latency_seconds": latency_seconds
        })
        
        # 更新成功率
        self._update_success_rate(action_type)
    
    def _update_success_rate(self, action_type: str) -> None:
        """更新成功率"""
        total = self.remediations_total.get(action_type=action_type, status="success") + \
                self.remediations_total.get(action_type=action_type, status="failed")
        
        if total > 0:
            success = self.remediations_total.get(action_type=action_type, status="success")
            rate = success / total
            self.remediation_success_rate.set(rate, action_type=action_type)
    
    def record_rule_evaluation(
        self,
        rule_name: str,
        triggered: bool,
        latency_seconds: float
    ) -> None:
        """記錄規則評估"""
        result = "triggered" if triggered else "not_triggered"
        
        self.rule_evaluations_total.inc(
            rule_name=rule_name,
            result=result
        )
        
        self.rule_evaluation_latency_seconds.observe(
            latency_seconds,
            rule_name=rule_name
        )
        
        if triggered:
            self.rule_triggers_total.inc(rule_name=rule_name)
    
    def record_decision(
        self,
        decision_type: str,
        latency_seconds: float
    ) -> None:
        """記錄決策"""
        self.decisions_total.inc(decision_type=decision_type)
        self.decision_latency_seconds.observe(latency_seconds)
    
    def update_mttd(self, mttd_seconds: float) -> None:
        """更新 MTTD"""
        self.mttd_seconds.set(mttd_seconds)
    
    def update_mttr(self, mttr_seconds: float) -> None:
        """更新 MTTR"""
        self.mttr_seconds.set(mttr_seconds)
    
    def set_active(self, active: bool) -> None:
        """設置閉環狀態"""
        self.closed_loop_active.set(1 if active else 0)
        
        if active and self._start_time is None:
            self._start_time = datetime.utcnow()
    
    def calculate_mttd(self, window_minutes: int = 60) -> float:
        """
        計算 MTTD (Mean Time To Detection)
        
        在指定時間窗口內的平均檢測時間
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        recent_detections = [
            d for d in self._detection_history
            if d["timestamp"] > cutoff
        ]
        
        if not recent_detections:
            return 0.0
        
        # 這裡簡化計算，實際應該基於異常發生時間
        latencies = [d["latency_seconds"] for d in recent_detections]
        return sum(latencies) / len(latencies)
    
    def calculate_mttr(self, window_minutes: int = 60) -> float:
        """
        計算 MTTR (Mean Time To Remediation)
        
        在指定時間窗口內的平均修復時間
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        recent_remediations = [
            r for r in self._remediation_history
            if r["timestamp"] > cutoff and r["status"] == "success"
        ]
        
        if not recent_remediations:
            return 0.0
        
        latencies = [r["latency_seconds"] for r in recent_remediations]
        return sum(latencies) / len(latencies)
    
    def get_summary(self) -> Dict[str, Any]:
        """獲取指標摘要"""
        return {
            "anomaly_detections": {
                "total": sum(
                    self.anomaly_detections_total._values.values()
                ) if self.anomaly_detections_total._values else 0,
                "by_severity": {
                    "critical": self.anomaly_detections_total.get(severity="critical"),
                    "warning": self.anomaly_detections_total.get(severity="warning"),
                    "info": self.anomaly_detections_total.get(severity="info")
                }
            },
            "remediations": {
                "total": sum(
                    self.remediations_total._values.values()
                ) if self.remediations_total._values else 0,
                "success_rate": self.remediation_success_rate.get()
            },
            "rule_evaluations": {
                "total": sum(
                    self.rule_evaluations_total._values.values()
                ) if self.rule_evaluations_total._values else 0,
                "triggers": sum(
                    self.rule_triggers_total._values.values()
                ) if self.rule_triggers_total._values else 0
            },
            "performance": {
                "mttd_seconds": self.calculate_mttd(),
                "mttr_seconds": self.calculate_mttr(),
                "avg_detection_latency": self._calculate_avg_latency(
                    self.anomaly_detection_latency_seconds
                ),
                "avg_remediation_latency": self._calculate_avg_latency(
                    self.remediation_latency_seconds
                )
            },
            "status": {
                "active": self.closed_loop_active.get() == 1,
                "uptime_seconds": self._calculate_uptime()
            }
        }
    
    def _calculate_avg_latency(self, histogram: Histogram) -> float:
        """計算平均延遲"""
        total_count = 0
        total_sum = 0.0
        
        for key, values in histogram._values.items():
            total_count += len(values)
            total_sum += sum(values)
        
        return total_sum / total_count if total_count > 0 else 0.0
    
    def _calculate_uptime(self) -> float:
        """計算運行時間"""
        if self._start_time:
            return (datetime.utcnow() - self._start_time).total_seconds()
        return 0.0
    
    def export_prometheus_format(self) -> str:
        """導出 Prometheus 格式指標"""
        lines = []
        
        # 異常檢測指標
        lines.append("# HELP closed_loop_anomaly_detections_total Total number of anomaly detections")
        lines.append("# TYPE closed_loop_anomaly_detections_total counter")
        for (metric, anomaly_type, severity), value in self.anomaly_detections_total._values.items():
            lines.append(
                f'closed_loop_anomaly_detections_total{{'
                f'metric_name="{metric}",anomaly_type="{anomaly_type}",severity="{severity}"'
                f'}} {value}'
            )
        
        # 修復指標
        lines.append("# HELP closed_loop_remediations_total Total number of remediation actions")
        lines.append("# TYPE closed_loop_remediations_total counter")
        for (action_type, status), value in self.remediations_total._values.items():
            lines.append(
                f'closed_loop_remediations_total{{'
                f'action_type="{action_type}",status="{status}"'
                f'}} {value}'
            )
        
        # 規則評估指標
        lines.append("# HELP closed_loop_rule_evaluations_total Total number of rule evaluations")
        lines.append("# TYPE closed_loop_rule_evaluations_total counter")
        for (rule_name, result), value in self.rule_evaluations_total._values.items():
            lines.append(
                f'closed_loop_rule_evaluations_total{{'
                f'rule_name="{rule_name}",result="{result}"'
                f'}} {value}'
            )
        
        # MTTD/MTTR
        lines.append("# HELP closed_loop_mttd_seconds Mean Time To Detection")
        lines.append("# TYPE closed_loop_mttd_seconds gauge")
        lines.append(f"closed_loop_mttd_seconds {self.mttd_seconds.get()}")
        
        lines.append("# HELP closed_loop_mttr_seconds Mean Time To Remediation")
        lines.append("# TYPE closed_loop_mttr_seconds gauge")
        lines.append(f"closed_loop_mttr_seconds {self.mttr_seconds.get()}")
        
        # 活躍狀態
        lines.append("# HELP closed_loop_active Whether closed loop is active")
        lines.append("# TYPE closed_loop_active gauge")
        lines.append(f"closed_loop_active {self.closed_loop_active.get()}")
        
        return "\n".join(lines)


# 全局閉環指標實例
closed_loop_metrics = ClosedLoopMetrics()
